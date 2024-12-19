import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import glob
from scipy.integrate import odeint
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
from PyFoam.Execution.BasicRunner import BasicRunner
import logging
import sys
from scipy.integrate import solve_bvp
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
import glob
from tensorflow.keras.models import load_model
# ログの設定
logging.basicConfig(filename='script.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

duringtraining_directory = "data/data-wallshear-save"

class PeriodicPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(PeriodicPadding2D, self).__init__(**kwargs)
        self.padding = self._normalize_padding(padding)

    def _normalize_padding(self, padding):
        if isinstance(padding, int):
            return ((padding, padding), (padding, padding))
        elif isinstance(padding, (list, tuple)):
            if len(padding) != 2:
                raise ValueError('Padding must be a list or tuple of length 2.')
            pad_width = self._convert_to_pair(padding[0])
            pad_height = self._convert_to_pair(padding[1])
            return (pad_width, pad_height)
        else:
            raise ValueError('Padding must be an int or a list/tuple of two ints.')

    def _convert_to_pair(self, value):
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return tuple(value)
        else:
            raise ValueError('Padding values must be int or list/tuple of two ints.')

    def call(self, inputs):
        # x（幅）方向のパディング
        pad_width = self.padding[0]
        left_pad = inputs[:, :, -pad_width[0]:, :]
        right_pad = inputs[:, :, :pad_width[1], :]
        x_padded = tf.concat([left_pad, inputs, right_pad], axis=2)

        # z（高さ）方向のパディング
        pad_height = self.padding[1]
        top_pad = x_padded[:, -pad_height[0]:, :, :]
        bottom_pad = x_padded[:, :pad_height[1], :, :]
        xz_padded = tf.concat([top_pad, x_padded, bottom_pad], axis=1)

        return xz_padded

    def get_config(self):
        config = super(PeriodicPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config

# Generatorの構築
def build_cnn_generator(input_shape=(32, 32, 6)):
    model = models.Sequential(name="Generator")
    model.add(layers.Input(shape=input_shape))

    model.add(PeriodicPadding2D(padding=(1, 1)))
    model.add(layers.Conv2D(16, (3, 3), padding='valid'))
    model.add(layers.ReLU())

    model.add(PeriodicPadding2D(padding=(1, 1)))
    model.add(layers.Conv2D(32, (3, 3), padding='valid'))
    model.add(layers.ReLU())

    model.add(PeriodicPadding2D(padding=(1, 1)))
    model.add(layers.Conv2D(32, (3, 3), padding='valid'))
    model.add(layers.ReLU())

    model.add(PeriodicPadding2D(padding=(1, 1)))
    model.add(layers.Conv2D(64, (3, 3), padding='valid'))
    model.add(layers.ReLU())

    model.add(PeriodicPadding2D(padding=(1, 1)))
    model.add(layers.Conv2D(64, (3, 3), padding='valid'))
    model.add(layers.ReLU())

    model.add(PeriodicPadding2D(padding=(1, 1)))
    model.add(layers.Conv2D(2, (3, 3), padding='valid'))

    return model

def load_discriminator():
    model = models.Sequential(name="Discriminator")
    model.add(layers.Input(shape=(32, 32, 2)))

    # 畳み込み + プーリング層
    model.add(tfa.layers.SpectralNormalization(layers.Conv2D(16, (3, 3), padding='same')))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 16x16x16 に次元削減

    model.add(tfa.layers.SpectralNormalization(layers.Conv2D(32, (3, 3), padding='same')))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 8x8x32 に次元削減

    model.add(tfa.layers.SpectralNormalization(layers.Conv2D(64, (3, 3), padding='same')))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 4x4x64 に次元削減

    model.add(tfa.layers.SpectralNormalization(layers.Conv2D(64, (3, 3), padding='same')))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 4x4x64 に次元削減

    # Flatten + 全結合層
    model.add(layers.Flatten())  # 次元削減後のデータを1次元に変換
    model.add(layers.Dense(8, activation='relu'))  # 隠れ層
    model.add(layers.Dense(1, activation='sigmoid'))  # 出力層
    
    print("Discriminator model created from scratch")
    return model

# 保存するための関数
def save_generated_bin_data(index, generated_data, directory, time_per_step):
    index = index / time_per_step
    # Remove .0 for integers by converting to an integer if it's whole, else keep it as a float
    if index.is_integer():
        index_str = f"{int(index)}"
    else:
        index_str = f"{index:.0f}"  # Keep zero decimal places for non-integers

    filename = os.path.join(directory, f'ux-gen-{index_str}.bin')
    generated_data = generated_data.astype(np.float64)
    generated_data.tofile(filename)

def delete_bin_files(directory):
    bin_files = glob.glob(os.path.join(directory, '*.bin'))
    for file_path in bin_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# 速度場を読み取る関数
def get_velocities(case, time_step):
    U_file_path = os.path.join(case, time_step, "U")
    logging.debug(f'U_file_path: {U_file_path}')
    
    # ファイルが存在しない場合、末尾の余計なゼロを省略した形でファイルを探す
    if not os.path.exists(U_file_path):
        # 例: "4.40" -> "4.4"
        alternative_time_step = time_step.rstrip("0").rstrip(".")
        U_file_path = os.path.join(case, alternative_time_step, "U")
        logging.debug(f'Trying alternative U_file_path: {U_file_path}')
    
    # それでも存在しない場合、エラーを表示
    if not os.path.exists(U_file_path):
        logging.error(f"File not found: {U_file_path}")
        raise FileNotFoundError(f"File not found: {U_file_path}")
    
    try:
        U_file = ParsedParameterFile(U_file_path)
        velocities = [Vector(*vec) for vec in U_file["internalField"]]
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise
    
    return velocities

# 速度場のデータを3次元配列に格納する関数
def prepare_velocity_field_fine(velocities):
    nx, ny, nz = 128, 128, 128
    half_ny = 64
    first_hight_fine = 0.00316096
    nu = 0.009
    u_data = []
    w_data = []
    
    i = 0
    
    # 速度データをリストに格納
    for iz in range(nz):
        for iy in range(half_ny):
            for ix in range(nx):
                u_data.append(velocities[i][0])  # u (x方向速度)
                w_data.append(velocities[i][2])  # w (z方向速度)
                i += 1

    # リストをnumpy配列に変換
    u_data = np.array(u_data).reshape((nx, half_ny, nz))  
    w_data = np.array(w_data).reshape((nx, half_ny, nz)) 
    
    # 四つおきにデータを取得し、壁面剪断応力を計算
    u_wallshear_fine = u_data[::4, 1, ::4] / first_hight_fine * nu
    w_wallshear_fine = w_data[::4, 1, ::4] / first_hight_fine * nu

    return u_wallshear_fine, w_wallshear_fine

# 速度場のデータを3次元配列に格納する関数
def prepare_velocity_field_coarse(velocities, nu, zero_hight_coarse):
    nx, ny, nz = 32, 16, 32
    half_ny = ny // 2
    u_data = []
    v_data = []
    w_data = []
    u_data_top = []
    v_data_top = []
    w_data_top = []
    selected_y_indices_bottom = [1,2,3,4]  # y方向のインデックス
    selected_y_indices_top = [6,5,4,3]  # y方向のインデックス
    u_data_coarse_input_bottom = []
    v_data_coarse_input_bottom = []
    w_data_coarse_input_bottom = []
    u_data_coarse_input_top = []
    v_data_coarse_input_top = []
    w_data_coarse_input_top = []
    u_wallshear_bottom = []
    w_wallshear_bottom = []
    u_wallshear_top = []
    w_wallshear_top = []

    i = 0

    # 速度データをリストに格納
    for iz in range(nz):
        for iy in range(0,8):
            for ix in range(nx):
                u_data.append(velocities[i][0])  # u (x方向速度)
                v_data.append(velocities[i][1])  # v (y方向速度)
                w_data.append(velocities[i][2])  # w (z方向速度)
                i += 1
    i = nx * half_ny * nz 
    for iz in range(nz):
        for iy in range(8,16):
            for ix in range(nx):
                u_data_top.append(velocities[i][0])  # u (x方向速度)
                v_data_top.append(velocities[i][1])  # v (y方向速度)
                w_data_top.append(velocities[i][2])  # w (z方向速度)
                i += 1 

    # リストをnumpy配列に変換
    u_data = np.array(u_data).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)
    v_data = np.array(v_data).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)
    w_data = np.array(w_data).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)
    u_data_top = np.array(u_data_top).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)
    v_data_top = np.array(v_data_top).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)
    w_data_top = np.array(w_data_top).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)

    # 壁面近傍の四層のデータを抽出 (32, 32, 4)
    for iy in selected_y_indices_bottom:
        u_data_coarse_input_bottom.append(u_data[:, iy, :])  # Extract y-slice
        v_data_coarse_input_bottom.append(v_data[:, iy, :])  # Extract y-slice
        w_data_coarse_input_bottom.append(w_data[:, iy, :])  # Extract y-slice
    
    # 壁面近傍の四層のデータを抽出 (32, 32, 4)
    for iy in selected_y_indices_top:
        u_data_coarse_input_top.append(u_data_top[:, iy, :])  # Extract y-slice
        v_data_coarse_input_top.append(v_data_top[:, iy, :])  # Extract y-slice
        w_data_coarse_input_top.append(w_data_top[:, iy, :])  # Extract y-slice

    # Stack u and w data from selected y-slices to get shape (32, 32, 4)
    u_data_coarse_input_bottom = np.stack(u_data_coarse_input_bottom, axis=-1)  # Shape: (32, 32, 4)
    v_data_coarse_input_bottom = np.stack(v_data_coarse_input_bottom, axis=-1)  # Shape: (32, 32, 4)
    w_data_coarse_input_bottom = np.stack(w_data_coarse_input_bottom, axis=-1)  # Shape: (32, 32, 4)
    u_data_coarse_input_top = np.stack(u_data_coarse_input_top, axis=-1)  # Shape: (32, 32, 4)
    v_data_coarse_input_top = np.stack(v_data_coarse_input_top, axis=-1)  # Shape: (32, 32, 4)
    w_data_coarse_input_top = np.stack(w_data_coarse_input_top, axis=-1)  # Shape: (32, 32, 4) 
    ave_u_o = np.mean(u_data_coarse_input_bottom[:, :, 0])
    print('u_data_coarse_input_bottom', ave_u_o) 

    # 壁面剪断応力を計算
    u_wallshear_bottom = (u_data[:, 0, :] / zero_hight_coarse * nu).reshape(32, 32, 1)  # Shape: (32, 32, 1)
    w_wallshear_bottom = (w_data[:, 0, :] / zero_hight_coarse * nu).reshape(32, 32, 1)  # Shape: (32, 32, 1)

    u_wallshear_top = (u_data_top[:, 7, :] / zero_hight_coarse * nu).reshape(32, 32, 1)  # Shape: (32, 32, 1)
    w_wallshear_top = (w_data_top[:, 7, :] / zero_hight_coarse * nu).reshape(32, 32, 1)  # Shape: (32, 32, 1)

    return u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom, u_data_coarse_input_top,v_data_coarse_input_top, w_data_coarse_input_top,u_wallshear_bottom, w_wallshear_bottom, u_wallshear_top, w_wallshear_top 


def get_velocities_normal(case, time_step):
    U_file_path = os.path.join(case, time_step, "U")
    logging.debug(f'U_file_path: {U_file_path}')
    
    try:
        U_file = ParsedParameterFile(U_file_path)
        velocities = [Vector(*vec) for vec in U_file["internalField"]]
    except Exception as e:
        raise
    
    return velocities

def get_locations(case, feed_loc):
    points_file_path = os.path.join(case, "constant", "polyMesh", "C")
    
    with open(points_file_path, 'r') as f:
        lines = f.readlines()
        
    points_data = lines[lines.index('(\n') + 1 : lines.index(')\n')]
    
    locations = []
    y_loc = []
    for line in points_data:
        x, y, z = map(float, line.strip('()\n').split())
        locations.append(Vector(x, y, z))
        y_loc.append(y)
    
    y_loc = list(set(y_loc))
    for iy in y_loc:
        if abs(iy - feed_loc) == abs(np.array(y_loc) - feed_loc).min():
            iy_feed = iy

    return locations, iy_feed

def find_iy_feed_index(case, feed_loc):
    locations, iy_feed = get_locations(case, feed_loc)
    
    # y_loc 配列を取得してソート
    y_loc = sorted(set([loc[1] for loc in locations]))
    
    # iy_feed のインデックスを取得
    iy_feed_index = y_loc.index(iy_feed)
    
    return y_loc,iy_feed_index

def calculate_body_forces(current_time, velocities, locations, iy_feed, iy_feed_2, U_0, U_1, time_step_coarse, ny, delta):
    forces = []
    i = 0
    j = 0
    for loc, vel in zip(locations, velocities):
        # Example force calculation using velocity
        if loc[1] == iy_feed:
            force = Vector(delta * U_0[i][0]/time_step_coarse, 0.0, delta * U_0[i][2]/time_step_coarse)
            i = i+1
        elif loc[1] == iy_feed_2:
            force = Vector(delta * U_1[j][0]/time_step_coarse, 0.0, delta * U_1[j][2]/time_step_coarse)
            j = j+1 
        else:
            force = Vector(0.0, 0.0, 0.0)
        forces.append(force)

    return forces

def write_forces_to_case(forces, case, time_step):
    # Define the path for the body force file
    body_force_file_path = os.path.join(case, time_step, "bodyForce")

    # Create the necessary directories if they don't exist
    os.makedirs(os.path.dirname(body_force_file_path), exist_ok=True)
    
    with open(body_force_file_path, 'w') as f:
        # Write the header for the OpenFOAM file
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       volVectorField;\n')
        f.write('    location    "{}";\n'.format(time_step))
        f.write('    object      bodyForce;\n')
        f.write('}\n')
        f.write('dimensions      [0 1 -2 0 0 0 0];\n')
        f.write('internalField   nonuniform List<vector>\n')
        f.write('{}\n'.format(len(forces)))
        f.write('(\n')

        # Write the forces to the file
        for force in forces:
            f.write('({} {} {})\n'.format(force[0], force[1], force[2]))
        
        f.write(');\n')
        
        # Boundary field - Adjust as per your case requirements
        f.write('boundaryField\n')
        f.write('{\n')
        f.write('    bottom\n')
        f.write('    {\n')
        f.write('        type            zeroGradient;\n')
        f.write('    }\n')
        f.write('    top\n')
        f.write('    {\n')
        f.write('        type            zeroGradient;\n')
        f.write('    }\n')
        f.write('    left\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    right\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    inlet\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    outlet\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('}\n')

def standardize_per_channel_global(data):
    """
    data: (N,H,W,C)
    全サンプルについて各チャンネルで平均・stdを計算し、全サンプルを標準化
    戻り値: standardized_data, means, stds
    means, stdsは(チャンネル数,)で保持
    """
    N,H,W,C = data.shape
    means = []
    stds = []
    for c in range(C):
        channel_data = np.copy(data[:, :, :, c])
        channel_data = data[:,:,:,c]
        mean_c = np.mean(channel_data)
        std_c = np.std(channel_data)
        if std_c < 1e-8:
            std_c = 1e-8
        means.append(mean_c)
        stds.append(std_c)
        data[:,:,:,c] = (channel_data - mean_c) / std_c
    means = np.array(means)
    stds = np.array(stds)
    return data, means, stds

def standardize_per_channel_global_basedon_means_stds(data,means,stds):
    """
    data: (N,H,W,C)
    全サンプルについて各チャンネルで平均・stdを計算し、全サンプルを標準化
    戻り値: standardized_data, means, stds
    means, stdsは(チャンネル数,)で保持
    """
    B,H,W,C = data.shape
    for c in range(C):
        data[:,:,:,c] = (data[:,:,:,c] - means[c]) / stds[c]
    return data

def inverse_standardize(data, means, stds, nu, zero_hight_coarse):
    """
    data: (H,W,C)
    means, stds: (C,)
    (data * std) + meanで元に戻す
    """
    #alpha = nu / zero_hight_coarse
    #*nu/zero_hight_coarse
    for c in range(data.shape[2]):
        if c == 0:
            data[:,:,c] = data[:,:,c] * (stds[0]) + (means[0])
        else:
            data[:,:,c] = data[:,:,c] * (stds[1]) + (means[1])
    print('means:',means)
    print('stds:',stds)
    #print('alpha:',alpha)
    return data

def generator_output(case_coarse,time_step_str_coarse,alpha,beta,_,output_directory,U,case,ny,nu,zero_hight_coarse,time_steps_coarse, input_data_directory, coarse_original_data_directory,input_data_directory_top,coarse_original_data_directory_top):
    batch_input_data_list_bottom = []
    batch_input_data_list_top = []
    OUTPUT_DB = np.zeros((32, 32, 4)) 
    custom_objects = {'PeriodicPadding2D': PeriodicPadding2D}
    generator = load_model('generator.h5', custom_objects=custom_objects, compile=False)
    # generator.compile(optimizer='adam', loss='binary_crossentropy') 
    # Get velocity data for the coarse case
    print("time_step_str_coarse",time_step_str_coarse)
    velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)
    u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom, u_data_coarse_input_top,v_data_coarse_input_top, w_data_coarse_input_top,u_wallshear_bottom, w_wallshear_bottom, u_wallshear_top, w_wallshear_top  = prepare_velocity_field_coarse(velocities_coarse, nu, zero_hight_coarse)
    # Combine u and w wall shear stress data
    data_combined = [u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom]
    batch_real_input_data = np.stack(data_combined, axis=-1)
    # Create filename for this time step
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(input_data_directory, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_real_input_data.tofile(output_filepath)

    # Combine u and w wall shear stress data
    data_combined = [u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom]
    batch_real_input_data = np.stack(data_combined, axis=-1)
    # Create filename for this time step
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(input_data_directory_top, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_real_input_data.tofile(output_filepath)

    # Combine u and w wall shear stress data
    data_combined = [u_wallshear_bottom, w_wallshear_bottom]
    batch_real_input_data = np.stack(data_combined, axis=-1)
    # Create filename for this time step
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(coarse_original_data_directory, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_real_input_data.tofile(output_filepath)

    # Combine u and w wall shear stress data
    data_combined = [u_wallshear_top, w_wallshear_top]
    batch_real_input_data = np.stack(data_combined, axis=-1)
    # Create filename for this time step
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(coarse_original_data_directory_top, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_real_input_data.tofile(output_filepath)  

    print(f"Data saved to {output_filepath}") 
    ############################################################################# bottom ############################################################################################
    #################### 1. prepare the data  ####################
    # Combine u, v, and w data (input becomes 32, 32, 8)
    batch_combined_data_bottom = np.concatenate((u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom), axis=-1)
    batch_input_data_list_bottom.append(batch_combined_data_bottom)
    # Select specific slices to reduce to 6 channels (example: first 6 combined channels)
    
    u_data_bottom = batch_combined_data_bottom[:, :, [0, 1, 4, 5, 8, 9]]  # 6 channels #######ここの取り方が違う
    print('u_data_bottom_shape',u_data_bottom.shape)

    #################### 2. save the data in the current time step ####################
    # Convert to NumPy array with shape (batch_size, 32, 32, 6)
    batch_input_data_bottom = np.array(u_data_bottom)  
    # batch_input_data_bottom = np.expand_dims(batch_input_data_bottom, axis=0)  # (1, 32, 32, 6)
    ave_0 = np.mean(batch_input_data_bottom[:, :, 0])
    ave_1 = np.mean(batch_input_data_bottom[:, :, 1]) 
    print('ave_0(流れ) before=',ave_0)
    print('ave_1(流れ) before=',ave_1) 
    """
    # calculate the average value
    batch_vel_input_data_bottom = np.stack(batch_combined_data_bottom, axis=-1)
    # Create filename for this time step
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(input_data_directory, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_vel_input_data_bottom.tofile(output_filepath) 
    # wall shear 
    data_combined = [u_wallshear_bottom, w_wallshear_bottom]
    batch_wallshear_input_data = np.stack(data_combined, axis=-1)
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(coarse_original_data_directory, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_wallshear_input_data.tofile(output_filepath) 
    """

    #################### 3. obatin average value and prepare the input data ####################
    coarse_data, coarse_means, coarse_stds, coarse_wallshear_means, coarse_wallshear_stds = obtain_average(time_steps_coarse,input_data_directory,coarse_original_data_directory)
    # print('time_steps_coarse',time_steps_coarse)
    print('coarse_means',coarse_means)
    print('coarse_stds',coarse_stds)
    print('coarse_wallshear_means',coarse_wallshear_means)
    print('coarse_wallshear_stds',coarse_wallshear_stds)

    # 生成時も標準化
    for c in range(6):
        u_data_bottom[:,:,c] = (u_data_bottom[:,:,c]-coarse_means[c])/coarse_stds[c]
    u_data_bottom = np.expand_dims(u_data_bottom, axis=0) 

    #################### 4. generator prediction ####################
    generated_images_bottom = generator.predict(u_data_bottom)  # Shape: (batch_size, 32, 32, 2)
    generated = generated_images_bottom[0]
    ave_0_original = np.mean(generated[:,:,0])  
    ave_1_original = np.mean(generated[:,:,1])  
    print('ave_0_original',ave_0_original)
    print('ave_1_original',ave_1_original)
    # generated_data_bottom_norm = generated_images_bottom.squeeze()
    print('generated shape', generated.shape)
    generated_data_bottom = inverse_standardize(generated, coarse_wallshear_means, coarse_wallshear_stds, nu, zero_hight_coarse)
    save_generated_bin_data(_, generated_data_bottom, output_directory, 1)

    #################### 5. save data ####################
    # For the first layer (generated_data[:,:,0])
    ave_0 = np.mean(generated_data_bottom[:,:,0])  # Compute the mean of all points
    std_0 = np.std(generated_data_bottom[:,:,0])  # Compute the standard deviation of all points
    print("ave_0_gen_wallshear = ",ave_0)
    # Create the ave and std arrays with the same shape as the grid (32x32)
    ave_0_grid = np.full((32, 32), ave_0)
    std_0_grid = np.full((32, 32), std_0)
    # For the second layer (generated_data[:,:,1])
    ave_1 = np.mean(generated_data_bottom[:,:,1])  # Compute the mean of all points
    std_1 = np.std(generated_data_bottom[:,:,1])  # Compute the standard deviation of all points
    # Create the ave and std arrays with the same shape as the grid (32x32)
    ave_1_grid = np.full((32, 32), ave_1)
    std_1_grid = np.full((32, 32), std_1)
    # Ensure the relationship is satisfied: generated_data = ave + std
    generated_data_bottom[:,:,0] = alpha * ave_0_grid + beta * (generated_data_bottom[:,:,0]-ave_0_grid) 
    generated_data_bottom[:,:,1] = alpha * ave_1_grid + beta * (generated_data_bottom[:,:,1]-ave_1_grid) 
    ave_0_U = np.mean(U[:, 0, :, 0])
    ave_1_U = np.mean(U[:, 0, :, 2])
    ave_0_U_grid = np.full((32, 32), ave_0_U)
    ave_1_U_grid = np.full((32, 32), ave_1_U)
    OUTPUT_DB[:, :, 0] = - U[:, 0, :, 0] + generated_data_bottom[:,:,0] / nu * zero_hight_coarse
    OUTPUT_DB[:, :, 1] = - U[:, 0, :, 2] + generated_data_bottom[:,:,1] / nu * zero_hight_coarse
    print('use the generator and update the wall shear stress at', case)

    ############################################################################# top ############################################################################################
    #################### 1. prepare the data  ####################
    # Concatenate u and w data to form (32, 32, 8) input
    batch_combined_data_top = np.concatenate((u_data_coarse_input_top, v_data_coarse_input_top, w_data_coarse_input_top), axis=-1)
    batch_input_data_list_top.append(batch_combined_data_top)
    # Select specific slices to reduce to 6 channels (example: first 6 combined channels)
    u_data_top = batch_combined_data_top[:, :, [0, 1, 4, 5, 8, 9]]  # 6 channels
    print('u_data_top_shape',u_data_top.shape)
    # Convert to NumPy array with shape (batch_size, 32, 32, 6)
    # batch_input_data_top = np.array(u_data)

    #################### 2. save the data in the current time step ####################
    """
    # calculate the average value
    batch_vel_input_data_top = np.stack(batch_combined_data_top, axis=-1)
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(input_data_directory_top, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_vel_input_data_bottom.tofile(output_filepath) 
    # wall shear 
    data_combined_top = [u_wallshear_top, w_wallshear_top]
    batch_wallshear_input_data_top = np.stack(data_combined_top, axis=-1)
    output_filename = f"U-{time_step_str_coarse}.bin"
    output_filepath = os.path.join(coarse_original_data_directory_top, output_filename)
    # Save the combined data (u and w wall shear stress) as binary file
    batch_wallshear_input_data_top.tofile(output_filepath) 
    """
    #################### 3. obatin average value and prepare the input data ####################
    coarse_data_top, coarse_means_top, coarse_stds_top, coarse_wallshear_means_top, coarse_wallshear_stds_top = obtain_average(time_steps_coarse,input_data_directory_top,coarse_original_data_directory_top)
    print('coarse_means_top',coarse_means_top)
    print('coarse_stds_top',coarse_stds_top)
    print('coarse_wallshear_means_top',coarse_wallshear_means_top)
    print('coarse_wallshear_stds_top',coarse_wallshear_stds_top)
    # 生成時も標準化
    for c in range(u_data_top.shape[2]):
        u_data_top[:,:,c] = (u_data_top[:,:,c]-coarse_means_top[c])/coarse_stds_top[c]
    ave_0_vel = np.mean(u_data_top[:, :, 0])
    ave_1_vel = np.mean(u_data_top[:, :, 1]) 
    print('ave_0(流れ)_top=',ave_0_vel)
    print('ave_1(流れ)_top=',ave_1_vel)
    u_data_top = np.expand_dims(u_data_top, axis=0)

    #################### 4. generator prediction ####################
    generated_images_top = generator.predict(u_data_top)  # Shape: (batch_size, 32, 32, 2)
    generated = generated_images_top[0]
    # generated_data_bottom_norm = generated_images_bottom.squeeze()
    generated_data_top = inverse_standardize(generated, coarse_wallshear_means_top, coarse_wallshear_stds_top, nu, zero_hight_coarse)

    #################### 5. save data ####################
    ave_0_top = np.mean(generated_data_top[:,:,0])  # Compute the mean of all points
    std_0 = np.std(generated_data_top[:,:,0])  # Compute the standard deviation of all points
    # Create the ave and std arrays with the same shape as the grid (32x32)
    ave_0_grid = np.full((32, 32), ave_0_top)
    std_0_grid = np.full((32, 32), std_0)
    # For the second layer (generated_data[:,:,1])
    ave_1_top = np.mean(generated_data_top[:,:,1])  # Compute the mean of all points
    std_1 = np.std(generated_data_top[:,:,1])  # Compute the standard deviation of all points
    # Create the ave and std arrays with the same shape as the grid (32x32)
    ave_1_grid = np.full((32, 32), ave_1_top)
    std_1_grid = np.full((32, 32), std_1)
    # Ensure the relationship is satisfied: generated_data = ave + std
    generated_data_top[:,:,0] = alpha * ave_0_grid + beta * (generated_data_top[:,:,0]-ave_0_grid) 
    generated_data_top[:,:,1] = alpha * ave_1_grid + beta * (generated_data_top[:,:,1]-ave_1_grid) 
    ave_0_U = np.mean(U[:, ny-1, :, 0])
    ave_1_U = np.mean(U[:, ny-1, :, 2])
    ave_0_U_grid = np.full((32, 32), ave_0_U)
    ave_1_U_grid = np.full((32, 32), ave_1_U)
    OUTPUT_DB[:, :, 2] = - U[:, ny-1, :, 0] + generated_data_top[:,:,0] / nu * zero_hight_coarse
    OUTPUT_DB[:, :, 3] = - U[:, ny-1, :, 2] + generated_data_top[:,:,1] / nu * zero_hight_coarse

    with open('wallshear.txt', 'a') as f:  # 'a' モードで開くと加筆される
        f.write(f"{_}, {ave_0}, {ave_1}, {ave_0_top}, {ave_1_top}\n")
    return OUTPUT_DB[:,:,:], ave_0, ave_1 

def obtain_average(time_steps_coarse,coarse_directory,wallshear_directory):
    coarse_data_list = []
    for t in time_steps_coarse:
        t_str = f"{t:.4f}"
        u_coarse_file = os.path.join(coarse_directory, f"U-{t_str}.bin")
        u_data_full = np.fromfile(u_coarse_file, dtype=np.float64).reshape((32, 32, 12))
        u_data = u_data_full[:, :, [0,3,1,4,2,5]]
        coarse_data_list.append(u_data)
    
    coarse_wallshear_data_list = []
    for t in time_steps_coarse:
        t_str = f"{t:.4f}"
        u_wallshear_coarse_file = os.path.join(wallshear_directory, f"U-{t_str}.bin")
        wallshear_data = np.fromfile(u_wallshear_coarse_file, dtype=np.float64).reshape((32, 32, 2))
        coarse_wallshear_data_list.append(wallshear_data)

    coarse_data = np.array(coarse_data_list)  # (N,32,32,3)
    coarse_wallshear_data = np.array(coarse_wallshear_data_list)      # (M,32,32,2)
    num_samples = min(len(coarse_data), len(coarse_wallshear_data))
    coarse_data = coarse_data[:num_samples]
    coarse_wallshear_data = coarse_wallshear_data[:num_samples]

    # チャネルごとにグローバルmean,stdで標準化
    coarse_data, coarse_means, coarse_stds,  = standardize_per_channel_global(coarse_data)
    coarse_wallshear_data, coarse_wallshear_means, coarse_wallshear_stds = standardize_per_channel_global(coarse_wallshear_data)

    return coarse_data, coarse_means, coarse_stds, coarse_wallshear_means, coarse_wallshear_stds 

def main(nu, case_dir, num_steps, save_steps, eed_loc, feed_loc_2, input_loc, feed_steps, first_hight_coarse, zero_hight_coarse, limit_steps, time_step_coarse, a_posteriori_steps, alpha, beta, delta, ny):
    case = SolutionDirectory(case_dir)
    locations, iy_feed = get_locations(case_dir, feed_loc)
    locations2, iy_feed_2 = get_locations(case_dir, feed_loc_2) 
    locations_input, iy_input = get_locations(case_dir, input_loc)
    y_loc,iy_feed_index = find_iy_feed_index(case_dir, feed_loc)
    y_loc,iy_input_index = find_iy_feed_index(case_dir, input_loc)

    current_time = case.getLast() 
    start_step = int(float(current_time) / time_step_coarse) 
    print(y_loc)

    for _ in range(start_step, start_step + num_steps):
        logging.debug(f'Step {_} start')
        current_time = case.getLast()
        logging.debug(f'Current time: {current_time}')
        velocities = get_velocities_normal(case_dir, current_time)
        logging.debug(f'Velocities obtained at step {_}')

        U = np.zeros((32, 16, 32, 3))
        i = 0
        for iz in range(32):
            for iy in range(0,8):
                for ix in range(32):
                    U[ix, iy, iz, 0] = velocities[i][0]
                    U[ix, iy, iz, 1] = velocities[i][1]
                    U[ix, iy, iz, 2] = velocities[i][2]
                    i += 1
        print(i)

        for iz in range(32):
            for iy in range(8,16):
                for ix in range(32):
                    U[ix, iy, iz, 0] = velocities[i][0]
                    U[ix, iy, iz, 1] = velocities[i][1]
                    U[ix, iy, iz, 2] = velocities[i][2]
                    i += 1
        
        logging.debug(f'U array populated at step {_}')               

        U_input = U[:, iy_input_index, :, :]
        #if _ % feed_steps == 0:
        INPUT_DB = np.zeros((32, 32, 3))
        OUTPUT_DB = np.zeros((32, 32, 4))
        #INPUT_DB[:, :, :] = U_input
        case_coarse = "LES_co"  # Coarse simulation case directory
        input_data_directory = "data/data-coarse" 
        real_data_directory = "data/data-fine"  # Fine data directory where binary files are saved
        output_directory = "data/data-wallshear-est"
        coarse_original_data_directory = "data/data-coarse-wallshear"
        input_data_directory_top = "data/data-coarse-top" 
        output_directory_top = "data/data-wallshear-est-top"
        coarse_original_data_directory_top = "data/data-coarse-wallshear-top"

        time_coarse_start = 0
        time_coarse_finish = 0
        previous_steps = 0
        n = 100 # for choosing data from last 2 sec
        time_coarse_start_loop = (_ + previous_steps - a_posteriori_steps) * time_step_coarse
        time_coarse_finish_loop = (_ + previous_steps) * time_step_coarse 
        time_coarse_start = (_ + previous_steps - feed_steps) * time_step_coarse 
        time_coarse_finish = (_ + previous_steps) * time_step_coarse
        feed_time= time_step_coarse  * n
        time_steps_coarse = np.arange(time_coarse_start, time_coarse_finish, feed_time)  # Time steps for coarse data
        time_steps_coarse_loop = np.arange(time_coarse_start_loop, time_coarse_finish_loop, feed_time)  # Time steps for coarse data
        time_steps_fine = np.arange(4.0, 8.008, 0.008)  
        time_per_step = 0.008

        if _ >= 50000: 
            if 1000 <= _ <= limit_steps:
                if _ % a_posteriori_steps == 0:
                    # モデルのインスタンス化
                    generator = build_cnn_generator()
                    discriminator = load_discriminator()
                    generator.summary()
                    discriminator.summary()

                    # Optimizers with adjusted learning rates
                    generator_optimizer = optimizers.Adam(learning_rate=0.0004)
                    discriminator_optimizer = optimizers.Adam(learning_rate=0.00001)

                    # Discriminatorのコンパイル
                    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                    # Generator + Discriminator（組み合わせモデル）
                    discriminator.trainable = False
                    combined_input = layers.Input(shape=(32, 32, 6))
                    generated_image = generator(combined_input)
                    discriminator_output = discriminator(generated_image)
                    combined_model = models.Model(combined_input, discriminator_output)
                    combined_model.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

                    delete_bin_files(duringtraining_directory)

                    epochs = 100
                    batch_size = 8
                    d_losses = []
                    d_accuracies = []
                    g_losses = []
                    #Prepare input data for generator
                    if _ == 50000:
                        """
                        delete_bin_files(input_data_directory)
                        print("create the coarse input data", feed_steps)
                        for time_step in time_steps_coarse:
                            time_step_str_coarse = f"{time_step:.4f}"  # Convert time step to string
                            # Get velocity data for the coarse case
                            velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)
                            # Prepare coarse velocity fields ########NON DIMENTIONAL VELOCITY FIELDS########
                            u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom, u_data_coarse_input_top,v_data_coarse_input_top, w_data_coarse_input_top,u_wallshear_bottom, w_wallshear_bottom, u_wallshear_top, w_wallshear_top  = prepare_velocity_field_coarse(velocities_coarse, nu, zero_hight_coarse)
                            
                            # Combine u and w wall shear stress data
                            data_combined = [u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse}.bin"
                            output_filepath = os.path.join(input_data_directory, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)

                            # Combine u and w wall shear stress data
                            data_combined = [u_data_coarse_input_top,v_data_coarse_input_top, w_data_coarse_input_top]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse}.bin"
                            output_filepath = os.path.join(input_data_directory_top, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)

                            # Combine u and w wall shear stress data
                            data_combined = [u_wallshear_bottom, w_wallshear_bottom]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse}.bin"
                            output_filepath = os.path.join(coarse_original_data_directory, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)

                            # Combine u and w wall shear stress data
                            data_combined = [u_wallshear_top, w_wallshear_top]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse}.bin"
                            output_filepath = os.path.join(coarse_original_data_directory_top, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)  

                            print(f"Data saved to {output_filepath}")
                        """
                    else:
                        
                        for time_step in time_steps_coarse_loop:
                            time_step_str_coarse_loop = f"{time_step:.4f}"  # Convert time step to string
                            # Get velocity data for the coarse case
                            velocities_coarse = get_velocities(case_coarse, time_step_str_coarse_loop)
                            # Prepare coarse velocity fields ########NON DIMENTIONAL VELOCITY FIELDS########
                            u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom, u_data_coarse_input_top,v_data_coarse_input_top, w_data_coarse_input_top,u_wallshear_bottom, w_wallshear_bottom, u_wallshear_top, w_wallshear_top = prepare_velocity_field_coarse(velocities_coarse, nu, zero_hight_coarse)
                            # Combine u and w wall shear stress data
                            data_combined = [u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse_loop}.bin"
                            output_filepath = os.path.join(input_data_directory, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)

                            # Combine u and w wall shear stress data
                            data_combined = [u_data_coarse_input_bottom, v_data_coarse_input_bottom, w_data_coarse_input_bottom]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse_loop}.bin"
                            output_filepath = os.path.join(input_data_directory_top, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)

                            # Combine u and w wall shear stress data
                            data_combined = [u_wallshear_bottom, w_wallshear_bottom]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse_loop}.bin"
                            output_filepath = os.path.join(coarse_original_data_directory, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)

                            # Combine u and w wall shear stress data
                            data_combined = [u_wallshear_top, w_wallshear_top]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse_loop}.bin"
                            output_filepath = os.path.join(coarse_original_data_directory_top, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)  

                            print(f"Data saved to {output_filepath}")
                        
                    ######## 標準化 ########    
                    coarse_data_list = []
                    for t in time_steps_coarse:
                        t_str = f"{t:.4f}"
                        u_coarse_file = os.path.join(input_data_directory, f"U-{t_str}.bin")
                        u_data_full = np.fromfile(u_coarse_file, dtype=np.float64).reshape((32, 32, 12))
                        # 必要なチャンネル抜粋 (例として0,3,7)
                        u_data = u_data_full[:, :, [0,3,1,4,2,5]]
                        coarse_data_list.append(u_data)

                    fine_data_list = []
                    for t in time_steps_fine:
                        t_str = f"{t:.3f}"
                        u_wallshear_fine_file = os.path.join(real_data_directory, f"wallshear-{t_str}.bin")
                        wallshear_data = np.fromfile(u_wallshear_fine_file, dtype=np.float64).reshape((32, 32, 2))
                        fine_data_list.append(wallshear_data)
                     
                    coarse_data = np.array(coarse_data_list)  # (N,32,32,3)
                    fine_data = np.array(fine_data_list)      # (M,32,32,2)
                    num_samples = min(len(coarse_data), len(fine_data))
                    #coarse_data = coarse_data[:num_samples]
                    #fine_data = fine_data[:num_samples]
                    coarse_data, coarse_means, coarse_stds = standardize_per_channel_global(coarse_data)
                    fine_data, fine_means, fine_stds = standardize_per_channel_global(fine_data)
                    if _ == 50000:
                        print('Now is ', _)
                    else:
                        ########################    
                        d_losses = []
                        d_accuracies = []
                        g_losses = []
                        ######## トレーニングループ ########
                        for epoch in range(epochs):
                            print(f'Epoch {epoch + 1}/{epochs}')
                            d_loss_list = []
                            d_accuracy_list = []
                            g_loss_list = []

                            permutation = np.random.permutation(num_samples)
                            coarse_data_shuffled = coarse_data[permutation]
                            fine_data_shuffled = fine_data[permutation]

                            for batch_start_index in range(0, num_samples, batch_size):
                                upper_bound = min(batch_start_index + batch_size, num_samples)
                                if upper_bound - batch_start_index < batch_size:
                                    break

                                batch_coarse = coarse_data_shuffled[batch_start_index:upper_bound]
                                batch_fine = fine_data_shuffled[batch_start_index:upper_bound]

                                # Generator Prediction
                                generated_images = generator.predict(batch_coarse)  # (batch,32,32,2)

                                # Discriminator Training
                                real_labels = np.ones((batch_size, 1))
                                fake_labels = np.zeros((batch_size, 1))

                                d_loss_real = discriminator.train_on_batch(batch_fine, real_labels)
                                d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
                                d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
                                d_accuracy = 0.5 * (d_loss_real[1] + d_loss_fake[1])

                                # Generator Training
                                g_loss = combined_model.train_on_batch(batch_coarse, real_labels)

                                # ログ出力
                                print(f'Epoch {epoch + 1}, Batch {batch_start_index}, D loss: {d_loss}, D accuracy: {d_accuracy}, G loss: {g_loss}')

                                d_loss_list.append(d_loss)
                                d_accuracy_list.append(d_accuracy)
                                g_loss_list.append(g_loss)

                                # 生成画像を途中で保存
                                save_generated_bin_data(batch_start_index + epoch*num_samples, generated_images[0], duringtraining_directory, time_per_step)

                            # エポックごとの平均ロス
                            epoch_d_loss = np.mean(d_loss_list)
                            epoch_d_accuracy = np.mean(d_accuracy_list)
                            epoch_g_loss = np.mean(g_loss_list)

                            d_losses.append(epoch_d_loss)
                            d_accuracies.append(epoch_d_accuracy)
                            g_losses.append(epoch_g_loss)

                            print(f'Epoch {epoch + 1}, D loss: {epoch_d_loss}, D accuracy: {epoch_d_accuracy}, G loss: {epoch_g_loss}')

                            # モデル保存
                            generator.save('generator.h5', include_optimizer=False)
                            discriminator.save('discriminator.h5', include_optimizer=False)

                            # lossとaccuracy保存
                            with open('loss_data.txt', 'w') as f:
                                for i in range(len(d_losses)):
                                    f.write(f"{i+1}, {d_losses[i]}, {d_accuracies[i]}, {g_losses[i]}\n")

                            # lossグラフ保存
                            plt.figure(figsize=(10,5))
                            plt.plot(range(1, len(d_losses)+1), d_losses, label='Discriminator loss')
                            plt.plot(range(1, len(d_losses)+1), d_accuracies, label='Discrimiantor accuracy')
                            plt.plot(range(1, len(d_losses)+1), g_losses, label='Generator loss')
                            plt.yscale('log')
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss/Accuracy')
                            plt.legend()
                            plt.savefig('loss_accuracy_graph.png')
                            plt.close()
                       
                    time_step_str_coarse = f"{time_coarse_finish:.4f}" 
                    OUTPUT_DB[:,:,:], ave_0, ave_1  = generator_output(case_coarse,time_step_str_coarse,alpha,beta,_,output_directory,U,time_coarse_finish,ny,nu,zero_hight_coarse,time_steps_coarse, input_data_directory, coarse_original_data_directory,input_data_directory_top,coarse_original_data_directory_top)
                    print("ave_0 = ",ave_0)
                    print("ave_1 = ",ave_1)
                        
                # 4. Caluculate Shear Stress
                if not _ % a_posteriori_steps == 0:
                    #custom_objects = {'PeriodicPadding2D': PeriodicPadding2D}
                    #generator = load_model('generator.h5', custom_objects=custom_objects, compile=False)

                    # 必要なオプティマイザと損失関数を指定してモデルをコンパイル
                    #generator.compile(optimizer='adam', loss='binary_crossentropy')
                    case = _ * time_step_coarse
                    time_step_str_coarse = f"{case:.4f}"   # Convert time step to string   
                    case_coarse = "LES_co" 
                    batch_input_data_list_bottom = []
                    batch_input_data_list_top = [] 
                    ##### save stepでしか保存していないから，その間のデータは存在しない．　そのためその前のsave stepのデータから100回のデータから500個の平均の値を使う
                    LAST_case_number = _ //save_steps
                    LAST_case = LAST_case_number *  save_steps
                    print('LAST_case', LAST_case)
                    time_step_str_coarse_LAST = f"{LAST_case:.4f}" 
                    time_coarse_start = (LAST_case - feed_steps) * time_step_coarse 
                    time_coarse_finish = (LAST_case) * time_step_coarse
                    feed_time= time_step_coarse  * save_steps
                    time_steps_coarse = np.arange(time_coarse_start, time_coarse_finish, feed_time)  
                    # Get velocity data for the coarse case
                    velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)
                    OUTPUT_DB[:,:,:], ave_0, ave_1  = generator_output(case_coarse,time_step_str_coarse,alpha,beta,_,output_directory,U,case,ny,nu,zero_hight_coarse,time_steps_coarse, input_data_directory, coarse_original_data_directory,input_data_directory_top,coarse_original_data_directory_top)
                    print("ave_0 = ",ave_0)
                    
            else:
                #custom_objects = {'PeriodicPadding2D': PeriodicPadding2D}
                #generator= load_model('generator.h5', custom_objects=custom_objects, compile=False)

                # 必要なオプティマイザと損失関数を指定してモデルをコンパイル
                #generator.compile(optimizer='adam', loss='binary_crossentropy')
                case = _ * time_step_coarse
                time_step_str_coarse = f"{case:.4f}"   # Convert time step to string 
                case_coarse = "LES_co" 
                batch_input_data_list_bottom = []
                batch_input_data_list_top = [] 
                ##### save stepでしか保存していないから，その間のデータは存在しない．　そのためその前のsave stepのデータから100回のデータから500個の平均の値を使う
                LAST_case_number = _ //save_steps
                LAST_case = LAST_case_number *  save_steps
                time_step_str_coarse_LAST = f"{LAST_case:.4f}"  
                # Get velocity data for the coarse case
                velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)
                OUTPUT_DB[:,:,:], ave_0, ave_1  = generator_output(case_coarse,time_step_str_coarse,alpha,beta,_,output_directory,U,case,ny,nu,zero_hight_coarse,time_steps_coarse, input_data_directory, coarse_original_data_directory,input_data_directory_top,coarse_original_data_directory_top)
                print("ave_0 = ",ave_0)
                print('use the generator and update the wall shear stress at', case)

        logging.debug(f'Output DB updated at step {_}')

        U_0 = []
        U_1 = [] 
        i = 0
        for iz in range(32):
            for ix in range(32):
                U_0.append((OUTPUT_DB[ix, iz, 0], 0.0, OUTPUT_DB[ix, iz, 1]))
                U_1.append((OUTPUT_DB[ix, iz, 2], 0.0, OUTPUT_DB[ix, iz, 3]))
                i += 1
        logging.debug(f'U_1 array populated at step {_}')  
        
        # Calculate body forces
        forces = calculate_body_forces(float(current_time), velocities, locations, iy_feed, iy_feed_2, U_0, U_1, time_step_coarse, ny, delta)

        if _ >= 50000:
            write_forces_to_case(forces, case_dir, current_time)
        
        runner = BasicRunner(argv=["pimpleFoamfeedback", "-case", case_dir], silent=False)
        runner.start()
        logging.debug(f'Runner started at step {_}')
    
        if _ % save_steps != 0:
            shutil.rmtree(f"{case_dir}/{current_time}")
        
        for file in glob.glob(f"{case_dir}/PyFoam.pimpleFoamfeedback.logfile.restart*"):
            os.remove(file)
        
        if runner.runOK():
            case = SolutionDirectory(case_dir)
        else:
            break

if __name__ == "__main__":
    case_dir = "LES_co/"
    num_steps = 50000
    save_steps = 100
    feed_steps = 50000
    limit_steps = 79999
    a_posteriori_steps = 5000
    time_step_coarse = 0.0008
    nu= 0.009
    zero_hight_coarse = 0.00756331
    first_hight_coarse = 0.0271056
    feed_loc = 1.0 / 150
    feed_loc_2 = 299.0 /150
    input_loc = 50 / 150
    alpha = 1
    beta = 1
    ny = 16
    delta = 1
    main(nu, case_dir, num_steps, save_steps, feed_loc, feed_loc_2, input_loc, feed_steps, first_hight_coarse, zero_hight_coarse, limit_steps, time_step_coarse, a_posteriori_steps, alpha, beta, delta, ny)
