#! /work/takumie/virtual/bin/python

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
# ログの設定
logging.basicConfig(filename='script.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

duringtraining_directory = "data/data-wallshear-save"

# Generatorの構築
def build_cnn_generator(input_shape=(32, 32, 8)):
    model = models.Sequential(name="Generator")
    model.add(layers.Input(shape=input_shape))
    
    # Conv2D + BatchNormalization + ReLU の順番でレイヤーを追加
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    #model.add(layers.BatchNormalization())  # バッチノーマライゼーションをアクティベーションの前に
    model.add(layers.ReLU())  # ReLUアクティベーション
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    #model.add(layers.BatchNormalization())  # バッチノーマライゼーションをアクティベーションの前に
    model.add(layers.ReLU())  # ReLUアクティベーション
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    #model.add(layers.BatchNormalization())  # バッチノーマライゼーションをアクティベーションの前に
    model.add(layers.ReLU())  # ReLUアクティベーション

    # 最終出力層
    model.add(layers.Conv2D(2, (3, 3), padding='same'))
    
    return model

# Discriminatorのインポート
def load_discriminator():
    try:
        model = tf.keras.models.load_model('discriminator.h5')
        print("Discriminator model loaded from discriminator.h5")
    except:
        model = models.Sequential(name="Discriminator")
        model.add(layers.Input(shape=(32, 32, 2)))
        
        # Conv2D + BatchNormalization + ReLU
        model.add(tfa.layers.SpectralNormalization(layers.Conv2D(16, (3, 3), padding='same')))
        #model.add(layers.BatchNormalization())  # バッチノーマライゼーションを追加
        model.add(layers.ReLU())  # ReLUをここで適用
        
        model.add(tfa.layers.SpectralNormalization(layers.Conv2D(32, (3, 3), padding='same')))
        #model.add(layers.BatchNormalization())  # バッチノーマライゼーションを追加
        model.add(layers.ReLU())  # ReLUをここで適用
        
        model.add(tfa.layers.SpectralNormalization(layers.Conv2D(64, (3, 3), padding='same')))
        #model.add(layers.BatchNormalization())  # バッチノーマライゼーションを追加
        model.add(layers.ReLU())  # ReLUをここで適用
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        print("Discriminator model created from scratch")
    
    return model
# データの読み込みと保存用関数
def load_data(filepath):
    data = np.fromfile(filepath, dtype=np.float64).reshape((32, 32, 1))
    return data

# データの前処理用関数
def preprocess_data(data, first_value, nu):
    data = data / first_value * nu
    return data

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
def prepare_velocity_field_coarse(velocities):
    nx, ny, nz = 32, 16, 32
    half_ny = 8
    u_data = []
    w_data = []

    selected_y_indices = [1,2,3,4]  # y方向のインデックス
    u_data_coarse_input = []
    w_data_coarse_input = []

    i = 0

    # 速度データをリストに格納
    for iz in range(nz):
        for iy in range(half_ny):
            for ix in range(nx):
                u_data.append(velocities[i][0])  # u (x方向速度)
                w_data.append(velocities[i][2])  # w (z方向速度)
                i += 1

    # リストをnumpy配列に変換
    u_data = np.array(u_data).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)
    w_data = np.array(w_data).reshape((nx, half_ny, nz))  # Shape: (32, 8, 32)

    # 壁面近傍の四層のデータを抽出 (32, 32, 4)
    for iy in selected_y_indices:
        u_data_coarse_input.append(u_data[:, iy, :])  # Extract y-slice
        w_data_coarse_input.append(w_data[:, iy, :])  # Extract y-slice

    # Stack u and w data from selected y-slices to get shape (32, 32, 4)
    u_data_coarse_input = np.stack(u_data_coarse_input, axis=-1)  # Shape: (32, 32, 4)
    w_data_coarse_input = np.stack(w_data_coarse_input, axis=-1)  # Shape: (32, 32, 4)

    return u_data_coarse_input, w_data_coarse_input

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
    
    return iy_feed_index

def write_forces_to_case(velocities, forces, case, time_step):
    body_force_file_path = os.path.join(case, time_step, "U")
    os.makedirs(os.path.dirname(body_force_file_path), exist_ok=True)
    
    with open(body_force_file_path, 'w') as f:
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       volVectorField;\n')
        f.write('    location    "{}";\n'.format(time_step))
        f.write('    object      U;\n')
        f.write('}\n')
        f.write('dimensions      [0 1 -1 0 0 0 0];\n')
        f.write('internalField   nonuniform List<vector>\n')
        f.write('{}\n'.format(len(velocities)))
        f.write('(\n')

        for velocity in velocities:
            f.write('({} {} {})\n'.format(velocity[0], velocity[1], velocity[2]))
        f.write(')\n')
        f.write(';\n')
        f.write('boundaryField\n')
        f.write('{\n')
        f.write('    bottom\n')
        f.write('    {\n')
        f.write('        type            fixedGradient;\n') 
        f.write('        gradient        nonuniform List<vector>\n') 
        f.write('        {}\n'.format(len(forces)))
        f.write('        (\n')

        for force in forces:
            f.write('           ({} {} {})\n'.format(force[0], force[1], force[2]))

        f.write('        );\n')
        f.write('    }\n')
        f.write('    top\n')
        f.write('    {\n')
        f.write('        type            noSlip;\n')
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

def write_only_velocity_to_case(velocities, case_dir, current_time):
    body_force_file_path = os.path.join(case_dir, current_time, "U")
    os.makedirs(os.path.dirname(body_force_file_path), exist_ok=True)
    
    with open(body_force_file_path, 'w') as f:
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       volVectorField;\n')
        f.write('    location    "{}";\n'.format(current_time))
        f.write('    object      U;\n')
        f.write('}\n')
        f.write('dimensions      [0 1 -1 0 0 0 0];\n')
        f.write('internalField   nonuniform List<vector>\n')
        f.write('{}\n'.format(len(velocities)))
        f.write('(\n')

        for velocity in velocities:
            f.write('({} {} {})\n'.format(velocity[0], velocity[1], velocity[2]))
        f.write(')\n')
        f.write(';\n')
        f.write('boundaryField\n')
        f.write('{\n')
        f.write('    bottom\n')
        f.write('    {\n')
        f.write('        type            noSlip;\n')
        f.write('    }\n')
        f.write('    top\n')
        f.write('    {\n')
        f.write('        type            noSlip;\n')
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

def main(nu, case_dir, num_steps, save_steps, feed_loc, input_loc, feed_steps, first_hight_coarse, zero_hight_coarse, limit_steps, time_step_coarse, a_posteriori_steps):
    case = SolutionDirectory(case_dir)
    locations, iy_feed = get_locations(case_dir, feed_loc)
    locations_input, iy_input = get_locations(case_dir, input_loc)
    iy_feed_index = find_iy_feed_index(case_dir, feed_loc)
    iy_input_index = find_iy_feed_index(case_dir, input_loc)

    current_time = case.getLast() 
    start_step = int(float(current_time) / time_step_coarse) 

    for _ in range(start_step, start_step + num_steps):
        logging.debug(f'Step {_} start')
        current_time = case.getLast()
        logging.debug(f'Current time: {current_time}')
        velocities = get_velocities_normal(case_dir, current_time)
        logging.debug(f'Velocities obtained at step {_}')

        U = np.zeros((32, 16, 32, 3))
        i = 0
        for iz in range(32):
            for iy in range(8):
                for ix in range(32):
                    U[ix, iy, iz, 0] = velocities[i][0]
                    U[ix, iy, iz, 1] = velocities[i][1]
                    U[ix, iy, iz, 2] = velocities[i][2]
                    i += 1
        
        logging.debug(f'U array populated at step {_}')               

        U_input = U[:, iy_input_index, :, :]
        #if _ % feed_steps == 0:
        INPUT_DB = np.zeros((32, 32, 3))
        OUTPUT_DB = np.zeros((32, 32, 2))
        #INPUT_DB[:, :, :] = U_input

        if _ != 0: 
            if _ >= limit_steps:
                if _ % a_posteriori_steps == 0:
                    # モデルのインスタンス化
                    generator = build_cnn_generator()
                    discriminator = load_discriminator()
                    generator.summary()
                    discriminator.summary()

                    # Optimizers with adjusted learning rates
                    generator_optimizer = optimizers.Adam(learning_rate=0.0001)
                    discriminator_optimizer = optimizers.Adam(learning_rate=0.00002)

                    # Discriminatorのコンパイル
                    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                    # Generator + Discriminator（組み合わせモデル）
                    discriminator.trainable = False
                    combined_input = layers.Input(shape=(32, 32, 8))
                    generated_image = generator(combined_input)
                    discriminator_output = discriminator(generated_image)
                    combined_model = models.Model(combined_input, discriminator_output)
                    combined_model.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

                    time_coarse_start = 0
                    time_coarse_finish = 0
                    previous_steps = 0
                    time_coarse_start_loop = (_ + previous_steps - a_posteriori_steps) * time_step_coarse
                    time_coarse_finish_loop = (_ + previous_steps) * time_step_coarse  
                    time_coarse_start = (_ + previous_steps - feed_steps) * time_step_coarse
                    time_coarse_finish = (_ + previous_steps) * time_step_coarse 
                    case_coarse = "LES_co"  # Coarse simulation case directory
                    input_data_directory = "data/data-coarse" 
                    real_data_directory = "data/data-fine"  # Fine data directory where binary files are saved
                    time_steps_coarse = np.arange(time_coarse_start, time_coarse_finish, time_step_coarse)  # Time steps for coarse data
                    time_steps_coarse_loop = np.arange(time_coarse_start_loop, time_coarse_finish_loop, time_step_coarse)  # Time steps for coarse data
                    time_steps_fine = np.arange(4.0, 8.008, 0.008)  
                    time_per_step = 0.008

                    epochs = 10
                    batch_size = 8
                    d_losses = []
                    d_accuracies = []
                    g_losses = []
                    #Prepare input data for generator
                    if _ == limit_steps:
                        """
                        delete_bin_files(input_data_directory)
                        print("create the coarse input data", feed_steps)
                        for time_step in time_steps_coarse:
                            time_step_str_coarse = f"{time_step:.4f}"  # Convert time step to string
                            # Get velocity data for the coarse case
                            velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)
                            # Prepare coarse velocity fields
                            u_data_coarse_input, w_data_coarse_input = prepare_velocity_field_coarse(velocities_coarse)
                            # Combine u and w wall shear stress data
                            data_combined = [u_data_coarse_input, w_data_coarse_input]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse}.bin"
                            output_filepath = os.path.join(input_data_directory, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)
                            print(f"Data saved to {output_filepath}")
                        """
                    else:
                        for time_step in time_steps_coarse_loop:
                            time_step_str_coarse_loop = f"{time_step:.4f}"  # Convert time step to string
                            # Get velocity data for the coarse case
                            velocities_coarse = get_velocities(case_coarse, time_step_str_coarse_loop)
                            # Prepare coarse velocity fields
                            u_data_coarse_input, w_data_coarse_input = prepare_velocity_field_coarse(velocities_coarse)
                            # Combine u and w wall shear stress data
                            data_combined = [u_data_coarse_input, w_data_coarse_input]
                            batch_real_input_data = np.stack(data_combined, axis=-1)
                            # Create filename for this time step
                            output_filename = f"U-{time_step_str_coarse_loop}.bin"
                            output_filepath = os.path.join(input_data_directory, output_filename)
                            # Save the combined data (u and w wall shear stress) as binary file
                            batch_real_input_data.tofile(output_filepath)
                            print(f"Data saved to {output_filepath}")

                    for epoch in range(epochs):
                        print(f'Epoch {epoch + 1}/{epochs}')
                        d_loss_list = []
                        d_accuracy_list = []
                        g_loss_list = []

                        # Reset the batch input data list at each epoch
                        batch_input_data_list = []
                        batch_real_data_list = []

                        for batch_start_index in range(0, feed_steps, batch_size):
                            upper_bound = min(batch_start_index + batch_size, feed_steps)

                            # 1. Process coarse velocity data and prepare input for the generator
                            batch_input_data_list = []  # Reset for each batch
                            for time_step in time_steps_coarse[batch_start_index:upper_bound]:
                                time_step_str_coarse = f"{time_step:.4f}"  # Convert time step to string

                                # Get velocity data for the coarse case
                                u_coarse_file = os.path.join(input_data_directory, f"U-{time_step_str_coarse}.bin")

                                # Prepare coarse velocity fields
                                u_data = np.fromfile(u_coarse_file, dtype=np.float64).reshape((32, 32, 8))

                                batch_input_data_list.append(u_data)
                                print(time_step)

                            # Convert to numpy array
                            batch_input_data = np.array(batch_input_data_list)  # Shape: (batch_size, 32, 32, 8)

                            # Check if this batch size is less than the expected batch size (only happens on the last batch)
                            if batch_input_data.shape[0] != batch_size:
                                print(f"Warning: Last batch size is smaller: {batch_input_data.shape[0]} (expected {batch_size})")
                            else:
                                assert batch_input_data.shape == (batch_size, 32, 32, 8), \
                                    f"batch_input_data shape mismatch: {batch_input_data.shape}, expected (batch_size, 32, 32, 8)"

                            # 2. Process fine wall shear stress data from pre-saved binary files
                            batch_real_data_list = []  # Reset for each batch
                            for time_step in time_steps_fine[batch_start_index:upper_bound]:
                                time_step_str_fine = f"{time_step:.3f}"  # Convert time step to string

                                # Load the pre-saved wall shear stress data for the fine case
                                u_wallshear_fine_file = os.path.join(real_data_directory, f"wallshear-{time_step_str_fine}.bin")

                                # Read binary data (32, 32, 2) as wall shear stress
                                wallshear_data = np.fromfile(u_wallshear_fine_file, dtype=np.float64).reshape((32, 32, 2))

                                # Append the fine data to the batch list
                                batch_real_data_list.append(wallshear_data)

                            # Convert to numpy array
                            batch_real_data = np.array(batch_real_data_list)  # Shape: (batch_size, 32, 32, 2)

                            # 3. Generator Prediction
                            generated_images = generator.predict(batch_input_data)  # Shape: (batch_size, 32, 32, 2)

                            # Skip batch if it was smaller
                            if batch_input_data.shape[0] != batch_size:
                                print(f"Skipping last batch with size: {batch_input_data.shape[0]}")
                                continue  # Skip this batch if it's smaller than the expected batch size
                                
                            # Save generated data
                            save_generated_bin_data(batch_start_index, generated_images, duringtraining_directory,time_per_step)

                            # 4. Discriminator Training
                            real_labels = np.ones((batch_size, 1))  # Real labels are 1
                            fake_labels = np.zeros((batch_size, 1))  # Fake labels are 0
                            
                            d_loss_real, d_accuracy_real = discriminator.train_on_batch(batch_real_data, real_labels)
                            d_loss_fake, d_accuracy_fake = discriminator.train_on_batch(generated_images, fake_labels)
                            
                            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                            d_accuracy = 0.5 * (d_accuracy_real + d_accuracy_fake)
                            
                            # 5. Generator Training (using combined model with discriminator frozen)
                            g_loss = combined_model.train_on_batch(batch_input_data, real_labels)
                            
                            print(f'Epoch {epoch + 1}, Batch {batch_start_index}, D loss: {d_loss}, D accuracy: {d_accuracy}, G loss: {g_loss}')
                            
                            # Append losses and accuracies for tracking
                            d_loss_list.append(d_loss)
                            d_accuracy_list.append(d_accuracy)
                            g_loss_list.append(g_loss)

                        # Calculate and store average losses per epoch
                        epoch_d_loss = np.mean(d_loss_list)
                        epoch_d_accuracy = np.mean(d_accuracy_list)
                        epoch_g_loss = np.mean(g_loss_list)
                        
                        d_losses.append(epoch_d_loss)
                        d_accuracies.append(epoch_d_accuracy)
                        g_losses.append(epoch_g_loss)

                        print(f'Epoch {epoch + 1}, D loss: {epoch_d_loss}, D accuracy: {epoch_d_accuracy}, G loss: {epoch_g_loss}')

                        # Save models after each epoch
                        generator.save('generator.h5')

                        # Save loss and accuracy data
                        with open('loss_data.txt', 'w') as f:
                            for i in range(len(d_losses)):
                                f.write(f"{i+1}, {d_losses[i]}, {d_accuracies[i]}, {g_losses[i]}\n")
                        
                    # lossのグラフを生成し、保存
                    plt.figure(figsize=(10, 5))
                    plt.semilogy(range(1, epochs+1), d_losses, label='Discriminator Loss')
                    plt.semilogy(range(1, epochs+1), d_accuracies, label='Discriminator Accuracy')
                    plt.semilogy(range(1, epochs+1), g_losses, label='Generator Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss/Accuracy')
                    plt.xticks(range(0, epochs+1, 10))
                    plt.legend()
                    plt.savefig('loss_accuracy_graph.png')
                    plt.close()

                    # generate results
                    batch_input_data_list = []
                    # Get velocity data for the coarse case
                    time_step_str_coarse = f"{time_coarse_finish:.4f}"  # Convert time step to string
        
                    # Get velocity data for the coarse case
                    velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)
                    
                    # Prepare coarse velocity fields
                    u_data_coarse_input, w_data_coarse_input = prepare_velocity_field_coarse(velocities_coarse)
                    
                    # Concatenate u and w data to form (32, 32, 8) input
                    batch_combined_data = np.concatenate((u_data_coarse_input, w_data_coarse_input), axis=-1)
                    batch_input_data_list.append(batch_combined_data)
                            
                    # Convert lists to numpy arrays for batch processing
                    batch_input_data = np.array(batch_input_data_list)  # Shape: (batch_size, 32, 32, 8)

                    # 3. Generator Prediction
                    generated_images = generator.predict(batch_input_data)  # Shape: (batch_size, 32, 32, 2)
                    generated_data = generated_images.squeeze()

                    OUTPUT_DB[:, :, 0] = generated_data[:,:,0] / nu
                    OUTPUT_DB[:, :, 1] = generated_data[:,:,1] / nu

                # 4. Caluculate Shear Stress
                if not _ % feed_steps == 0:
                    generator = tf.keras.models.load_model('generator.h5')
                    case = _ * time_step_coarse 
                    case_coarse = "LES_co" 
                    batch_input_data_list = []
                    time_step_str_coarse = f"{case:.4f}"   # Convert time step to string
                    
                    # Get velocity data for the coarse case
                    velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)
                    
                    # Prepare coarse velocity fields
                    u_data_coarse_input, w_data_coarse_input = prepare_velocity_field_coarse(velocities_coarse)
                    
                    # Concatenate u and w data to form (32, 32, 8) input
                    batch_combined_data = np.concatenate((u_data_coarse_input, w_data_coarse_input), axis=-1)
                    batch_input_data_list.append(batch_combined_data)
                            
                    # Convert lists to numpy arrays for batch processing
                    batch_input_data = np.array(batch_input_data_list)  # Shape: (batch_size, 32, 32, 8)

                    # 3. Generator Prediction
                    generated_images = generator.predict(batch_input_data)  # Shape: (batch_size, 32, 32, 2)
                    generated_data = generated_images.squeeze()

                    OUTPUT_DB[:, :, 0] = generated_data[:,:,0] / nu
                    OUTPUT_DB[:, :, 1] = generated_data[:,:,1] / nu
                    print('use the generator and update the wall shear stress at', case)
            else:
                for iz in range(32):
                    for ix in range(32):
                        OUTPUT_DB[ix, iz, 0]  =  U[ix, 0, iz, 0] / zero_hight_coarse
                        OUTPUT_DB[ix, iz, 1]  =  U[ix, 0, iz, 2] / zero_hight_coarse

        logging.debug(f'Output DB updated at step {_}')

        U_0 = []
        U_1 = [] 
        i = 0
        for iz in range(32):
            for ix in range(32):
                U_0.append((OUTPUT_DB[ix, iz, 0], 0.0, OUTPUT_DB[ix, iz, 1]))
                U_1.append((OUTPUT_DB[ix, iz, 0], 0.0, OUTPUT_DB[ix, iz, 1]))
                i += 1
        logging.debug(f'U_1 array populated at step {_}')   
        if _ >= limit_steps:
            if _ % a_posteriori_steps == 0:
                write_forces_to_case(velocities, U_0, case_dir, current_time)
                logging.debug(f'Forces written to case at step {_} for time {current_time}')
            else:
                write_only_velocity_to_case(velocities,  case_dir, current_time)
                logging.debug(f'Only velocities written to case at step {_} for time {current_time}') 
        runner = BasicRunner(argv=["pimpleFoam", "-case", case_dir], silent=False)
        runner.start()
        logging.debug(f'Runner started at step {_}')
    
        if _ % save_steps != 0:
            shutil.rmtree(f"{case_dir}/{current_time}")
        
        for file in glob.glob(f"{case_dir}/PyFoam.pimpleFoam.logfile.restart*"):
            os.remove(file)
        
        if runner.runOK():
            case = SolutionDirectory(case_dir)
        else:
            break

if __name__ == "__main__":
    case_dir = "LES_co/"
    num_steps = 3000
    save_steps = 1
    feed_steps = 500
    limit_steps = 1500
    a_posteriori_steps = 5
    time_step_coarse = 0.002
    nu= 0.009
    zero_hight_coarse = 0.00756331
    first_hight_coarse = 0.0271056
    feed_loc = 1.0 / 150.0
    input_loc = 50 / 150.0
    main(nu, case_dir, num_steps, save_steps, feed_loc, input_loc, feed_steps, first_hight_coarse, zero_hight_coarse, limit_steps, time_step_coarse, a_posteriori_steps)
