import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import glob


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
def prepare_velocity_field_coarse(velocities):
    nx, ny, nz = 32, 16, 32
    half_ny = 8
    top_bottom = 2
    u_data = []
    w_data = []

    #selected_y_indices = [1, 2, 3, 4]  # y方向のインデックス
    selected_y_indices = [6,5,4,3]  # y方向のインデックス
    u_data_coarse_input = []
    w_data_coarse_input = []

    i = nx * half_ny * nz
    print('Start of i =', i)

    # 速度データをリストに格納
    for iz in range(nz):
        for iy in range(half_ny):
            for ix in range(nx):
                u_data.append(velocities[i][0])  # u (x方向速度)
                w_data.append(velocities[i][2])  # w (z方向速度)
                i += 1
    """
    for iz in range(nz):
        for iy in range(half_ny, ny, 1):
            for ix in range(nx):
                u_data.append(velocities[i][0])  # u (x方向速度)
                w_data.append(velocities[i][2])  # w (z方向速度)
                i += 1
    """
    # リストをnumpy配列に変換
    u_data = np.array(u_data).reshape((nx, half_ny, nz))  # Shape: (32, 16, 32)
    w_data = np.array(w_data).reshape((nx, half_ny, nz))  # Shape: (32, 16, 32)

    # 壁面近傍の四層のデータを抽出 (32, 32, 4)
    for iy in selected_y_indices:
        u_data_coarse_input.append(u_data[:, iy, :])  # Extract y-slice
        w_data_coarse_input.append(w_data[:, iy, :])  # Extract y-slice

    # Stack u and w data from selected y-slices to get shape (32, 32, 4)
    u_data_coarse_input = np.stack(u_data_coarse_input, axis=-1)  # Shape: (32, 32, 4)
    w_data_coarse_input = np.stack(w_data_coarse_input, axis=-1)  # Shape: (32, 32, 4)

    return u_data_coarse_input, w_data_coarse_input

# 結果をテキストファイルに保存する関数
def save_averages_to_file(y_planes, u_averages, w_averages, save_path):
    # データを結合して保存
    data = np.column_stack((y_planes, u_averages, w_averages))
    header = "y_plane_index\tAverage_U\tAverage_W (Time-averaged)"
    
    np.savetxt(save_path, data, header=header, fmt='%.6f', delimiter='\t')
    print(f"Data saved to {save_path}")

# Main training function
def main(): 
    case_coarse = "../RUN004-wo-ODE/LES_co/"  # Coarse simulation case directory
    case_output = "data/data-coarse"  # coarse data directory where binary files are saved
    time_steps_coarse = np.arange(4, 8, 0.016) 
    add_time = 4.000
    
    if not os.path.exists(case_output):
        os.makedirs(case_output)  # Create the output directory if it doesn't exist
    
    for time_step in time_steps_coarse:
        time_step_str_coarse = f"{time_step:.3f}"  # Convert time step to string
        
        # Get velocity data for the coarse case
        velocities_coarse = get_velocities(case_coarse, time_step_str_coarse)

        # Prepare coarse velocity fields
        u_data_coarse_input, w_data_coarse_input = prepare_velocity_field_coarse(velocities_coarse)

        # Combine u and w wall shear stress data
        data_combined = [u_data_coarse_input, w_data_coarse_input]
        batch_real_input_data = np.stack(data_combined, axis=-1)

        output_time = time_step + add_time 
        output_time_str_coarse = time_step_str_coarse = f"{output_time:.3f}" 

        # Create filename for this time step
        output_filename = f"U-{output_time_str_coarse}.bin"
        output_filepath = os.path.join(case_output, output_filename)

        # Save the combined data (u and w wall shear stress) as binary file
        batch_real_input_data.tofile(output_filepath)
        print(f"Data saved to {output_filepath}")

if __name__ == "__main__":
    main()

