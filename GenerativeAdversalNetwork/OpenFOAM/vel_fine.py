from mpi4py import MPI
import os
import numpy as np
import logging
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector

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

# Main training function with MPI
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 各プロセスのランク（ID）を取得
    size = comm.Get_size()  # 総プロセス数を取得
    
    case_fine = "../RUN007-fine/LES_co/"  # Fine simulation case directory
    case_output = "data/data-fine"  # Output directory where you want to save the .bin files
    time_steps_fine = np.arange(4.112, 4.4, 0.008)  # Time steps for fine data
    
    if rank == 0 and not os.path.exists(case_output):
        os.makedirs(case_output)  # Create the output directory if it doesn't exist
    
    # 各プロセスにタイムステップを分割
    time_steps_local = np.array_split(time_steps_fine, size)[rank]  # 各プロセスにサブセットを割り当て
    
    for time_step in time_steps_local:
        time_step_str_fine = f"{time_step:.3f}"  # Convert time step to string
        
        # Get velocity data for the fine case
        velocities_fine = get_velocities(case_fine, time_step_str_fine)
        
        # Prepare fine wall shear stress data
        u_wallshear_fine, w_wallshear_fine = prepare_velocity_field_fine(velocities_fine)

        # Combine u and w wall shear stress data
        data_combined = [u_wallshear_fine, w_wallshear_fine]
        batch_real_input_data = np.stack(data_combined, axis=-1)

        # Create filename for this time step
        output_filename = f"wallshear-{time_step_str_fine}.bin"
        output_filepath = os.path.join(case_output, output_filename)

        # Save the combined data (u and w wall shear stress) as binary file
        batch_real_input_data.tofile(output_filepath)
        print(f"Process {rank}: Data saved to {output_filepath}")

if __name__ == "__main__":
    main()
