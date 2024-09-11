import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
import glob

# データ
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

"""
# Discriminatorのインポート
def load_discriminator():
    try:
        model = tf.keras.models.load_model('discriminator.h5')
        print("Discriminator model loaded from discriminator.h5")
    except:
        model = models.Sequential(name="Discriminator")
        model.add(layers.Input(shape=(32, 32, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        #model.add(layers.BatchNormalization())
        #model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        #model.add(layers.BatchNormalization())
        #model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        #model.add(layers.BatchNormalization())
        #model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        print("Discriminator model created from scratch")
    return model
"""
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
    nx, ny, nz = 128, 64, 128
    first_hight_fine = 0.0625
    nu = 0.009
    u_data = []
    w_data = []
    
    i = 0
    
    # 速度データをリストに格納
    for iy in range(ny):  # y方向すべてのインデックスを処理
        for iz in range(nz):  # z方向
            for ix in range(nx):  # x方向
                u_data.append(velocities[i][0])  # u (x方向速度)
                w_data.append(velocities[i][2])  # w (z方向速度)
                i += 1

    # リストをnumpy配列に変換
    u_data = np.array(u_data).reshape((nx, ny, nz))  
    w_data = np.array(w_data).reshape((nx, ny, nz)) 
    
    # 四つおきにデータを取得し、壁面剪断応力を計算
    u_wallshear_fine = u_data[::4, 1, ::4] / first_hight_fine * nu
    w_wallshear_fine = w_data[::4, 1, ::4] / first_hight_fine * nu

    return u_wallshear_fine, w_wallshear_fine

# 速度場のデータを3次元配列に格納する関数
def prepare_velocity_field_coarse(velocities):
    nx, ny, nz = 32, 16, 32
    half_ny = 8
    top_bottom = 2
    u_data = []
    w_data = []

    selected_y_indices = [1, 2, 3, 4]  # y方向のインデックス
    #selected_y_indices = [14,13,12,11]  # y方向のインデックス
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

# Main training function
def main():
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

    input_data_directory = "data/data-coarse"  # Coarse simulation case directory
    case_coarse ="../RUN004-wo-ODE/LES_co/"  # Coarse simulation case directory
    real_data_directory = "data/data-fine"  # Fine data directory where binary files are saved
    time_steps_coarse = np.arange(4.0, 12.0, 0.016)  # Time steps for coarse data
    time_steps_fine = np.arange(4, 8, 0.008)    # Time steps for fine data
    time_per_step = 0.016

    epochs = 20
    batch_size = 16
    d_losses = []
    d_accuracies = []
    g_losses = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        d_loss_list = []
        d_accuracy_list = []
        g_loss_list = []

        # Reset the batch input data list at each epoch
        batch_input_data_list = []
        batch_real_data_list = []

        for batch_start_index in range(0, len(time_steps_coarse), batch_size):
            upper_bound = min(batch_start_index + batch_size, len(time_steps_coarse))

            # 1. Process coarse velocity data and prepare input for the generator
            batch_input_data_list = []  # Reset for each batch
            for time_step in time_steps_coarse[batch_start_index:upper_bound]:
                time_step_str_coarse = f"{time_step:.3f}"  # Convert time step to string

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
            save_generated_bin_data(batch_start_index, generated_images, duringtraining_directory, time_per_step)

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
        #discriminator.save('discriminator.h5')

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

    ## 構築したモデルを用いて、速度場を生成する
    generator = tf.keras.models.load_model('generator.h5')
    output_directory = "data/data-wallshear-est"
    delete_bin_files(output_directory)
    time_steps_coarse_output = np.arange(7.2, 8.0, 0.016) 
    for time_step in time_steps_coarse_output:
        batch_input_data_list = []
        time_step_str_coarse = f"{time_step:.3f}"  # Convert time step to string
        
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
        
        save_generated_bin_data(time_step, generated_data, output_directory, time_per_step)

    print("All generated and transformed data saved.")

if __name__ == "__main__":
    main()
