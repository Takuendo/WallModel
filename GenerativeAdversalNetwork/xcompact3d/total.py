import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# データ
input_directory = '../data-LES-2/data-32-33-32'
real_data_directory = "data/data-wallshear-fine"
duringtraining_directory = "data/data-wallshear-save"

# Generatorの構築
def build_cnn_generator(input_shape=(32, 32, 8)):
    model = models.Sequential(name="Generator")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    #model.add(layers.LayerNormalization())
    #model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(layers.LayerNormalization())
    #model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(layers.LayerNormalization())
    #model.add(layers.MaxPooling2D((2, 2), padding='same'))
    #model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    #model.add(layers.LayerNormalization())
    #model.add(layers.UpSampling2D((2, 2)))
    #model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    #model.add(layers.LayerNormalization())
    #model.add(layers.UpSampling2D((2, 2)))
    #model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    #model.add(layers.LayerNormalization())
    #model.add(layers.UpSampling2D((2, 2)))
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
        model.add(tfa.layers.SpectralNormalization(layers.Conv2D(16, (3, 3), activation='relu', padding='same')))
        model.add(tfa.layers.SpectralNormalization(layers.Conv2D(32, (3, 3), activation='relu', padding='same')))
        model.add(tfa.layers.SpectralNormalization(layers.Conv2D(64, (3, 3), activation='relu', padding='same')))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        print("Discriminator model created from scratch")
    return model
""""
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
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        #model.add(layers.BatchNormalization())
        #model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
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
def save_generated_bin_data(index, generated_data, directory):
    filename = os.path.join(directory, f'ux-gen-{index}.bin')
    generated_data = generated_data.astype(np.float64)
    generated_data.tofile(filename)

def delete_bin_files(directory):
    bin_files = glob.glob(os.path.join(directory, '*.bin'))
    for file_path in bin_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# モデルのインスタンス化
generator = build_cnn_generator()
discriminator = load_discriminator()
generator.summary()
discriminator.summary()

# Optimizers with adjusted learning rates
generator_optimizer = optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = optimizers.Adam(learning_rate=0.00001)

# Discriminatorのコンパイル
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Generator + Discriminator（組み合わせモデル）
discriminator.trainable = False
combined_input = layers.Input(shape=(32, 32, 8))
generated_image = generator(combined_input)
discriminator_output = discriminator(generated_image)
combined_model = models.Model(combined_input, discriminator_output)
combined_model.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

# トレーニングパラメータ
epochs = 20
batch_size = 8
start_index = 3501
end_index = 4001
fine_first_height = 5.555778295597058E-003
shape = (32, 33, 32)
nu = 1 / 4200
d_losses = []
d_accuracies = []
g_losses = []

# トレーニングループ
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    d_loss_list = []
    d_accuracy_list = []
    g_loss_list = []
    delete_bin_files(duringtraining_directory)

    # すべてのデータに対して生成を行い、保存
    for batch_start_index in range(start_index, end_index, batch_size):
        batch_input_data_list = []
        upper_bound = min(batch_start_index + batch_size, end_index)
        for i in range(batch_start_index, upper_bound):
            # ux-データを読み込み
            ux_file = os.path.join(input_directory, f'ux-{i}.bin')
            ux_data = np.fromfile(ux_file, dtype=np.float64).reshape(shape)
            
            # uz-データを読み込み
            uz_file = os.path.join(input_directory, f'uz-{i}.bin')
            uz_data = np.fromfile(uz_file, dtype=np.float64).reshape(shape)
            
            # スライスしたデータを保持するリスト
            sliced_ux_data_list = []
            sliced_uz_data_list = []
            
            for j in range(1, 5):
                sliced_ux_data = ux_data[:, j, :]
                sliced_uz_data = uz_data[:, j, :]
                sliced_ux_data_list.append(sliced_ux_data)
                sliced_uz_data_list.append(sliced_uz_data)
            
            # (32, 32, 4) の形にデータを結合
            batch_ux_data = np.stack(sliced_ux_data_list, axis=-1)
            batch_uz_data = np.stack(sliced_uz_data_list, axis=-1)
            
            # ux-とuz-データを結合して (32, 32, 8) にする
            batch_combined_data = np.concatenate((batch_ux_data, batch_uz_data), axis=-1)
            batch_input_data_list.append(batch_combined_data)

        # バッチサイズ分のデータを (batch_size, 32, 32, 8) に結合
        batch_input_data = np.array(batch_input_data_list)
    
        # Generatorで速度場を生成
        generated_images = generator.predict(batch_input_data)
        save_generated_bin_data(batch_start_index, generated_images, duringtraining_directory)
        batch_generated_data = np.array(generated_images)

        # 実データの読み込みと標準化
        batch_real_data = []
        for i in range(batch_start_index, upper_bound):
            data_combined = []
            for vel_direction in ['x', 'z']:
                filename = f'u{vel_direction}-tau{i+500}.bin'
                filepath = os.path.join(real_data_directory, filename)
                data = np.fromfile(filepath, dtype=np.float64).reshape((32,32), order='F')
                
                # 一層目のデータをスライスして取得し、前処理
                #sliced_data = preprocess_data(data[::4, 1, ::4], fine_first_height, nu)
                center_data = data[:, :]
                data_combined.append(center_data)
            
            # (32, 32, 2) の形にデータを結合
            batch_real_input_data = np.stack(data_combined, axis=-1)
            batch_real_data.append(batch_real_input_data)

        batch_real_data = np.array(batch_real_data)

        # ラベルの準備
        real_labels = np.ones((len(batch_real_data), 1))
        fake_labels = np.zeros((len(batch_generated_data), 1))

        d_loss_real, d_accuracy_real = discriminator.train_on_batch(batch_real_data, real_labels)
        d_loss_fake, d_accuracy_fake = discriminator.train_on_batch(batch_generated_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_accuracy = 0.5 * (d_accuracy_real + d_accuracy_fake)

        g_loss = combined_model.train_on_batch(batch_input_data, real_labels)
        """
        # Discriminatorの精度が0.55未満になるまでGeneratorをトレーニング
        if epoch > 4:
            while g_loss > 0.1 or d_accuracy < 0.9:
                g_loss = combined_model.train_on_batch(batch_input_data, real_labels)
                generated_images = generator.predict(batch_input_data)
                d_loss_real, d_accuracy_real = discriminator.train_on_batch(batch_real_data, real_labels)
                d_loss_fake, d_accuracy_fake = discriminator.train_on_batch(generated_images, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_accuracy = 0.5 * (d_accuracy_real + d_accuracy_fake)
                print("D loss:", d_loss, "D accuracy:", d_accuracy, "G loss:", g_loss)
        """
        print(f'Epoch {epoch + 1}, Batch {batch_start_index}, D loss: {d_loss}, D accuracy: {d_accuracy}, G loss: {g_loss}')
        d_loss_list.append(d_loss)
        d_accuracy_list.append(d_accuracy)
        g_loss_list.append(g_loss) 

    epoch_d_loss = np.mean(d_loss_list)
    epoch_d_accuracy = np.mean(d_accuracy_list)
    epoch_g_loss = np.mean(g_loss_list)
    d_losses.append(epoch_d_loss)
    d_accuracies.append(epoch_d_accuracy)
    g_losses.append(epoch_g_loss)

    print(f'Epoch {epoch + 1}, D loss: {epoch_d_loss}, D accuracy: {epoch_d_accuracy}, G loss: {epoch_g_loss}')

    # モデルの保存
    generator.save('generator.h5')
    #discriminator.save('discriminator.h5')
    print("Generator and Discriminator models saved.")
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

#############################################################################################
## 構築したモデルを用いて、速度場を生成する
generator = tf.keras.models.load_model('generator.h5')
output_directory = "data/data-wallshear-est"
delete_bin_files(output_directory)
for index in range(3001, 3500):
    batch_input_data_list = []
    # ux-データを読み込み
    ux_file = os.path.join(input_directory, f'ux-{index}.bin')
    ux_data = np.fromfile(ux_file, dtype=np.float64).reshape(shape)
            
    # uz-データを読み込み
    uz_file = os.path.join(input_directory, f'uz-{index}.bin')
    uz_data = np.fromfile(uz_file, dtype=np.float64).reshape(shape)
            
    # スライスしたデータを保持するリスト
    sliced_ux_data_list = []
    sliced_uz_data_list = []
            
    for j in range(1,5):
        sliced_ux_data = ux_data[:, j, :]
        sliced_uz_data = uz_data[:, j, :]
        sliced_ux_data_list.append(sliced_ux_data)
        sliced_uz_data_list.append(sliced_uz_data)
            
    # (32, 32, 4) の形にデータを結合
    batch_ux_data = np.stack(sliced_ux_data_list, axis=-1)
    batch_uz_data = np.stack(sliced_uz_data_list, axis=-1)
            
    # ux-とuz-データを結合して (32, 32, 8) にする
    batch_combined_data = np.concatenate((batch_ux_data, batch_uz_data), axis=-1)
    batch_input_data_list.append(batch_combined_data)

    # バッチサイズ分のデータを (batch_size, 32, 32, 8) に結合
    batch_input_data = np.array(batch_input_data_list)
    
    generated_data = generator.predict(batch_input_data)
    generated_data = generated_data.squeeze()
    
    save_generated_bin_data(index, generated_data, output_directory)

print("All generated and transformed data saved.")
