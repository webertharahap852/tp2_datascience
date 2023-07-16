import pandas as pd
 
# Data training
data = [
    ['Laki-Laki', 20, 8000000, 'Single', 'Kendaraan pribadi'],
    ['Laki-Laki', 35, 14000000, 'Single', 'Kendaraan umum'],
    ['Perempuan', 26, 10000000, 'Single', 'Kendaraan umum'],
    ['Perempuan', 27, 12000000, 'Menikah', 'Kendaraan pribadi'],
    ['Laki-Laki', 21, 9000000, 'Single', 'Kendaraan pribadi'],
    ['Laki-Laki', 22, 11000000, 'Single', 'Kendaraan pribadi'],
    ['Perempuan', 32, 15000000, 'Menikah', 'Kendaraan umum'],
    ['Perempuan', 26, 8000000, 'Menikah', 'Kendaraan umum'],
    ['Laki-Laki', 25, 9000000, 'Single', 'Kendaraan umum'],
    ['Perempuan', 20, 10000000, 'Single', 'Kendaraan pribadi'],
    ['Perempuan', 27, 12000000, 'Single', ''],
    ['Laki-Laki', 35, 14000000, 'Menikah', '']
]
 
# Buat dataframe dari data training
df_training = pd.DataFrame(data, columns=['Jenis Kelamin', 'Umur Karyawan', 'Gaji', 'Status', 'Transportasi'])
 
# Menghitung jumlah data training
total_data = len(df_training)
 
# Membagi data training berdasarkan kelas target
df_pribadi = df_training[df_training['Transportasi'] == 'Kendaraan pribadi']
df_umum = df_training[df_training['Transportasi'] == 'Kendaraan umum']
 
# Menghitung jumlah data berdasarkan kelas target
total_pribadi = len(df_pribadi)
total_umum = len(df_umum)
 
# Fungsi untuk menghitung probabilitas kondisional
def hitung_prob_kondisional(df, feature, value, target):
    count = len(df[df[feature] == value][df['Transportasi'] == target])
    total = len(df[df['Transportasi'] == target])
    if total == 0:
        return 0
    return count / total
 
# Data baru untuk diprediksi (no 11 dan 12)
data_baru = [
    ['Perempuan', 27, 12000000, 'Single'],
    ['Laki-Laki', 35, 14000000, 'Menikah']
]
 
# Prediksi jenis transportasi dari data baru (no 11 dan 12)
for i in range(10, 12):
    data = df_training.loc[i, ['Jenis Kelamin', 'Umur Karyawan', 'Gaji', 'Status']]
 
    prob_pribadi = total_pribadi / total_data
    prob_umum = total_umum / total_data
 
    for j in range(len(data)):
        feature = df_training.columns[j]
        value = data[j]
        prob_pribadi *= hitung_prob_kondisional(df_pribadi, feature, value, 'Kendaraan pribadi')
        prob_umum *= hitung_prob_kondisional(df_umum, feature, value, 'Kendaraan umum')
 
    # Membandingkan probabilitas untuk memutuskan kelas target
    if prob_pribadi > prob_umum:
        prediksi = 'Kendaraan pribadi'
    else:
        prediksi = 'Kendaraan umum'
 
    # Tampilkan hasil prediksi hanya untuk data 11 dan 12
    if i == 10:
        print("Data 11:")
    else:
        print("Data 12:")
    print("Jenis Kelamin:", data[0])
    print("Umur Karyawan:", data[1])
    print("Gaji:", data[2])
    print("Status:", data[3])
    print("Transportasi:", prediksi)
    print()
 