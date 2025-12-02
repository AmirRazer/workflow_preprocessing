import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- KONFIGURASI ---
DATA_URL = "../students_performance_dataset.csv"
TARGET_COLUMN = 'Status' # Ganti ini jika nama kolom target Anda berbeda
OUTPUT_TRAIN = 'train_data_processed.csv'
OUTPUT_TEST = 'test_data_processed.csv'

def load_data(url):
    # print(f"Mengunduh data dari: {url} ...")
    # Menggunakan sep=";" sesuai dataset Anda sebelumnya
    df = pd.read_csv(url, sep=",")
    print(f"Data berhasil dimuat. Dimensi awal: {df.shape}")
    return df

def preprocess(df):
    print("Memulai preprocessing...")
    
    # 1. Handling Missing Values (Opsional, sesuaikan kebutuhan)
    # df = df.dropna() 

    # 2. Encoding Data Kategorikal
    # Mengubah semua kolom tipe 'object' (teks) menjadi angka
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    
    for col in object_cols:
        
        df[col] = le.fit_transform(df[col])
        
    print("Encoding selesai.")
    return df

def split_and_scale(df):
    # Pisahkan Fitur (X) dan Target (y)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Split Train dan Test (80% Train, 20% Test)
    print("Membagi data training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling (Standarisasi)
    # Penting: Fit pada training, Transform pada training & test
    print("Melakukan scaling (StandardScaler)...")
    scaler = StandardScaler()
    
    # ubah kembali ke DataFrame agar nama kolom tidak hilang saat disimpan
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Kembalikan index agar pas saat digabung dengan y
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    print("Menyimpan data hasil proses...")
    
    # Gabungkan kembali X dan y untuk disimpan dalam satu file CSV
    train_set = X_train.copy()
    train_set[TARGET_COLUMN] = y_train
    
    test_set = X_test.copy()
    test_set[TARGET_COLUMN] = y_test
    
    train_set.to_csv(OUTPUT_TRAIN, index=False)
    test_set.to_csv(OUTPUT_TEST, index=False)
    
    print(f"SUKSES! Data tersimpan di '{OUTPUT_TRAIN}' dan '{OUTPUT_TEST}'")

if __name__ == "__main__":
    # Eksekusi
    df = load_data(DATA_URL)
    df_clean = preprocess(df)
    X_train, X_test, y_train, y_test = split_and_scale(df_clean)
    save_data(X_train, X_test, y_train, y_test)