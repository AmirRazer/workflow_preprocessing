import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# --- KONFIGURASI ---
DATA_URL = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv"
TARGET_COLUMN = 'Status'
OUTPUT_TRAIN = 'train_data_processed.csv'
OUTPUT_TEST = 'test_data_processed.csv'
MODEL_DIR = 'model'


PCA_COLS_1 = [
    'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
]

PCA_COLS_2 = [
    'Mothers_qualification', 'Fathers_qualification',
    'Mothers_occupation', 'Fathers_occupation'
]

NUMERIC_COLS = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade',
    'Nacionality', 'Admission_grade', 'Displaced', 'Educational_special_needs',
    'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International', 'Unemployment_rate', 'Inflation_rate', 'GDP'
]

def init_folders():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Folder '{MODEL_DIR}' berhasil dibuat.")

def load_data(url):
    print(f"Mengunduh data dari: {url} ...")
    df = pd.read_csv(url, sep=";")
    print(f"Data dimuat. Dimensi: {df.shape}")
    return df

def split_data(df):
    print("Membagi data training dan testing (80/20)...")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def encode_target(y_train, y_test):
    print("Encoding target column (Status)...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Simpan encoder target
    joblib.dump(le, os.path.join(MODEL_DIR, 'target_encoder.joblib'))
    
    return y_train_enc, y_test_enc

def custom_scaling(features, df_train, df_test):
    print("Melakukan Custom Scaling (MinMaxScaler per fitur)...")
    
    
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    for feature in features:
        scaler = MinMaxScaler()
        
        # Reshape karena scaler butuh array 2D
        X_col = np.asanyarray(df_train_scaled[feature]).reshape(-1, 1)
        scaler.fit(X_col)

        df_train_scaled[feature] = scaler.transform(X_col)
        
        X_test_col = np.asanyarray(df_test_scaled[feature]).reshape(-1, 1)
        df_test_scaled[feature] = scaler.transform(X_test_col)
        
        # Simpan Scaler untuk setiap fitur
        save_path = os.path.join(MODEL_DIR, f'scaler_{feature}.joblib')
        joblib.dump(scaler, save_path)
    
    print(f"Scaler tersimpan di folder '{MODEL_DIR}'.")
    return df_train_scaled, df_test_scaled

def apply_pca_1(df_train, df_test):
    print("Melakukan PCA Tahap 1 (Curricular Units)...")
    n_comp = 4
    pca = PCA(n_components=n_comp, random_state=123)
    
    # Fit pada data train
    pca.fit(df_train[PCA_COLS_1])

    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_1.joblib'))

    train_pca = pca.transform(df_train[PCA_COLS_1])
    test_pca = pca.transform(df_test[PCA_COLS_1])

    cols = [f"pc1_{i+1}" for i in range(n_comp)]
    

    df_train[cols] = pd.DataFrame(train_pca, columns=cols, index=df_train.index)
    df_test[cols] = pd.DataFrame(test_pca, columns=cols, index=df_test.index)

    df_train.drop(columns=PCA_COLS_1, inplace=True)
    df_test.drop(columns=PCA_COLS_1, inplace=True)
    
    return df_train, df_test

def apply_pca_2(df_train, df_test):
    print("Melakukan PCA Tahap 2 (Parents Info)...")
    n_comp = 2
    pca = PCA(n_components=n_comp, random_state=123)
    
    # Fit pada data train
    pca.fit(df_train[PCA_COLS_2])

    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_2.joblib'))

    train_pca = pca.transform(df_train[PCA_COLS_2])
    test_pca = pca.transform(df_test[PCA_COLS_2])
    
    # Buat nama kolom baru
    cols = [f"pc2_{i+1}" for i in range(n_comp)]
    
    # Masukkan hasil ke DataFrame
    df_train[cols] = pd.DataFrame(train_pca, columns=cols, index=df_train.index)
    df_test[cols] = pd.DataFrame(test_pca, columns=cols, index=df_test.index)
    
    # Hapus kolom lama
    df_train.drop(columns=PCA_COLS_2, inplace=True)
    df_test.drop(columns=PCA_COLS_2, inplace=True)
    
    return df_train, df_test

def save_processed_data(X_train, X_test, y_train, y_test):
    print("Menyimpan data final...")
    
    # Gabungkan fitur dan target
    train_set = X_train.copy()
    train_set[TARGET_COLUMN] = y_train
    
    test_set = X_test.copy()
    test_set[TARGET_COLUMN] = y_test
    
    train_set.to_csv(OUTPUT_TRAIN, index=False)
    test_set.to_csv(OUTPUT_TEST, index=False)
    
    print(f"Data training disimpan ke: {OUTPUT_TRAIN}")
    print(f"Data testing disimpan ke: {OUTPUT_TEST}")

if __name__ == "__main__":
    
    init_folders()
    
    
    df = load_data(DATA_URL)
    
   
    X_train, X_test, y_train, y_test = split_data(df)
    
    
    y_train_enc, y_test_enc = encode_target(y_train, y_test)
    
    
    X_train_scaled, X_test_scaled = custom_scaling(NUMERIC_COLS, X_train, X_test)

    X_train_pca1, X_test_pca1 = apply_pca_1(X_train_scaled, X_test_scaled)

    X_train_final, X_test_final = apply_pca_2(X_train_pca1, X_test_pca1)

    save_processed_data(X_train_final, X_test_final, y_train_enc, y_test_enc)
    
    print("\n=== Proses Otomatisasi Selesai ===")
    print("File yang dihasilkan:")
    print(f" - {OUTPUT_TRAIN}")
    print(f" - {OUTPUT_TEST}")
    print(f" - Folder '{MODEL_DIR}/' berisi scaler_*.joblib, pca_1.joblib, pca_2.joblib")