"""
automate_FirahMaulida.py
========================
Script otomatisasi preprocessing dataset Telco Customer Churn.
Digunakan untuk pipeline CI/CD via GitHub Actions.

Author  : FirahMaulida
Course  : Membangun Sistem Machine Learning - Dicoding
Python  : 3.12.7
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Konfigurasi Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocessing.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "namadataset_raw",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "preprocessed_output")

BINARY_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]
MULTICLASS_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------------------------------------------------------------------------
# Fungsi-fungsi utama
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset CSV dari path yang diberikan.

    Parameters
    ----------
    filepath : str
        Lokasi file CSV dataset mentah.

    Returns
    -------
    pd.DataFrame
        DataFrame berisi data mentah.

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan di path yang diberikan.
    """
    logger.info("Memuat dataset dari: %s", filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Dataset berhasil dimuat. Shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dataset:
    - Mengonversi kolom TotalCharges dari object ke numeric.
    - Mengisi missing values dengan median.
    - Menghapus kolom customerID.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mentah.

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah dibersihkan.
    """
    logger.info("Memulai proses data cleaning...")
    df = df.copy()

    # Konversi TotalCharges ke numeric (string kosong → NaN)
    before = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    after = df["TotalCharges"].isna().sum()
    logger.info(
        "TotalCharges: %d nilai kosong baru ditemukan setelah konversi "
        "(total NaN sebelum=%d, sesudah=%d).",
        after - before, before, after,
    )

    # Isi missing values numerik dengan median
    median_val = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
    logger.info("Missing values TotalCharges diisi dengan median = %.2f", median_val)

    # Hapus kolom customerID (tidak relevan untuk model)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        logger.info("Kolom 'customerID' berhasil dihapus.")

    logger.info("Data cleaning selesai. Shape sekarang: %s", df.shape)
    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan Label Encoding pada fitur biner (2 kategori unik).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur biner yang sudah di-encode.
    """
    logger.info("Melakukan Label Encoding untuk %d fitur biner...", len(BINARY_FEATURES))
    df = df.copy()
    le = LabelEncoder()
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            logger.info("  └─ %s: %s → %s", col, list(le.classes_), list(range(len(le.classes_))))
    return df


def encode_multiclass_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan One-Hot Encoding pada fitur kategori multi-label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan OHE diterapkan dan kolom asli dihapus.
    """
    logger.info("Melakukan One-Hot Encoding untuk %d fitur multi-label...", len(MULTICLASS_FEATURES))
    cols_to_encode = [c for c in MULTICLASS_FEATURES if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
    # Pastikan tipe data bool hasil OHE dikonversi ke int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    logger.info("One-Hot Encoding selesai. Shape sekarang: %s", df.shape)
    return df


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan StandardScaler pada fitur numerik kontinyu.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur numerik yang sudah di-scale.
    """
    logger.info("Melakukan StandardScaler untuk fitur numerik: %s", NUMERIC_FEATURES)
    df = df.copy()
    cols_to_scale = [c for c in NUMERIC_FEATURES if c in df.columns]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    logger.info("Scaling selesai. Mean per fitur: %s",
                dict(zip(cols_to_scale, np.round(scaler.mean_, 3))))
    return df


def save_output(df: pd.DataFrame, output_dir: str) -> str:
    """
    Menyimpan DataFrame hasil preprocessing ke file CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame hasil preprocessing.
    output_dir : str
        Direktori tujuan penyimpanan.

    Returns
    -------
    str
        Path lengkap file output yang disimpan.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "telco_churn_preprocessed.csv")
    df.to_csv(output_path, index=False)
    logger.info("Output disimpan di: %s | Shape akhir: %s", output_path, df.shape)
    return output_path


def run_pipeline(raw_data_path: str = RAW_DATA_PATH, output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan.

    Parameters
    ----------
    raw_data_path : str
        Path ke file CSV mentah.
    output_dir : str
        Direktori untuk menyimpan hasil preprocessing.

    Returns
    -------
    pd.DataFrame
        DataFrame final setelah seluruh tahap preprocessing.
    """
    logger.info("=" * 60)
    logger.info("  PIPELINE PREPROCESSING TELCO CUSTOMER CHURN")
    logger.info("  Author: FirahMaulida")
    logger.info("=" * 60)

    # Tahap 1: Load data
    df = load_data(raw_data_path)

    # Tahap 2: Data cleaning & feature engineering
    df = clean_data(df)

    # Tahap 3: Encoding fitur biner
    df = encode_binary_features(df)

    # Tahap 4: One-Hot Encoding fitur multi-label
    df = encode_multiclass_features(df)

    # Tahap 5: Scaling fitur numerik
    df = scale_numeric_features(df)

    # Tahap 6: Simpan output
    save_output(df, output_dir)

    logger.info("=" * 60)
    logger.info("  PIPELINE SELESAI DENGAN SUKSES!")
    logger.info("  Total fitur akhir : %d kolom", df.shape[1])
    logger.info("  Total baris       : %d sampel", df.shape[0])
    logger.info("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
