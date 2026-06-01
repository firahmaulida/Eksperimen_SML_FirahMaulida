# Eksperimen_SML_FirahMaulida

> **Kriteria 1 — Eksperimen Dataset**  
> Course: Membangun Sistem Machine Learning | Dicoding  
> Author: FirahMaulida | Python 3.12.7

---

## Struktur Proyek

```
Eksperimen_SML_FirahMaulida/
├── .github/
│   └── workflows/
│       └── preprocessing.yml        # CI workflow GitHub Actions
├── namadataset_raw/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── preprocessing/
│   ├── Eksperimen_FirahMaulida.ipynb   # Notebook EDA + preprocessing
│   └── automate_FirahMaulida.py        # Script otomatisasi
├── preprocessed_output/             # Dibuat otomatis saat pipeline berjalan
│   └── telco_churn_preprocessed.csv
├── requirements.txt
└── README.md
```

---

## Dataset

**Telco Customer Churn** — IBM sample dataset  
- 7.043 baris × 21 kolom  
- Target: `Churn` (Yes/No) — apakah pelanggan berhenti berlangganan  
- Sumber: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Cara Menjalankan

### 1. Clone & Install Dependencies

```bash
git clone <repo-url>
cd Eksperimen_SML_FirahMaulida
pip install -r requirements.txt
```

### 2. Jalankan Script Otomatisasi

```bash
cd preprocessing
python automate_FirahMaulida.py
```

### 3. Buka Notebook

```bash
jupyter notebook preprocessing/Eksperimen_FirahMaulida.ipynb
```

---

## CI/CD Pipeline

Setiap `push` ke branch `main` akan secara otomatis:

1. Setup Python 3.12.7
2. Install dependencies
3. Menjalankan `automate_FirahMaulida.py`
4. Menyimpan hasil preprocessing sebagai **artifact** di GitHub Actions (retention 30 hari)

---

## Tahap Preprocessing

| Tahap | Teknik | Fitur |
|-------|--------|-------|
| Data Cleaning | Konversi + Median Imputation | `TotalCharges` |
| Feature Engineering | Drop column | `customerID` |
| Encoding | Label Encoding | `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn` |
| Encoding | One-Hot Encoding | `MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`, dll. |
| Scaling | StandardScaler | `tenure`, `MonthlyCharges`, `TotalCharges` |
