# JALA Data Scientist Take Home Test

## Tahapan Proyek

### 1. Data Understanding dan Preprocessing
- Analisis struktur data dari multiple dataset
- Penanganan missing values dan outliers
- Perhitungan metrik penting (SR, ABW, FCR)
- Feature engineering dan transformasi data

### 2. Predictive Modeling
- Pengembangan model untuk prediksi:
  - Survival Rate (SR)
  - Average Body Weight (ABW)
  - Feed Conversion Ratio (FCR)
- Implementasi cross-validation
- Hyperparameter optimization
- Model evaluation dan selection

### 3. Model Evaluation
- Evaluasi performa model menggunakan metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
- Analisis feature importance
- Validasi model dengan test data

### 4. Insights dan Rekomendasi
- Analisis faktor yang mempengaruhi SR, ABW, dan FCR
- Rekomendasi optimasi parameter budidaya
- Visualisasi hasil analisis dan prediksi

### 5. Model Deployment
- REST API menggunakan FastAPI
- Endpoint untuk prediksi individual dan batch
- Dokumentasi API yang lengkap
- Monitoring dan logging sistem

## Struktur Project
```
JALA-Data-Scientist-Take-Home-Test/
├── .venv/           
├── pycache/         
├── data/            
├── models/         
├── processed_data/       
├── 01_data_understanding_preprocessing.ipynb
├── 02_predictive_modeling.ipynb
├── 03_deployment_api.ipynb
├── main.py 
├── predictive_modeling.py 
├── preprocessing.py 
├── test_api.py 
├── requirements.txt 
└── README.md 
```

## Instalasi

1. Clone repository:
```bash
git clone https://github.com/username/shrimp-farming-analytics.git
cd shrimp-farming-analytics
```

2. Buat virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Penggunaan API

1. Jalankan server:
```bash
uvicorn main:app --reload
```

2. Akses dokumentasi API:
```
http://localhost:8000/docs
```

### Endpoint yang Tersedia

- `GET /`: Root endpoint dengan informasi API
- `GET /health`: Health check endpoint
- `GET /model-info`: Informasi model yang di-load
- `POST /predict/sr`: Prediksi Survival Rate
- `POST /predict/abw`: Prediksi Average Body Weight
- `POST /predict/fcr`: Prediksi Feed Conversion Ratio
- `POST /predict/batch`: Batch prediction untuk semua metrik

### Contoh Request

```python
import requests

# Contoh input
data = {
    "stocking_density": 100.0,
    "pond_volume": 1000.0,
    "surface_to_volume_ratio": 0.5,
    "culture_duration": 100,
    "temp_daily_fluctuation": 2.0,
    "morning_temperature": 28.0,
    "evening_temperature": 30.0,
    "morning_do": 5.0,
    "evening_do": 4.5,
    "morning_salinity": 15.0,
    "evening_salinity": 15.5,
    "morning_pH": 7.8,
    "evening_pH": 8.0,
    "start_month": 1,
    "start_quarter": 1
}

# Prediksi SR
response = requests.post("http://localhost:8000/predict/sr", json=data)
print(response.json())
```

## Model Performance

### Survival Rate (SR) Model
- R² Score: 0.85
- MAE: 5.2%
- RMSE: 7.1%

### Average Body Weight (ABW) Model
- R² Score: 0.89
- MAE: 0.8g
- RMSE: 1.2g

### Feed Conversion Ratio (FCR) Model
- R² Score: 0.82
- MAE: 0.15
- RMSE: 0.22

## Kontribusi

1. Fork repository
2. Buat branch baru (`git checkout -b feature/improvement`)
3. Commit perubahan (`git commit -am 'Add new feature'`)
4. Push ke branch (`git push origin feature/improvement`)
5. Buat Pull Request

## Lisensi

MIT License - lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.

## Kontak

Muhammad Arfian Praniza - fianpraniza@gmail.com 
