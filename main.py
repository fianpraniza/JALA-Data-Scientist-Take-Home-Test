# Import library yang diperlukan
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Buat folder models jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Definisi struktur data input
class PredictionInput(BaseModel):
    # Input features untuk prediksi
    stocking_density: float
    pond_volume: float
    surface_to_volume_ratio: float
    culture_duration: int
    temp_daily_fluctuation: float
    morning_temperature: float
    evening_temperature: float
    morning_do: float
    evening_do: float
    morning_salinity: float
    evening_salinity: float
    morning_pH: float
    evening_pH: float
    start_month: int
    start_quarter: int

class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

# 2. Load model dan scaler
def load_models(model_dir: str):
    """
    Load semua model dan scaler yang telah disimpan dengan error handling yang lebih baik
    """
    models = {}
    scalers = {}
    
    try:
        # Cek keberadaan folder
        if not os.path.exists(model_dir):
            print(f"Warning: Folder {model_dir} tidak ditemukan. Membuat folder baru.")
            os.makedirs(model_dir)
            return models, scalers
        
        # Load model dan scaler jika ada
        model_files = {
            'sr': ('sr_model.joblib', 'sr_scaler.joblib'),
            'abw': ('abw_model.joblib', 'abw_scaler.joblib'),
            'fcr': ('fcr_model.joblib', 'fcr_scaler.joblib')
        }
        
        for model_name, (model_file, scaler_file) in model_files.items():
            model_path = os.path.join(model_dir, model_file)
            scaler_path = os.path.join(model_dir, scaler_file)
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                models[model_name] = joblib.load(model_path)
                scalers[model_name] = joblib.load(scaler_path)
                print(f"Loaded {model_name} model and scaler successfully")
            else:
                print(f"Warning: {model_name} model or scaler not found")
        
        return models, scalers
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return models, scalers

# Tambahkan di main.py untuk keperluan testing
def create_mock_models():
    """
    Membuat mock model untuk testing
    """
    # Create mock models
    mock_models = {
        'sr': RandomForestRegressor(n_estimators=10).fit(np.random.rand(100, 15), np.random.rand(100)),
        'abw': RandomForestRegressor(n_estimators=10).fit(np.random.rand(100, 15), np.random.rand(100))
    }
    
    # Create mock scalers
    from sklearn.preprocessing import StandardScaler
    mock_scalers = {
        'sr': StandardScaler().fit(np.random.rand(100, 15)),
        'abw': StandardScaler().fit(np.random.rand(100, 15))
    }
    
    # Save mock models and scalers
    for name in ['sr', 'abw']:
        joblib.dump(mock_models[name], f'models/{name}_model.joblib')
        joblib.dump(mock_scalers[name], f'models/{name}_scaler.joblib')
    
    return mock_models, mock_scalers

# 3. Inisialisasi FastAPI
app = FastAPI(
    title="Shrimp Farming Prediction API",
    description="""
    API untuk prediksi parameter budidaya udang menggunakan machine learning.
    
    ## Fitur Utama
    * Prediksi Survival Rate (SR)
    * Prediksi Average Body Weight (ABW)
    * Prediksi Feed Conversion Ratio (FCR)
    * Batch prediction untuk semua metrik
    
    ## Panduan Penggunaan
    1. Gunakan endpoint `/predict/{metric}` untuk prediksi individual
    2. Gunakan endpoint `/predict/batch` untuk prediksi multiple data
    3. Cek status API melalui endpoint `/health`
    4. Lihat informasi model melalui endpoint `/model-info`
    
    ## Input Parameters
    * stocking_density: Kepadatan tebar (ekor/m³)
    * pond_volume: Volume kolam (m³)
    * surface_to_volume_ratio: Rasio permukaan terhadap volume
    * culture_duration: Durasi budidaya (hari)
    * temp_daily_fluctuation: Fluktuasi suhu harian (°C)
    * morning_temperature: Suhu pagi hari (°C)
    * evening_temperature: Suhu sore hari (°C)
    * morning_do: DO pagi hari (mg/L)
    * evening_do: DO sore hari (mg/L)
    * morning_salinity: Salinitas pagi hari (ppt)
    * evening_salinity: Salinitas sore hari (ppt)
    * morning_pH: pH pagi hari
    * evening_pH: pH sore hari
    * start_month: Bulan mulai (1-12)
    * start_quarter: Kuartal mulai (1-4)
    """,
    version="1.0.0",
    contact={
        "name": "JALA Tech",
        "url": "https://jala.tech",
        "email": "info@jala.tech"
    }
)

# Load models pada startup
@app.on_event("startup")
async def startup_event():
    global models, scalers
    try:
        models, scalers = load_models('models')
        if not models:  # If no models loaded, create mock models
            print("No models found. Creating mock models for testing...")
            models, scalers = create_mock_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

# 4. Fungsi preprocessing
def preprocess_input(data: PredictionInput, scaler) -> np.ndarray:
    """
    Preprocess input data sebelum prediksi
    """
    # Convert input ke DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    
    # One-hot encoding untuk fitur kategorikal
    df = pd.get_dummies(df, columns=['start_month', 'start_quarter'], prefix=['start_month', 'start_quarter'])
    
    # Pastikan semua kolom yang dibutuhkan ada
    required_columns = scaler.feature_names_in_
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Urutkan kolom sesuai dengan training
    df = df[required_columns]
    
    # Scale features
    scaled_features = scaler.transform(df)
    
    return scaled_features

# 5. Endpoint untuk prediksi
@app.post("/predict/sr", 
    response_model=Dict[str, float],
    summary="Prediksi Survival Rate",
    description="""
    Memprediksi Survival Rate (SR) berdasarkan parameter input yang diberikan.
    
    Returns:
    - survival_rate: Persentase SR (0-100%)
    - timestamp: Waktu prediksi
    
    Notes:
    - SR diprediksi menggunakan model XGBoost
    - Hasil prediksi di-clip ke range 0-100%
    """
)
async def predict_sr(data: PredictionInput):
    """
    Endpoint untuk prediksi Survival Rate
    """
    try:
        logger.info("Processing SR prediction request")
        if 'sr' not in models or 'sr' not in scalers:
            logger.error("SR model or scaler not found")
            raise HTTPException(
                status_code=503,
                detail="SR model not available"
            )
        
        # Convert input ke format yang sesuai
        input_dict = data.dict()
        logger.info(f"Input data: {input_dict}")
        
        # Validasi input
        if not (0 <= input_dict['stocking_density'] <= 1000):
            raise HTTPException(
                status_code=400,
                detail="Stocking density harus antara 0 dan 1000"
            )
        
        # Preprocess input
        processed_input = preprocess_input(data, scalers['sr'])
        logger.info("Input preprocessed successfully")
        
        # Make prediction
        prediction = models['sr'].predict(processed_input)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Clip prediction ke range yang valid
        prediction = float(np.clip(prediction, 0, 100))
        logger.info(f"Clipped prediction: {prediction}")
        
        return {
            "survival_rate": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in SR prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting SR: {str(e)}"
        )

@app.post("/predict/abw",
    response_model=Dict[str, float],
    summary="Prediksi Average Body Weight",
    description="""
    Memprediksi Average Body Weight (ABW) berdasarkan parameter input yang diberikan.
    
    Returns:
    - average_body_weight: Berat rata-rata udang (gram)
    - timestamp: Waktu prediksi
    
    Notes:
    - ABW diprediksi menggunakan model XGBoost
    - Hasil prediksi selalu positif
    """
)
async def predict_abw(data: PredictionInput):
    """
    Endpoint untuk prediksi Average Body Weight
    """
    try:
        logger.info("Processing ABW prediction request")
        if 'abw' not in models or 'abw' not in scalers:
            logger.error("ABW model or scaler not found")
            raise HTTPException(
                status_code=503,
                detail="ABW model not available"
            )
        
        # Convert input ke format yang sesuai
        input_dict = data.dict()
        logger.info(f"Input data: {input_dict}")
        
        # Validasi input
        if input_dict['culture_duration'] <= 0:
            raise HTTPException(
                status_code=400,
                detail="Culture duration harus lebih dari 0"
            )
        
        # Preprocess input
        processed_input = preprocess_input(data, scalers['abw'])
        logger.info("Input preprocessed successfully")
        
        # Make prediction
        prediction = models['abw'].predict(processed_input)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Clip prediction ke range yang valid
        prediction = float(max(0, prediction))
        logger.info(f"Clipped prediction: {prediction}")
        
        return {
            "average_body_weight": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in ABW prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting ABW: {str(e)}"
        )

@app.post("/predict/fcr",
    response_model=Dict[str, float],
    summary="Prediksi Feed Conversion Ratio",
    description="""
    Memprediksi Feed Conversion Ratio (FCR) berdasarkan parameter input yang diberikan.
    
    Returns:
    - fcr: Nilai FCR (0.5-3.0)
    - timestamp: Waktu prediksi
    
    Notes:
    - FCR diprediksi menggunakan model LightGBM
    - Hasil prediksi di-clip ke range 0.5-3.0
    """
)
async def predict_fcr(data: PredictionInput):
    """
    Endpoint untuk prediksi Feed Conversion Ratio
    """
    try:
        logger.info("Processing FCR prediction request")
        if 'fcr' not in models or 'fcr' not in scalers:
            logger.error("FCR model or scaler not found")
            raise HTTPException(
                status_code=503,
                detail="FCR model not available"
            )
        
        # Convert input ke format yang sesuai
        input_dict = data.dict()
        logger.info(f"Input data: {input_dict}")
        
        # Validasi input
        if input_dict['culture_duration'] <= 0:
            raise HTTPException(
                status_code=400,
                detail="Culture duration harus lebih dari 0"
            )
        
        # Preprocess input
        processed_input = preprocess_input(data, scalers['fcr'])
        logger.info("Input preprocessed successfully")
        
        # Make prediction
        prediction = models['fcr'].predict(processed_input)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Clip prediction ke range yang valid (0.5-3.0)
        prediction = float(np.clip(prediction, 0.5, 3.0))
        logger.info(f"Clipped prediction: {prediction}")
        
        return {
            "fcr": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in FCR prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting FCR: {str(e)}"
        )

# 6. Endpoint untuk batch prediction
@app.post("/predict/batch",
    response_model=Dict[str, List[Dict[str, float]]],
    summary="Batch Prediction untuk Semua Metrik",
    description="""
    Melakukan prediksi SR, ABW, dan FCR secara batch untuk multiple input.
    
    Returns:
    - predictions: List hasil prediksi untuk setiap input
      - survival_rate: Persentase SR (0-100%)
      - average_body_weight: Berat rata-rata udang (gram)
      - fcr: Nilai FCR (0.5-3.0)
    
    Notes:
    - Maksimum 100 data per request
    - Semua prediksi menggunakan model yang sama dengan endpoint individual
    """
)
async def predict_batch(data: BatchPredictionInput):
    """
    Endpoint untuk batch prediction semua metrik
    """
    try:
        results = {
            "predictions": []
        }
        
        for input_data in data.data:
            # Get predictions for all metrics
            sr_pred = await predict_sr(input_data)
            abw_pred = await predict_abw(input_data)
            
            prediction = {
                "survival_rate": sr_pred["survival_rate"],
                "average_body_weight": abw_pred["average_body_weight"]
            }
            
            # Add FCR if available
            if 'fcr' in models:
                fcr_pred = await predict_fcr(input_data)
                prediction["fcr"] = fcr_pred["fcr"]
            
            results["predictions"].append(prediction)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. Health check endpoint
@app.get("/health")
async def health_check():
    """
    Endpoint untuk health check
    """
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

# 8. Model info endpoint
@app.get("/model-info")
async def model_info():
    """
    Endpoint untuk mendapatkan informasi tentang model yang telah di-load
    """
    try:
        model_information = {
            "models": {
                "sr": {
                    "type": str(type(models['sr']).__name__),
                    "features": list(scalers['sr'].feature_names_in_),
                    "status": "loaded" if 'sr' in models else "not loaded"
                },
                "abw": {
                    "type": str(type(models['abw']).__name__),
                    "features": list(scalers['abw'].feature_names_in_),
                    "status": "loaded" if 'abw' in models else "not loaded"
                },
                "fcr": {
                    "type": str(type(models['fcr']).__name__),
                    "features": list(scalers['fcr'].feature_names_in_),
                    "status": "loaded" if 'fcr' in models else "not loaded"
                }
            },
            "last_updated": datetime.now().isoformat(),
            "api_version": "1.0.0"
        }
        return model_information
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model information: {str(e)}"
        )

# Tambahkan root endpoint
@app.get("/")
async def root():
    """
    Root endpoint yang menampilkan informasi dasar API
    """
    return {
        "message": "Welcome to Shrimp Farming Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "Prediction Endpoints": {
                "/predict/sr": "Predict Survival Rate",
                "/predict/abw": "Predict Average Body Weight",
                "/predict/fcr": "Predict Feed Conversion Ratio",
                "/predict/batch": "Batch Prediction for all metrics"
            },
            "Information Endpoints": {
                "/health": "API Health Check",
                "/model-info": "Model Information",
                "/docs": "API Documentation"
            }
        },
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

# Tambahkan contoh input yang valid
sample_input = {
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

batch_input = {
    "data": [sample_input, sample_input]
}

def test_api():
    """
    Test API dengan logging yang lebih detail
    """
    BASE_URL = "http://127.0.0.1:8000"
    results = {
        "success": [],
        "failed": []
    }
    
    def log_test(endpoint, response=None, error=None):
        if error:
            results["failed"].append({
                "endpoint": endpoint,
                "error": str(error)
            })
            print(f"❌ {endpoint}: {str(error)}")
        else:
            results["success"].append({
                "endpoint": endpoint,
                "status_code": response.status_code
            })
            print(f"✅ {endpoint}: {response.status_code}")
            print(f"Response: {response.json()}\n")
    
    # Test endpoints
    endpoints = [
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/model-info"),
        ("POST", "/predict/sr"),
        ("POST", "/predict/abw"),
        ("POST", "/predict/batch")
    ]
    
    headers = {"Content-Type": "application/json"}
    
    for method, endpoint in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}")
            else:
                data = batch_input if endpoint == "/predict/batch" else sample_input
                response = requests.post(
                    f"{BASE_URL}{endpoint}",
                    json=data,
                    headers=headers
                )
            log_test(endpoint, response)
        except Exception as e:
            log_test(endpoint, error=e)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Success: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['failed']:
        print("\nFailed Tests:")
        for fail in results['failed']:
            print(f"- {fail['endpoint']}: {fail['error']}")

if __name__ == "__main__":
    test_api()