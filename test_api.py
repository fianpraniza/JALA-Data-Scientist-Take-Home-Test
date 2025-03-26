# test_api.py
import requests
import json
from requests.exceptions import RequestException
import time

# Contoh input yang valid
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

def wait_for_server(base_url: str, max_retries: int = 5, delay: int = 2):
    """
    Menunggu server siap menerima request
    """
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except RequestException:
            print(f"Waiting for server... ({i+1}/{max_retries})")
            time.sleep(delay)
    return False

def test_api():
    """
    Test API dengan logging yang lebih detail dan error handling yang lebih baik
    """
    BASE_URL = "http://127.0.0.1:8000"
    results = {
        "success": [],
        "failed": []
    }
    
    # Tunggu server siap
    if not wait_for_server(BASE_URL):
        print("Error: Server tidak dapat diakses")
        return
    
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
            try:
                print(f"Response: {response.json()}\n")
            except json.JSONDecodeError:
                print(f"Response: {response.text}\n")
    
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
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            else:
                data = batch_input if endpoint == "/predict/batch" else sample_input
                response = requests.post(
                    f"{BASE_URL}{endpoint}",
                    json=data,
                    headers=headers,
                    timeout=10
                )
            log_test(endpoint, response)
        except RequestException as e:
            log_test(endpoint, error=f"Request error: {str(e)}")
        except Exception as e:
            log_test(endpoint, error=f"Unexpected error: {str(e)}")
    
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