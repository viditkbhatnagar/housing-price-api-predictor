import requests
import json

# Define the API endpoint
# For local testing: "http://localhost:8000"
# For Kubernetes: Replace with your service URL
API_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    response = requests.get(f"{API_URL}/")
    print("Root endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print("Health check response:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{API_URL}/model/info")
    print("Model info response:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_prediction():
    """Test prediction endpoint"""
    # Test data
    test_features = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    response = requests.post(f"{API_URL}/predict", json=test_features)
    print("Prediction response:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_multiple_predictions():
    """Test multiple predictions"""
    test_cases = [
        {
            "name": "Low income area",
            "features": {
                "MedInc": 2.5,
                "HouseAge": 30.0,
                "AveRooms": 4.0,
                "AveBedrms": 1.0,
                "Population": 1000.0,
                "AveOccup": 3.0,
                "Latitude": 34.0,
                "Longitude": -118.0
            }
        },
        {
            "name": "High income area",
            "features": {
                "MedInc": 10.0,
                "HouseAge": 5.0,
                "AveRooms": 8.0,
                "AveBedrms": 2.0,
                "Population": 500.0,
                "AveOccup": 2.0,
                "Latitude": 37.5,
                "Longitude": -122.5
            }
        }
    ]
    
    print("Multiple prediction tests:")
    for test_case in test_cases:
        response = requests.post(f"{API_URL}/predict", json=test_case["features"])
        result = response.json()
        print(f"{test_case['name']}: ${result['predicted_price']*100000:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    print("Testing Housing Price Prediction API")
    print("=" * 50)
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_prediction()
        test_multiple_predictions()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure it's running!")
    except Exception as e:
        print(f"Error: {e}")