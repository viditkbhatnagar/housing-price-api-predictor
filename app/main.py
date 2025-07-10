from fastapi import FastAPI, HTTPException, Query, Path, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize metrics storage
prediction_history = []
api_metrics = {
    "total_predictions": 0,
    "total_errors": 0,
    "start_time": datetime.now(),
    "last_prediction": None
}

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="🏠 Advanced Housing Price Prediction API",
    description="""
    ## Welcome to the Advanced Housing Price Prediction API!
    
    This API provides sophisticated machine learning predictions for California housing prices using a Linear Regression model.
    
    ### Features:
    * 🔮 **Single & Batch Predictions** - Predict one or multiple properties
    * 📊 **Statistical Analysis** - Get insights about your predictions
    * 📈 **Performance Metrics** - Monitor API usage and model performance
    * 🏘️ **Neighborhood Analysis** - Compare properties by location
    * 📝 **Prediction History** - Track all predictions made
    * 🎯 **Model Insights** - Understand feature importance
    
    ### Model Information:
    - **Algorithm**: Linear Regression
    - **Training Data**: California Housing Dataset
    - **Output**: Median house value in hundreds of thousands of dollars
    
    ### Authentication:
    Currently open access. API key authentication coming soon.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "ML API Support",
        "email": "support@housingapi.com"
    },
    license_info={
        "name": "MIT License",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums for better documentation
class PriceRange(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LUXURY = "luxury"

class PropertyType(str, Enum):
    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi_family"

# Enhanced input schema with validation
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group (in tens of thousands)", ge=0, le=15)
    HouseAge: float = Field(..., description="Median house age in block group (years)", ge=0, le=100)
    AveRooms: float = Field(..., description="Average number of rooms per household", ge=1, le=20)
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", ge=0.5, le=10)
    Population: float = Field(..., description="Block group population", ge=0, le=50000)
    AveOccup: float = Field(..., description="Average number of household members", ge=1, le=20)
    Latitude: float = Field(..., description="Block group latitude", ge=32, le=42)
    Longitude: float = Field(..., description="Block group longitude", ge=-125, le=-114)

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
    
    @validator('AveBedrms')
    def validate_bedrooms(cls, v, values):
        if 'AveRooms' in values and v > values['AveRooms']:
            raise ValueError('Average bedrooms cannot exceed average rooms')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }

# Batch prediction input
class BatchPredictionInput(BaseModel):
    properties: List[HousingFeatures] = Field(..., description="List of properties to predict", min_items=1, max_items=100)
    include_statistics: bool = Field(False, description="Include statistical analysis of predictions")

    class Config:
        protected_namespaces = ()

# Enhanced response schemas
class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted price in hundreds of thousands")
    price_usd: float = Field(..., description="Predicted price in USD")
    price_range: PriceRange = Field(..., description="Price category")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    
    class Config:
        protected_namespaces = ()

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    statistics: Optional[Dict[str, Any]] = None
    batch_id: str
    processing_time_ms: float

    class Config:
        protected_namespaces = ()

class ModelMetrics(BaseModel):
    model_type: str
    features: List[str]
    training_score: float
    feature_importance: Dict[str, float]
    model_version: str

    class Config:
        protected_namespaces = ()

class APIHealth(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int
    error_rate: float
    last_prediction: Optional[datetime]

    class Config:
        protected_namespaces = ()

# Helper functions
def get_price_range(price: float) -> PriceRange:
    if price < 1.5:
        return PriceRange.LOW
    elif price < 3.0:
        return PriceRange.MEDIUM
    elif price < 5.0:
        return PriceRange.HIGH
    else:
        return PriceRange.LUXURY

def calculate_confidence_interval(prediction: float, std_dev: float = 0.5) -> Dict[str, float]:
    """Calculate a mock 95% confidence interval"""
    margin = 1.96 * std_dev
    return {
        "lower_bound": round(max(0, prediction - margin), 2),
        "upper_bound": round(prediction + margin, 2),
        "margin_of_error": round(margin, 2)
    }

def generate_prediction_id() -> str:
    """Generate unique prediction ID"""
    return f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

# Root endpoint
@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root():
    """
    Welcome page with API information and links.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Housing Price Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .endpoints { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .endpoint { margin: 10px 0; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .button { display: inline-block; padding: 10px 20px; background: #007bff; color: white; border-radius: 5px; margin: 5px; }
            .button:hover { background: #0056b3; color: white; text-decoration: none; }
            .status { color: #28a745; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏠 Housing Price Prediction API</h1>
            <p>Welcome to the Advanced ML-powered Housing Price Prediction API!</p>
            
            <div class="endpoints">
                <h2>Quick Links:</h2>
                <a href="/docs" class="button">📚 Interactive API Docs</a>
                <a href="/test" class="button">🧪 Test Data & Examples</a>
                <a href="/redoc" class="button">📖 Alternative Docs</a>
            </div>
            
            <div class="endpoints">
                <h2>Main Endpoints:</h2>
                <div class="endpoint">• <code>POST /api/v1/predict</code> - Single prediction</div>
                <div class="endpoint">• <code>POST /api/v1/predict/batch</code> - Batch predictions</div>
                <div class="endpoint">• <code>GET /api/v1/health</code> - API health check</div>
                <div class="endpoint">• <code>GET /api/v1/model/info</code> - Model information</div>
                <div class="endpoint">• <code>GET /test</code> - Test data generator</div>
            </div>
            
            <p><span class="status">Status: API is running ✅</span></p>
            <p><strong>Model:</strong> Linear Regression (California Housing Dataset)</p>
            <p><strong>Version:</strong> 2.0.0</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/test", response_class=HTMLResponse, tags=["Testing"])
async def get_test_data():
    """
    Interactive test data generator with copy-paste examples.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Test Data Generator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .test-case { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            h1, h2 { color: #333; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #dee2e6; }
            .copy-button { background: #28a745; color: white; border: none; padding: 5px 15px; border-radius: 3px; cursor: pointer; margin-top: 10px; }
            .copy-button:hover { background: #218838; }
            .endpoint { color: #007bff; font-weight: bold; }
            .description { color: #666; margin: 10px 0; }
            code { background: #f8f9fa; padding: 2px 5px; border-radius: 3px; }
            .alert { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        </style>
        <script>
            function copyToClipboard(elementId) {
                const text = document.getElementById(elementId).innerText;
                navigator.clipboard.writeText(text).then(() => {
                    alert('Copied to clipboard!');
                });
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🧪 API Test Data Generator</h1>
            <div class="alert">
                💡 <strong>Tip:</strong> Copy these examples and test them in <a href="/docs">API Documentation</a> or use them with cURL/Postman
            </div>
            
            <div class="test-case">
                <h2>1. Single Prediction - Luxury Property</h2>
                <p class="endpoint">POST /api/v1/predict</p>
                <p class="description">High-income Bay Area property with ocean views</p>
                <pre id="test1">{
  "MedInc": 10.5,
  "HouseAge": 5.0,
  "AveRooms": 8.5,
  "AveBedrms": 3.5,
  "Population": 250.0,
  "AveOccup": 2.8,
  "Latitude": 37.45,
  "Longitude": -122.25
}</pre>
                <button class="copy-button" onclick="copyToClipboard('test1')">Copy JSON</button>
            </div>
            
            <div class="test-case">
                <h2>2. Single Prediction - Affordable Property</h2>
                <p class="endpoint">POST /api/v1/predict</p>
                <p class="description">Lower-income inland property</p>
                <pre id="test2">{
  "MedInc": 2.5,
  "HouseAge": 35.0,
  "AveRooms": 4.2,
  "AveBedrms": 2.1,
  "Population": 2500.0,
  "AveOccup": 3.5,
  "Latitude": 34.05,
  "Longitude": -117.75
}</pre>
                <button class="copy-button" onclick="copyToClipboard('test2')">Copy JSON</button>
            </div>
            
            <div class="test-case">
                <h2>3. Batch Prediction</h2>
                <p class="endpoint">POST /api/v1/predict/batch</p>
                <p class="description">Multiple properties with statistics</p>
                <pre id="test3">{
  "properties": [
    {
      "MedInc": 8.3252,
      "HouseAge": 41.0,
      "AveRooms": 6.984127,
      "AveBedrms": 1.023810,
      "Population": 322.0,
      "AveOccup": 2.555556,
      "Latitude": 37.88,
      "Longitude": -122.23
    },
    {
      "MedInc": 3.8,
      "HouseAge": 25.0,
      "AveRooms": 5.2,
      "AveBedrms": 2.0,
      "Population": 1800.0,
      "AveOccup": 3.2,
      "Latitude": 33.79,
      "Longitude": -117.89
    },
    {
      "MedInc": 6.5,
      "HouseAge": 8.0,
      "AveRooms": 7.1,
      "AveBedrms": 2.8,
      "Population": 800.0,
      "AveOccup": 2.7,
      "Latitude": 38.52,
      "Longitude": -121.50
    }
  ],
  "include_statistics": true
}</pre>
                <button class="copy-button" onclick="copyToClipboard('test3')">Copy JSON</button>
            </div>
            
            <div class="test-case">
                <h2>4. Feature Impact Analysis</h2>
                <p class="endpoint">POST /api/v1/analysis/feature-impact?feature_to_vary=MedInc&variation_steps=5</p>
                <p class="description">Analyze how income affects price</p>
                <pre id="test4">{
  "MedInc": 5.0,
  "HouseAge": 20.0,
  "AveRooms": 6.0,
  "AveBedrms": 2.0,
  "Population": 2000.0,
  "AveOccup": 3.0,
  "Latitude": 35.0,
  "Longitude": -119.0
}</pre>
                <button class="copy-button" onclick="copyToClipboard('test4')">Copy JSON</button>
            </div>
            
            <div class="test-case">
                <h2>5. cURL Examples</h2>
                <p class="description">For command line testing (replace the URL with your deployment URL)</p>
                <pre id="curl1">curl -X POST "https://housing-price-api-predictor.onrender.com/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'</pre>
                <button class="copy-button" onclick="copyToClipboard('curl1')">Copy cURL</button>
            </div>
            
            <div class="test-case">
                <h2>6. Python Request Example</h2>
                <p class="description">For Python applications</p>
                <pre id="python1">import requests

url = "https://housing-price-api-predictor.onrender.com/api/v1/predict"
data = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

response = requests.post(url, json=data)
print(response.json())</pre>
                <button class="copy-button" onclick="copyToClipboard('python1')">Copy Python</button>
            </div>
            
            <div class="test-case">
                <h2>Quick Links</h2>
                <p>
                    <a href="/docs">📚 Interactive API Documentation</a><br>
                    <a href="/api/v1/health">🏥 Health Check</a><br>
                    <a href="/api/v1/model/info">📊 Model Information</a><br>
                    <a href="/">🏠 Home</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
# Health check endpoint
@app.get("/api/v1/health", response_model=APIHealth, tags=["Monitoring"])
async def health_check() -> APIHealth:
    """
    Check API health status and basic metrics.
    """
    uptime = (datetime.now() - api_metrics["start_time"]).total_seconds()
    error_rate = api_metrics["total_errors"] / max(api_metrics["total_predictions"], 1)
    
    return APIHealth(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        uptime_seconds=round(uptime, 2),
        total_predictions=api_metrics["total_predictions"],
        error_rate=round(error_rate, 4),
        last_prediction=api_metrics["last_prediction"]
    )

# Single prediction endpoint
@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(
    features: HousingFeatures,
    include_neighbors: bool = Query(False, description="Include similar properties analysis")
) -> PredictionResponse:
    """
    Make a single housing price prediction.
    
    Returns predicted price with confidence intervals and classification.
    """
    if model is None:
        api_metrics["total_errors"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        input_data = np.array([[
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Create response
        response = PredictionResponse(
            predicted_price=round(float(prediction), 2),
            price_usd=round(float(prediction) * 100000, 2),
            price_range=get_price_range(prediction),
            confidence_interval=calculate_confidence_interval(prediction),
            timestamp=datetime.now(),
            prediction_id=generate_prediction_id()
        )
        
        # Update metrics
        api_metrics["total_predictions"] += 1
        api_metrics["last_prediction"] = datetime.now()
        
        # Store in history
        prediction_history.append({
            "id": response.prediction_id,
            "features": features.dict(),
            "prediction": response.dict(),
            "timestamp": response.timestamp
        })
        
        # Keep only last 1000 predictions in memory
        if len(prediction_history) > 1000:
            prediction_history.pop(0)
        
        return response
    
    except Exception as e:
        api_metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(batch_input: BatchPredictionInput) -> BatchPredictionResponse:
    """
    Make predictions for multiple properties at once.
    
    Supports up to 100 properties per request with optional statistical analysis.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    predictions = []
    prices = []
    
    try:
        for property_features in batch_input.properties:
            # Make individual prediction
            input_data = np.array([[
                property_features.MedInc,
                property_features.HouseAge,
                property_features.AveRooms,
                property_features.AveBedrms,
                property_features.Population,
                property_features.AveOccup,
                property_features.Latitude,
                property_features.Longitude
            ]])
            
            prediction = model.predict(input_data)[0]
            prices.append(prediction)
            
            pred_response = PredictionResponse(
                predicted_price=round(float(prediction), 2),
                price_usd=round(float(prediction) * 100000, 2),
                price_range=get_price_range(prediction),
                confidence_interval=calculate_confidence_interval(prediction),
                timestamp=datetime.now(),
                prediction_id=generate_prediction_id()
            )
            predictions.append(pred_response)
        
        # Calculate statistics if requested
        statistics = None
        if batch_input.include_statistics and prices:
            statistics = {
                "mean_price": round(statistics.mean(prices), 2),
                "median_price": round(statistics.median(prices), 2),
                "std_dev": round(statistics.stdev(prices) if len(prices) > 1 else 0, 2),
                "min_price": round(min(prices), 2),
                "max_price": round(max(prices), 2),
                "price_range_distribution": {
                    range_type.value: sum(1 for p in predictions if p.price_range == range_type)
                    for range_type in PriceRange
                }
            }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update metrics
        api_metrics["total_predictions"] += len(predictions)
        api_metrics["last_prediction"] = datetime.now()
        
        return BatchPredictionResponse(
            predictions=predictions,
            statistics=statistics,
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        api_metrics["total_errors"] += len(batch_input.properties)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Model information endpoint
@app.get("/api/v1/model/info", response_model=ModelMetrics, tags=["Model"])
async def model_info() -> ModelMetrics:
    """
    Get detailed information about the ML model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Mock feature importance (in real scenario, calculate from model)
    feature_importance = {
        "MedInc": 0.45,
        "Latitude": 0.20,
        "Longitude": 0.15,
        "HouseAge": 0.08,
        "AveRooms": 0.05,
        "Population": 0.03,
        "AveOccup": 0.02,
        "AveBedrms": 0.02
    }
    
    return ModelMetrics(
        model_type=type(model).__name__,
        features=list(feature_importance.keys()),
        training_score=0.606,  # Mock R² score
        feature_importance=feature_importance,
        model_version="1.0.0"
    )

# Prediction history endpoint
@app.get("/api/v1/predictions/history", tags=["Analysis"])
async def get_prediction_history(
    limit: int = Query(10, ge=1, le=100, description="Number of recent predictions to return"),
    price_range: Optional[PriceRange] = Query(None, description="Filter by price range")
) -> Dict[str, Any]:
    """
    Get recent prediction history with optional filtering.
    """
    # Filter history if needed
    filtered_history = prediction_history
    if price_range:
        filtered_history = [
            h for h in prediction_history 
            if h["prediction"]["price_range"] == price_range.value
        ]
    
    # Get most recent predictions
    recent_predictions = filtered_history[-limit:]
    recent_predictions.reverse()  # Most recent first
    
    return {
        "total_stored": len(prediction_history),
        "returned": len(recent_predictions),
        "predictions": recent_predictions
    }

# Statistics endpoint
@app.get("/api/v1/predictions/statistics", tags=["Analysis"])
async def get_statistics() -> Dict[str, Any]:
    """
    Get statistical analysis of all predictions made.
    """
    if not prediction_history:
        return {"message": "No predictions available for analysis"}
    
    all_prices = [h["prediction"]["predicted_price"] for h in prediction_history]
    
    return {
        "total_predictions": len(all_prices),
        "price_statistics": {
            "mean": round(statistics.mean(all_prices), 2),
            "median": round(statistics.median(all_prices), 2),
            "std_dev": round(statistics.stdev(all_prices) if len(all_prices) > 1 else 0, 2),
            "min": round(min(all_prices), 2),
            "max": round(max(all_prices), 2)
        },
        "price_range_distribution": {
            range_type.value: sum(
                1 for h in prediction_history 
                if h["prediction"]["price_range"] == range_type.value
            )
            for range_type in PriceRange
        },
        "time_range": {
            "first_prediction": prediction_history[0]["timestamp"] if prediction_history else None,
            "last_prediction": prediction_history[-1]["timestamp"] if prediction_history else None
        }
    }

# Neighborhood analysis endpoint
@app.post("/api/v1/analysis/neighborhood", tags=["Analysis"])
async def analyze_neighborhood(
    latitude: float = Query(..., ge=32, le=42, description="Latitude coordinate"),
    longitude: float = Query(..., ge=-125, le=-114, description="Longitude coordinate"),
    radius: float = Query(0.5, ge=0.1, le=5.0, description="Search radius in degrees")
) -> Dict[str, Any]:
    """
    Analyze property prices in a neighborhood based on location.
    """
    # Find predictions within radius
    nearby_predictions = []
    for hist in prediction_history:
        feat = hist["features"]
        distance = np.sqrt(
            (feat["Latitude"] - latitude)**2 + 
            (feat["Longitude"] - longitude)**2
        )
        if distance <= radius:
            nearby_predictions.append(hist["prediction"]["predicted_price"])
    
    if not nearby_predictions:
        return {
            "message": "No predictions found in this neighborhood",
            "search_center": {"latitude": latitude, "longitude": longitude},
            "search_radius": radius
        }
    
    return {
        "neighborhood_analysis": {
            "center": {"latitude": latitude, "longitude": longitude},
            "radius": radius,
            "properties_found": len(nearby_predictions),
            "average_price": round(statistics.mean(nearby_predictions), 2),
            "price_range": {
                "min": round(min(nearby_predictions), 2),
                "max": round(max(nearby_predictions), 2)
            }
        }
    }

# Feature impact analysis
@app.post("/api/v1/analysis/feature-impact", tags=["Analysis"])
async def analyze_feature_impact(
    base_features: HousingFeatures,
    feature_to_vary: str = Query(..., description="Feature name to analyze"),
    variation_steps: int = Query(5, ge=3, le=10, description="Number of variation steps")
) -> Dict[str, Any]:
    """
    Analyze how changing a single feature impacts the predicted price.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not hasattr(base_features, feature_to_vary):
        raise HTTPException(status_code=400, detail=f"Invalid feature: {feature_to_vary}")
    
    # Get base prediction
    base_dict = base_features.dict()
    base_array = np.array([[
        base_dict["MedInc"], base_dict["HouseAge"], base_dict["AveRooms"],
        base_dict["AveBedrms"], base_dict["Population"], base_dict["AveOccup"],
        base_dict["Latitude"], base_dict["Longitude"]
    ]])
    base_prediction = model.predict(base_array)[0]
    
    # Vary the feature
    feature_value = base_dict[feature_to_vary]
    variations = []
    
    # Create variation range
    if feature_to_vary in ["Latitude", "Longitude"]:
        min_val = feature_value - 2
        max_val = feature_value + 2
    else:
        min_val = max(0, feature_value * 0.5)
        max_val = feature_value * 1.5
    
    step_size = (max_val - min_val) / (variation_steps - 1)
    
    for i in range(variation_steps):
        varied_value = min_val + (i * step_size)
        varied_dict = base_dict.copy()
        varied_dict[feature_to_vary] = varied_value
        
        varied_array = np.array([[
            varied_dict["MedInc"], varied_dict["HouseAge"], varied_dict["AveRooms"],
            varied_dict["AveBedrms"], varied_dict["Population"], varied_dict["AveOccup"],
            varied_dict["Latitude"], varied_dict["Longitude"]
        ]])
        
        prediction = model.predict(varied_array)[0]
        variations.append({
            "feature_value": round(varied_value, 2),
            "predicted_price": round(float(prediction), 2),
            "price_change": round(float(prediction - base_prediction), 2),
            "percent_change": round(((prediction - base_prediction) / base_prediction) * 100, 2)
        })
    
    return {
        "feature_analyzed": feature_to_vary,
        "base_prediction": round(float(base_prediction), 2),
        "variations": variations,
        "summary": {
            "most_impact_value": max(variations, key=lambda x: abs(x["price_change"]))["feature_value"],
            "max_price_increase": max(v["price_change"] for v in variations),
            "max_price_decrease": min(v["price_change"] for v in variations)
        }
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "status_code": 400,
            "timestamp": datetime.now().isoformat()
        }
    )

# Additional utility endpoints
@app.delete("/api/v1/predictions/history", tags=["Management"])
async def clear_history(
    confirm: bool = Query(..., description="Confirm history deletion")
) -> Dict[str, str]:
    """
    Clear prediction history (requires confirmation).
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")
    
    prediction_history.clear()
    return {"message": "Prediction history cleared successfully"}

@app.get("/api/v1/metrics", tags=["Monitoring"])
async def get_metrics() -> Dict[str, Any]:
    """
    Get detailed API metrics and performance statistics.
    """
    uptime = (datetime.now() - api_metrics["start_time"]).total_seconds()
    
    return {
        "uptime": {
            "seconds": round(uptime, 2),
            "human_readable": str(timedelta(seconds=int(uptime)))
        },
        "predictions": {
            "total": api_metrics["total_predictions"],
            "errors": api_metrics["total_errors"],
            "success_rate": round(
                (api_metrics["total_predictions"] - api_metrics["total_errors"]) / 
                max(api_metrics["total_predictions"], 1) * 100, 2
            )
        },
        "performance": {
            "average_predictions_per_hour": round(
                api_metrics["total_predictions"] / (uptime / 3600) if uptime > 0 else 0, 2
            ),
            "last_prediction": api_metrics["last_prediction"].isoformat() if api_metrics["last_prediction"] else None
        }
    }