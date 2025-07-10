# Housing Price Prediction API

A machine learning API that predicts California housing prices using a Linear Regression model, containerized with Docker and deployable on Kubernetes.

## Features

- FastAPI-based REST API
- Linear Regression model trained on California Housing dataset
- Docker containerization
- Kubernetes deployment ready
- Interactive API documentation (Swagger UI)
- Health checks and monitoring endpoints

## Quick Start

### 1. Train the Model
```bash
pip install scikit-learn pandas joblib
python train_model.py