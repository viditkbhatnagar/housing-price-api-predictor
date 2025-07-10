from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create app directory if it doesn't exist
os.makedirs('app', exist_ok=True)

# Load data
print("Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)
X = data.frame.drop('MedHouseVal', axis=1)
y = data.frame['MedHouseVal']

# Split data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save model
model_path = 'app/model.pkl'
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Save feature names for reference
feature_names = X.columns.tolist()
print(f"\nFeature names: {feature_names}")