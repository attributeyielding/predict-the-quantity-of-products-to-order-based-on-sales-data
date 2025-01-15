import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load sales data
def load_data(file_path):
    """
    Load sales data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data Loaded Successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess data
def preprocess_data(data):
    """
    Preprocess data to create features and target variables.
    """
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure date column is datetime
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Weekday'] = data['Date'].dt.weekday  # Day of the week
    
    # Aggregate sales data (group by product and time features if needed)
    aggregated = data.groupby(['Product', 'Year', 'Month']).agg({
        'Sales': 'sum'
    }).reset_index()
    
    return aggregated

# Train a model
def train_model(data):
    """
    Train a regression model to predict product order quantities.
    """
    # Features and target variable
    X = data[['Year', 'Month']]
    y = data['Sales']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Trained. Mean Absolute Error: {mae}")
    
    return model

# Predict future order quantities
def predict_quantities(model, year, month):
    """
    Predict future quantities to order for given year and month.
    """
    prediction_data = pd.DataFrame({
        'Year': [year],
        'Month': [month]
    })
    predictions = model.predict(prediction_data)
    return predictions

# Main function
if __name__ == "__main__":
    # File path for sales data
    file_path = 'sales_data.csv'  # Replace with your CSV file path
    
    # Load data
    data = load_data(file_path)
    
    if data is not None:
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Train model
        model = train_model(processed_data)
        
        # Predict future quantities
        year = 2025
        month = 2
        prediction = predict_quantities(model, year, month)
        print(f"Predicted quantity for {year}-{month}: {prediction[0]:.2f}")
