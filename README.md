# predict-the-quantity-of-products-to-order-based-on-sales-data

To predict the quantity of products to order based on sales data, you can use a Machine Learning (ML) model. The following Python script demonstrates a simple approach using the Linear Regression algorithm from the ```scikit-learn``` library.

# Steps
1. Load the Dataset: Sales data is loaded from a database or file (CSV for simplicity).
2. Feature Engineering: Extract features like historical sales, seasonal patterns, or lead time.
3. Train the Model: Use historical sales data to train a regression model.
4. Make Predictions: Predict future quantities to order.

# Python Script
```
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
```
## Dataset Example (```sales_data.csv```)

----Date-----Product-----Sales

2023-01-01--Product A-----120

2023-02-01--Product A-----130

2023-03-01--Product A-----110

2023-01-01--Product B-----200

2023-02-01--Product B-----210



# How It Works

1. ## Load Data:
   
  - Sales data is loaded from a CSV file.
  - The Date column is used for feature extraction.
    
2. ## Preprocess Data:

  - Extract features like Year, Month, and Weekday from the Date.
  - Group sales by product and time features.
    
3. ## Train Model:

  - A linear regression model is trained using historical sales as the target variable.
    
4. ## Predict:
   
  - Given a future Year and Month, predict the quantity to order.

# Enhancements

1. ### Add more features:
   
  - ```Seasonality```: Capture sales patterns during holidays.
    
  - ```Inventory Levels```: Include stock levels to refine predictions.
    
  - ```Lead Time```: Account for the delay between ordering and delivery.
    
2. ### Use advanced models:
   
  - ```Random Forest``` or ```Gradient Boosting``` for better accuracy.
    
  - Time-series models like ```ARIMA``` or ```LSTM``` for sequential data.
    
3. ### Integrate with a ```database``` for real-time predictions.

   
  
# Let me know if you'd like to explore advanced techniques or need further help! 😊

