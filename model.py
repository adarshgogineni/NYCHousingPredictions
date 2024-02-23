import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle

# Assuming your dataset is named 'NY-House-Dataset.csv'
filepath = 'NY-House-Dataset.csv'
input_data = {
        'beds': 4,
        'bath': 2,
        'area': 2000,
        'place': 1,  # Note: This should be encoded similarly to how it was encoded during training
        'sublocality': 1,  # Same as above
        'type': 1  # Same as above
    }

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Data cleaning
    df = df.drop(['LONGITUDE', 'LATITUDE', 'FORMATTED_ADDRESS', 'LONG_NAME', 'STREET_NAME', 'ADMINISTRATIVE_AREA_LEVEL_2', 'MAIN_ADDRESS', 'STATE', 'ADDRESS', 'BROKERTITLE'], axis=1)
    df.rename(columns={'PRICE': 'price', 'BEDS': 'beds', 'BATH': 'bath', 'PROPERTYSQFT': 'area', 'LOCALITY': 'place', 'SUBLOCALITY': 'sublocality', 'TYPE': "type"}, inplace=True)
    df = df.drop(df[(df['price'] == 2147483647) | (df['price'] == 195000000)].index)
    
    # Encoding categorical variables
    le = LabelEncoder()
    df['place'] = le.fit_transform(df['place'])
    df['sublocality'] = le.fit_transform(df['sublocality'])
    df['type'] = le.fit_transform(df['type'])

    # Scaling numerical variables
    scaler = StandardScaler()
    df[['beds', 'bath', 'area']] = scaler.fit_transform(df[['beds', 'bath', 'area']])
    
    
    return df

def train_model(df):
    X = df.drop(['price'], axis=1)
    y = df['price']

    scaler = StandardScaler()
    X[['beds', 'bath', 'area']] = scaler.fit_transform(X[['beds', 'bath', 'area']])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Include the eval_metric and early_stopping_rounds in the constructor
    model = xgb.XGBRegressor(objective="reg:squarederror", seed=42, eval_metric='rmse', early_stopping_rounds=10)
    model.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)])
    
    return model


def predict(model, input_data):
    # Convert input data to DataFrame
    # Ensure the columns match the training dataset's order
    feature_order = ['type', 'beds', 'bath', 'area', 'place', 'sublocality']
    input_df = pd.DataFrame([input_data], columns=feature_order)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return prediction[0]



df = load_and_preprocess_data(filepath)
model = train_model(df)
predicted_price = predict(model, input_data)
print(f"Predicted Price: ${predicted_price:.2f}")
# Save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))



""" 
# Main function to run for your web app backend
if __name__ == '__main__':
    filepath = '/path/to/your/dataset/NY-House-Dataset.csv'  # Update this path
    df = load_and_preprocess_data(filepath)
    model = train_model(df)
    
    # Example input, replace with your actual input features
    input_data = {
        'beds': 4,
        'bath': 2,
        'area': 2000,
        'place': 1,  # Note: This should be encoded similarly to how it was encoded during training
        'sublocality': 1,  # Same as above
        'type': 1  # Same as above
    }
    
    predicted_price = predict(model, input_data)
    print(f"Predicted Price: ${predicted_price:.2f}")
 """