from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained and saved model
model = pickle.load(open('finalized_model.sav', 'rb'))

# Define a route for the default URL, which loads the form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the action of the form, for when the form is submitted
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    input_data = {
        'type': request.form.get('type', type=int),
        'beds': request.form.get('beds', type=int),
        'bath': request.form.get('bath', type=int),
        'area': request.form.get('area', type=float),
        'place': request.form.get('place', type=int),
        'sublocality': request.form.get('sublocality', type=int),
    }

    # Make prediction using the loaded model
    prediction = model.predict(pd.DataFrame([input_data]))[0]

    # Return prediction
    return render_template('result.html', prediction=f"${prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
