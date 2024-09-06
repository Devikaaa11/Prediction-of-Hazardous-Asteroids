from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model, scaler, and label encoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    le = pickle.load(encoder_file)

# Load the feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Define the home route
@app.route('/')
def home():
    return render_template('pha.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Fetching data from the form and converting to correct type
            form_data = {
                'H': float(request.form['H']),
                'diameter': float(request.form['diameter']),
                'albedo': float(request.form['albedo']),
                'diameter_sigma': float(request.form['diameter_sigma']),
                'epoch': float(request.form['epoch']),
                'e': float(request.form['e']),
                'a': float(request.form['a']),
                'q': float(request.form['q']),
                'i': float(request.form['i']),
                'om': float(request.form['om']),
                'w': float(request.form['w']),
                'ma': float(request.form['ma']),
                'ad': float(request.form['ad']),
                'n': float(request.form['n']),
                'tp_cal': float(request.form['tp_cal']),
                'per': float(request.form['per']),
                'moid': float(request.form['moid']),
                'sigma_a': float(request.form['sigma_a']),
                'sigma_w': float(request.form['sigma_w']),
                'sigma_n': float(request.form['sigma_n']),
                'rms': float(request.form['rms']),
                'class': request.form['class'],
                'neo': request.form['neo']
            }

            # Encode categorical features
            class_map = {'APO': 0, 'ATE': 1, 'AMO': 2, 'IEO': 3, 'Other': 4}
            neo_map = {'N': 0, 'Y': 1}

            form_data['class'] = class_map.get(form_data['class'], 4)
            form_data['neo'] = neo_map.get(form_data['neo'], 0)

            # Create a DataFrame with the feature names
            input_df = pd.DataFrame([form_data], columns=feature_names)

            # Ensure the DataFrame has all columns in the correct order and type
            input_df = input_df[feature_names]
            input_df = input_df.apply(pd.to_numeric, errors='coerce')

            # Handle missing values (if any)
            input_df.fillna(0, inplace=True)  # Example: Fill missing values with 0

            # Scale the input data using the loaded scaler
            scaled_input_data = scaler.transform(input_df)

            # Make the prediction using the pre-trained model
            prediction = model.predict(scaled_input_data)

            # Decode the predicted label (0 -> Not Hazardous, 1 -> Hazardous)
            prediction_result = 'Potentially Hazardous' if le.inverse_transform([prediction])[0] == 'Y' else 'Not Hazardous'

            # Render the result in the HTML
            return render_template('pha.html', prediction_result=f'Prediction: {prediction_result}')
        
        except Exception as e:
            return render_template('pha.html', prediction_result=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
