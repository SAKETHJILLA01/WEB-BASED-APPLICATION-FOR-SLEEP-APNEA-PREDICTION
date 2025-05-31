from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = 'sleep_apnea_model.joblib'
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Make sure to run the training script first.")
    model = None

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        bmi = data['bmi']
        neck_circumference = data['neck_circumference']
        age = data['age']
        gender = data['gender']
        ahi = data['ahi']

        input_features = np.array([[bmi, neck_circumference, age, gender, ahi]])
        prediction_proba = model.predict_proba(input_features)[0][1] # Probability of class 1 (Sleep Apnea)
        prediction_label = 1 if prediction_proba >= 0.5 else 0
        prediction_text = "High Risk" if prediction_label == 1 else "Low Risk"

        return jsonify({'prediction': f'{prediction_text} (Probability: {prediction_proba:.2f})'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)