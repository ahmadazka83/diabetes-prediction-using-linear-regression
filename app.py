from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model_diabetes.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    bmi = float(request.form['bmi'])
    bp = float(request.form['bp'])
    s1 = float(request.form['s1'])
    s2 = float(request.form['s2'])
    s3 = float(request.form['s3'])
    s4 = float(request.form['s4'])
    s5 = float(request.form['s5'])
    s6 = float(request.form['s6'])

    # Create DataFrame with the input values
    new_data = pd.DataFrame([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]],
                            columns=["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"])

    # Standardize the data using the loaded scaler
    new_data_scaled = scaler.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_scaled)[0]

    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
