from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    minutes = float(request.form['minutes'])  
    prediction = model.predict([[minutes]])   
    calories = round(prediction[0], 2)
    return render_template('result.html', minutes=minutes, calories=calories)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
