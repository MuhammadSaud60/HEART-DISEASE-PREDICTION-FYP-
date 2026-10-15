from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import xgboost


app = Flask(__name__)

model = joblib.load('model/xgbModel.pkl')
encoders = joblib.load('model/encoders.pkl')
scaler = joblib.load('model/Scaler.pkl')
columns = joblib.load('model/model_columns.joblib')





@app.route('/')
def index():
    return render_template('index.html')


@app.route('/form')
def form():
    return render_template('form.html')



def preprocess(data,  encoders, scaler, columns):
    df = pd.DataFrame(data, index=[0])
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except:
                raise ValueError(f'Unknown Category {df[col].iloc[0]} for feature {col}')
        else:
            raise ValueError(f'Expected Column {col} not found in input data')
    df = df[columns]

    scaled_features = scaler.transform(df)
        
    return scaled_features

@app.route('/submit', methods=['POST'])
def sumbit():
    user_input = request.json

    if not user_input:
        return jsonify({'error': 'No Data Received'}), 400

    print(user_input)
    preprocess_input = preprocess(user_input,encoders, scaler, columns)
    
    prediction = model.predict(preprocess_input)
    

    output_class = int(prediction[0])
    
    if output_class == 0:
        return jsonify({'message': 'No Heart Disease'}), 200
    else:
        return jsonify({'message': 'Heart Disease'}), 200
    


@app.route('/about')
def about():
    return render_template('about.html')
  

@app.route('/dataSource')
def dataSource():
    return render_template('data.html')
  


  
    
if __name__ == '__main__':
    app.run(debug=True)
