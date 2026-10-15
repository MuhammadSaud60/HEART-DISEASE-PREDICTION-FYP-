from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier


app = Flask(__name__)


model = XGBClassifier()
model.load_model('model/xgbModel_corrected.json')
encoders = joblib.load('model/encoders_corrected.pkl')
scaler = joblib.load('model/Scaler_corrected.pkl')
columns = joblib.load('model/model_columns_corrected.joblib')  # feature order
_imp = joblib.load('model/imputers_corrected.pkl')
num_med, cat_mod = _imp['num_med'], _imp['cat_mod']
numeric_cols, categorical_cols = _imp['numeric_cols'], _imp['categorical_cols']
modes = joblib.load('model/modes_corrected.pkl')  # train modes, original labels

# form field name -> training column name
FIELD_MAP = {
    'age': 'age', 'sex': 'sex',
    'chest_pain_type': 'cp', 'resting_blood_pressure': 'trestbps',
    'cholesterol': 'chol', 'fasting_blood_sugar': 'fbs', 'Restecg': 'restecg',
    'max_heart_rate_achieved': 'thalch', 'exercise_induced_angina': 'exang',
    'st_depression': 'oldpeak', 'st_slope_type': 'slope',
    'num_major_vessels': 'ca', 'thalassemia_type': 'thal',
}

VALUE_MAP = {
    'sex': {'1': 'Male', '0': 'Female'},
    'cp': {'typical_angina': 'typical angina',
           'atypical_angina': 'atypical angina',
           'non-anginal': 'non-angina',
           'asymptomatic': 'asymptomatic'},
    'fbs': {'1': 1.0, '0': 0.0},
    'restecg': {'ST-T_wave_abnormality': 'st-t abnormality',
                'left_ventricular_hypertrophy': 'lv hypertrophy',
                'normal': 'normal'},
    'exang': {'1': 1.0, '0': 0.0},
    'thal': {'fixed_defect': 'fixed defect',
             'reversable_defect': 'reversable defect',
             'normal': 'normal'},
}
# form fields that must be numeric-or-empty (validated before imputation)
NUMERIC_FIELDS = ['age', 'sex', 'resting_blood_pressure', 'cholesterol',
                  'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']


def preprocess(data):
    """Map the form submission into the corrected pipeline's feature space.

    Steps mirror the training notebook: translate the form vocabulary to the
    training labels, median/mode-impute anything missing (imputers fit on the
    training set), label-encode (unseen labels fall back to the training
    mode), and MinMax-scale in the exact training column order.
    """
    row = {}
    for form_field, col in FIELD_MAP.items():
        v = data.get(form_field, '')
        v = '' if v is None else str(v).strip()
        if col in VALUE_MAP:
            v = VALUE_MAP[col].get(v, v)
        row[col] = v
    df = pd.DataFrame([row])

    for col in numeric_cols:  # '' -> NaN -> training median
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_cols] = num_med.transform(df[numeric_cols])

    for col in categorical_cols:  # '' -> NaN -> training mode
        df[col] = df[col].replace('', np.nan)
    df[categorical_cols] = cat_mod.transform(df[categorical_cols])

    for col in categorical_cols:
        le = encoders[col]
        sval = str(df[col].iloc[0])
        if sval not in le.classes_:
            sval = str(modes[col])
        df[col] = le.transform([sval])

    return scaler.transform(df[columns])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.json

    if not user_input:
        return jsonify({'error': 'No Data Received'}), 400

    print(user_input)
    try:
        for field in NUMERIC_FIELDS:  # empty -> imputed; garbage -> 400
            v = user_input.get(field, '')
            if str(v).strip() == '':
                continue
            float(v)
        preprocess_input = preprocess(user_input)
        prediction = model.predict(preprocess_input)
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

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
    app.run(debug=False)
