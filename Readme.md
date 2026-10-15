# HeartCare: Web-based Heart Disease Prediction Using Machine Learning

A Flask web application that predicts heart disease risk from routine clinical measurements, built on a carefully validated machine learning pipeline.

## Results

Trained and evaluated on the UCI Heart Disease dataset (920 records, combined Cleveland/Hungarian/Switzerland/Long Beach). Target binarised (disease vs. no disease), evaluated on a held-out test set (n=276) with preprocessing strictly separated from the test data (no leakage):

| Model | Accuracy | ROC AUC |
|---|---|---|
| Random Forest | 0.826 | 0.909 |
| XGBoost | **0.837** | 0.908 |
| SVC | 0.822 | 0.887 |

The three models perform comparably, with XGBoost marginally best. See `UCI Dataset.ipynb` for the full corrected pipeline and `CORRECTED_RESULTS.md` for details.

**Methodology note:** an earlier version of this project contained data leakage (preprocessing fitted before the train/test split) and models compared on different test sets. The current results come from a fully corrected, leak-free pipeline with per-column random-forest imputers and honest cross-validation.

## Run it

```bash
pip install flask pandas numpy scikit-learn xgboost joblib
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Project structure

- `app.py` — Flask application (loads the XGBoost model in native JSON format)
- `model/` — trained artefacts: `xgbModel_corrected.json`, encoders, scaler, imputers
- `UCI Dataset.ipynb` — corrected training/evaluation notebook
- `UCI Dataset.original.ipynb` — original notebook, kept for reference
- `templates/`, `static/` — web interface
- `thesis.pdf` — full BSc thesis (55 pages)

## Authors

Muhammad Saud, Saad Hussain, Muhammad Anas — Government Postgraduate College Mardan, Pakistan
(Supervisor: Muhammad Assad)

Research paper in preparation: *"HeartCare: Web-based Heart Disease Prediction Using Machine Learning."*

## Disclaimer

This is an academic machine learning project. It is not a medical device and must not be used for clinical diagnosis.
