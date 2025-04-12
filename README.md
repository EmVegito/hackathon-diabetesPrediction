# Diabetes Prediction Dashboard

## Overview
This dashboard provides an interactive interface for predicting diabetes risk using multiple machine learning models. The application applies Logistic Regression, Support Vector Classification (SVC), Random Forest, Gradient Boosting, and Decision Tree algorithms on the Diabetes Prediction dataset from Kaggle to offer comprehensive predictive insights.
(`https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database`)

## Features
- **Multi-model Predictions**: Compare predictions from five different machine learning models
- **Interactive Interface**: Input patient data and receive real-time prediction results
- **Model Performance Metrics**: View and compare accuracy, precision, recall, F1-score and roc-auc score across models
- **Data Visualization**: Explore feature importance, Confustion Matrix and, PR and AUC curves
- **Confidence Scores**: See prediction confidence levels for each model

## Dataset
The dashboard uses the Diabetes Prediction dataset from Kaggle, which includes various health metrics such as:
- Pregnancies
- Glucose
- Blood Pressure
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age Groups
- and additional engineered Features

## Models Implemented
1. **Logistic Regression**: A statistical model for binary classification
2. **Support Vector Classification (SVC)**: Uses kernel functions to map data into higher dimensions
3. **Random Forest**: Ensemble method using multiple decision trees
4. **Gradient Boosting**: Sequential ensemble technique that builds on weak learners
5. **Decision Tree**: Single tree-based model using feature thresholds for classification

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/EmVegito/hackathon-diabetesPrediction.git
   cd hackathon-diabetesPrediction
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Starting the Dashboard
Run the following command in your terminal from the parent directory:
```
streamlit run app/model_dashboard.py
```
Then open your web browser and navigate to `http://localhost:8050` (or the port specified in the terminal output).

### Making Predictions
1. Enter patient data in the input fields
2. Click "Predict" to see results from all models
3. Compare model outputs and confidence scores
4. Explore visualization tabs for deeper insights

## Project Structure
```
diabetes-prediction-dashboard/
├── app/
│   ├──model_dashboard.py           # Main dashboard application
├── data/                           # Trained model files
│   ├── best_models                 # Fine Tuned Model using GridSearch
│       ├──best_dtModel.pkl
│       ├──best_gbModel.pkl
│       ├──best_lrModel.pkl
│       ├──best_model.pkl           # Based On AUC ROC Curve
│       ├──best_rfModel.pkl
│       ├──best_svcModel.pkl
│   ├── default_models              # Model with default hyperparameters
│       ├──dtModel.pkl
│       ├──gbModel.pkl
│       ├──lrModel.pkl
│       ├──rfModel.pkl
│       ├──svcModel.pkl
│   ├── plots                       # For default model after threshold optimization
│   ├── diabetes_preprocessed.csv   # Cleaned and preprocessed data
├── models/                         # model training and saving scripts
│   ├── best_model.py
│   └── dt_model.py
│   └── gb_model.py
│   └── logistic_model.py
│   └── rf_model.py
│   └── svc_model.py
│   └── util.py                     # Other functions like save_model
├── notebooks/
│   ├── diabetes.csv                # Original Dataset
│   ├── preprocessed.ipynb          # Jupyter notebooks for exploration and feature engineering
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Model Performance
Below is a comparison of model performance metrics on the test dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.76 | 0.80 | 0.76 | 0.77 |
| SVC | 0.74 | 0.73 | 0.74 | 0.73 |
| Random Forest | 0.75 | 0.75 | 0.75 | 0.75 |
| Gradient Boosting | 0.74 | 0.73 | 0.74 | 0.73 |
| Decision Tree | 0.71 | 0.71 | 0.71 | 0.77 |

## Customization
You can customize the dashboard by:
- Adding new model training scripts in the `models` directory
- Modifying the UI components in `model_dashboard.py`
- Updating preprocessing steps in `preprocessed.ipynb`

## Troubleshooting
- **Missing dependencies**: Ensure all packages in `requirements.txt` are installed
- **Model loading errors**: Check that all model files exist in the `data/` directory
- **Data format issues**: Verify that input data matches the expected format of the trained models

## Future Improvements
- Implementation of additional machine learning algorithms
- Advanced feature engineering options
- API endpoint for external application integration

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Prima Indians Diabetes Prediction dataset from Kaggle
- Scikit-learn library for machine learning models
- Seaborn and, matplotlib for interactive visualization