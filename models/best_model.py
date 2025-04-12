import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import warnings
from util import save_model
warnings.filterwarnings('ignore')

df = pd.read_csv("./data/diabetes_preprocessed.csv")
y = df['Outcome']
X = df.drop(columns=['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42,)

best_svcModel = joblib.load("./data/best_models/best_svcModel.pkl")
best_gbModel = joblib.load("./data/best_models/best_gbModel.pkl")
best_rfModel = joblib.load("./data/best_models/best_rfModel.pkl")
best_lrModel = joblib.load("./data/best_models/best_lrModel.pkl")
best_dtModel = joblib.load("./data/best_models/best_dtModel.pkl")


voting_clf = VotingClassifier(
    estimators=[
        ('lr', best_lrModel),
        ('rf', best_rfModel),
        ('gb', best_gbModel),
        ('dt', best_dtModel)
    ],
    voting='soft'
)

models = {
    'RandomForest': best_rfModel,
    'GradientBoosting': best_gbModel,
    'SVC': best_svcModel,
    'Logistic Regression': best_lrModel,
    'Decision Tree': best_dtModel,
    'Voting Classifier': voting_clf,
}


best_model = None
best_score = -1
best_name = None


for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"{name} ROC AUC Score: {roc_auc:.4f}")
    
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model
        best_name = name

print(f"\nBest Model: {best_name} with ROC AUC: {best_score:.4f}")


#save_model(best_model, filename="./data/best_models/best_model.pkl")