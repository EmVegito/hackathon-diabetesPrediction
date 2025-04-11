import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from util import save_model
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

file_path = "./data/diabetes_preprocessed.csv"

df = pd.read_csv(file_path)

y = df['Outcome']
X = df.drop(columns=['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

save_model(gb_model, filename='gbModel.pkl')

y_pred = gb_model.predict(X_test)
y_prob = gb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)


print(f"\nGradient Boosting - Classification Report:")
print(classification_report(y_test, y_pred))

print(f"Gradient Bossting - ROC AUC: {roc_auc:.4f}")
print(f"Gradient Bossting - PR AUC: {pr_auc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - Gradient Bossting')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show(block=True)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0],
    'max_features': ['sqrt', 'log2', None],
}

print("\n=== Stratified K-Fold Cross-Validation ===")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
# Create grid search
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
print("Performing grid search (this may take some time)...")
grid_search.fit(X_train, y_train)

# Print results
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")


print("\n=== THRESHOLD OPTIMIZATION ===")
    
# Get predicted probabilities
y_prob = gb_model.predict_proba(X_test)[:, 1]

# Calculate precision and recall for different thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Calculate F1 score for each threshold
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

# Find threshold with best F1 score
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC = {auc(recall, precision):.4f})')
plt.axvline(x=recall[best_idx], color='r', linestyle='--', 
            label=f'Best threshold: {best_threshold:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show(block=True)

# Apply best threshold and show metrics
y_pred_optimized = (y_prob >= best_threshold).astype(int)

print("\nClassification report with optimized threshold:")
print(classification_report(y_test, y_pred_optimized))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix with Optimized Threshold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show(block=True)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
print(best_model)

save_model(best_model, filename='./app/best_models/best_gbModel.pkl')