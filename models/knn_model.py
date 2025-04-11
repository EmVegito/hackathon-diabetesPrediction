import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from util import save_model
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


file_path = "./data/processed_data.csv"

df = pd.read_csv(file_path)

y = df['Fraud_Label']
X = df.drop(columns=['Fraud_Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

knn_model = KNeighborsClassifier()

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
y_prob = knn_model.predict_proba(X_test)[:, 1]

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)


print(f"\nK Nieghrest Neighbour - Classification Report:")
print(classification_report(y_test, y_pred))

print(f"K Nieghrest Neighbour - ROC AUC: {roc_auc:.4f}")
print(f"K Nieghrest Neighbour - PR AUC: {pr_auc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - K Nieghrest Neighbour Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

param_grid = {
    'weights': ['distance'],
    'metric': ['manhattan'],
}

print("\n=== Stratified K-Fold Cross-Validation ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
# Create grid search
grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    cv=cv,
    scoring='average_precision',  # Better for imbalanced datasets
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
y_prob = knn_model.predict_proba(X_test)[:, 1]

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
plt.show()

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
plt.show()

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
print(best_model)

#save_model(best_model, filename='fraud_detection_lrModel.pkl')