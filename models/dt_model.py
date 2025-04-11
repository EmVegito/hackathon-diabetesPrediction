import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from util import execute_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

file_path = "./data/data.csv"


    
# Execute the pipeline
pipeline, X_train, X_test, y_train, y_test = execute_pipeline(file_path)

print(X_train.shape)
print(X_test.shape)

# Class distribution in training set
print("\nClass distribution in training set:")
class_counts = np.bincount(y_train)
print(f"Legitimate transactions: {class_counts[0]} ({class_counts[0]/len(y_train)*100:.2f}%)")
print(f"Fraudulent transactions: {class_counts[1]} ({class_counts[1]/len(y_train)*100:.2f}%)")

# use SMOTE is class is imbalanced which it is
if class_counts[1] / sum(class_counts) < 0.1:
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original training shape: {X_train.shape}")
    print(f"Resampled training shape: {X_train_resampled.shape}")
    
    # Update the class distribution
    resampled_class_counts = np.bincount(y_train_resampled)
    print("\nClass distribution after resampling:")
    print(f"Legitimate transactions: {resampled_class_counts[0]} ({resampled_class_counts[0]/len(y_train_resampled)*100:.2f}%)")
    print(f"Fraudulent transactions: {resampled_class_counts[1]} ({resampled_class_counts[1]/len(y_train_resampled)*100:.2f}%)")
else:
    X_train_resampled, y_train_resampled = X_train, y_train


print("\nTraining Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train_resampled, y_train_resampled)
print('\nDecision Tree Classifier - Training complete!')

#Evaluate on the training set

y_pred = dt_model.predict(X_train_resampled)  
y_prob = dt_model.predict_proba(X_train_resampled)[:, 1]

roc_auc = roc_auc_score(y_train_resampled, y_prob)
precision, recall, _ = precision_recall_curve(y_train_resampled, y_prob)
pr_auc = auc(recall, precision)


print(f"\n Decision Tree Classifier - Classification Report:")
print(classification_report(y_train_resampled, y_pred))

print(f"Decision Tree Classifier - ROC AUC: {roc_auc:.4f}")
print(f"Decision Tree Classifier - PR AUC: {pr_auc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_train_resampled, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - Decision Tree Classifier Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()