import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and evaluate their performance
    """
    print("\n=== MODEL TRAINING AND EVALUATION ===")
    
    # Check class imbalance
    print("\nClass distribution in training set:")
    class_counts = np.bincount(y_train)
    print(f"Legitimate transactions: {class_counts[0]} ({class_counts[0]/len(y_train)*100:.2f}%)")
    print(f"Fraudulent transactions: {class_counts[1]} ({class_counts[1]/len(y_train)*100:.2f}%)")
    
    # If severe imbalance, apply SMOTE
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
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        # Print classification report
        print(f"\n{name} - Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print AUC scores
        print(f"{name} - ROC AUC: {roc_auc:.4f}")
        print(f"{name} - PR AUC: {pr_auc:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    # Find best model based on PR AUC (better for imbalanced data)
    best_model_name = max(results, key=lambda k: results[k]['pr_auc'])
    print(f"\nBest model based on PR AUC: {best_model_name} (PR AUC: {results[best_model_name]['pr_auc']:.4f})")
    
    # Fine-tune best model
    best_model = results[best_model_name]['model']
    fine_tuned_model = fine_tune_model(best_model_name, best_model, X_train_resampled, y_train_resampled)
    
    return results, fine_tuned_model

def fine_tune_model(model_name, model, X_train, y_train):
    """
    Fine-tune the best performing model using GridSearchCV
    """
    print(f"\n=== FINE-TUNING {model_name} ===")
    
    # Define parameter grid based on model type
    param_grid = {}
    
    if model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'saga']
        }
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced']
        }
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'scale_pos_weight': [1, 5, 10]
        }
    
    # Use stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=model,
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
    
    return grid_search.best_estimator_

def evaluate_threshold(model, X_test, y_test):
    """
    Find the optimal threshold for classification
    """
    print("\n=== THRESHOLD OPTIMIZATION ===")
    
    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
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
    
    return best_threshold

def feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models
    """
    print("\n=== FEATURE IMPORTANCE ===")
    
    # Check if the model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, min(20, len(indices))])  # Show top 20 features at most
        plt.tight_layout()
        plt.show()
        
        # Print top 10 features
        print("\nTop 10 Important Features:")
        for i in range(min(10, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    else:
        print("This model doesn't support feature importance visualization.")

def save_model(model, pipeline, filename='fraud_detection_model.pkl'):
    """
    Save the model and preprocessing pipeline for future use
    """
    import joblib
    
    # Create a dictionary with pipeline and model
    model_package = {
        'preprocessing_pipeline': pipeline,
        'model': model
    }
    
    # Save to file using joblib
    joblib.dump(model_package, filename, compress=3)
    
    print(f"\nModel and preprocessing pipeline saved to {filename} using joblib")

def load_model(filename='fraud_detection_model.pkl'):
    """
    Load the saved model and preprocessing pipeline
    """
    import pickle
    
    # Load from file
    with open(filename, 'rb') as file:
        model_package = pickle.load(file)
    
    # Extract pipeline and model
    pipeline = model_package['preprocessing_pipeline']
    model = model_package['model']
    
    print(f"Model and preprocessing pipeline loaded from {filename}")
    
    return pipeline, model

def make_predictions(pipeline, model, data, threshold=0.5):
    """
    Make predictions using the saved model and pipeline
    """
    # Preprocess the data
    processed_data = pipeline.transform(data)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities

# Main function to execute the pipeline
def execute_full_pipeline(file_path):
    """
    Execute the complete pipeline from preprocessing to model training
    """
    from util import execute_pipeline

    # Execute preprocessing pipeline
    pipeline, X_train_processed, X_test_processed, y_train, y_test = execute_pipeline(file_path)
    
    # Train and evaluate models
    results, best_model = train_models(X_train_processed, y_train, X_test_processed, y_test)
    
    # Find optimal threshold
    best_threshold = evaluate_threshold(best_model, X_test_processed, y_test)
    
    # Get feature names after preprocessing
    # This is tricky as the preprocessing pipeline transforms column names
    # For demonstration, we'll create generic feature names
    feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    
    # Analyze feature importance
    feature_importance(best_model, feature_names)
    
    # Save the model
    save_model(best_model, pipeline)
    
    print("\n=== PIPELINE COMPLETE ===")
    print("The fraud detection model is now ready for deployment.")
    
    return pipeline, best_model, best_threshold

if __name__ == "__main__":
    # Replace with your local path to the downloaded dataset
    file_path = "../data/data.csv"
    
    # Execute the full pipeline
    pipeline, model, threshold = execute_full_pipeline(file_path)