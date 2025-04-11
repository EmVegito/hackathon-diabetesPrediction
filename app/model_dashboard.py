import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Set a title for the dashboard
st.title("Classification Model Comparison Dashboard")
st.markdown(
    "An end-to-end machine learning project built by our team to evaluate and compare several classification models "
    "for diabetes detection. This Streamlit dashboard performs hyperparameter tuning, displays detailed evaluation metrics,"
    " and visualizes model performance (including ROC, Precision-Recall curves, and confusion matrices)."
    
)

# --- 1. Generate or Load Data ---
st.sidebar.header("Data Configuration")
data_option = st.sidebar.radio("Choose Data Source:", ("Generate Synthetic Data", "Upload CSV File"))

if data_option == "Generate Synthetic Data":
    
    df = pd.read_csv('././data/diabetes_preprocessed.csv')

    y = df["Outcome"]
    X = df.drop(columns=["Outcome"])
    class_names = sorted(y.unique().astype(str).tolist())
    best_of_five = st.sidebar.selectbox("Select the best of five models:", ["No", "Yes"])
    if best_of_five == "No":
        best_model = st.sidebar.selectbox("Select the model:", ["Best Model", "Default Model"])
    else:
        st.sidebar.write("Best of five models will be used.")
    feature_names = X.columns.tolist()

elif data_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file for classification:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.markdown("Ensure your CSV has a target variable column.")
        target_column = st.sidebar.selectbox("Select the target variable column:", df.columns)
        best_of_five = st.sidebar.selectbox("Select the best of five models:", ["No", "Yes"])
        if best_of_five == "No":
            best_model = st.sidebar.selectbox("Select the model:", ["Best Model", "Default Model"])
        else:
            st.sidebar.write("Best of five models will be used.")
        feature_columns = [col for col in df.columns if col != target_column]
        if not feature_columns:
            st.error("No feature columns found after excluding the target column.")
        else:
            X = df[feature_columns].values
            y = df[target_column].values
            class_names = sorted(df[target_column].unique().astype(str).tolist())
            feature_names = feature_columns
    else:
        st.info("Please upload a CSV file to proceed.")
        st.stop()

# --- 2. Select Models ---
st.sidebar.header("Model Selection")

if best_of_five == "No":
    selected_models = st.sidebar.multiselect(
        "Choose classification models to compare:",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Support Vector Machine (SVM)",
        ],
        default=[
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
        ],
    )
else:
    selected_models = []  # We won't use this if best_of_five is Yes

# ---Logistic Regression---
lrModel = joblib.load("././data/default_models/lrModel.pkl")
best_lrModel = joblib.load("././data/best_models/best_lrModel.pkl")

#---Decision Tree---
dtModel = joblib.load("././data/default_models/dtModel.pkl")
best_dtModel = joblib.load("././data/best_models/best_dtModel.pkl")

#---Random Forest---
rf_Model = joblib.load("././data/default_models/rfModel.pkl")
best_rfModel = joblib.load("././data/best_models/best_rfModel.pkl")

#---Support Vector Machines Classifier---
svcModel = joblib.load("././data/default_models/svcModel.pkl")
best_svcModel = joblib.load("././data/best_models/best_svcModel.pkl")

#---Gradient Boosting Classifier---
gbModel = joblib.load("././data/default_models/gbModel.pkl")
best_gbModel = joblib.load("././data/best_models/best_gbModel.pkl")

#---best_model_overall---
best_model_overall = joblib.load("././data/best_models/best_model.pkl")

models = {}
if best_of_five == "Yes":
    models["Best Of All"] = best_model_overall
else:
    if "Logistic Regression" in selected_models:
        models["Logistic Regression"] = best_lrModel if best_model == "Best Model" else lrModel
    if "Decision Tree" in selected_models:
        models["Decision Tree"] = best_dtModel if best_model == "Best Model" else dtModel
    if "Random Forest" in selected_models:
        models["Random Forest"] = best_rfModel if best_model == "Best Model" else rf_Model
    if "Support Vector Machine (SVM)" in selected_models:
        models["Support Vector Machine (SVM)"] = best_svcModel if best_model == "Best Model" else svcModel
    if "Gradient Boosting" in selected_models:
        models["Gradient Boosting"] = best_gbModel if best_model == "Best Model" else gbModel

if not models:
    st.warning("Please select at least one model to compare.")
    st.stop()

st.header("Dataset Overview")
st.subheader("Data Preview")
st.write("Here are the first 5 rows of the dataset:")
st.dataframe(df.head())

# --- 3. Train and Evaluate Models ---
st.header("Model Performance")

if 'df' in locals():  # Proceed only if data is loaded or generated
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            if y_prob is not None:
                if len(np.unique(y_test)) > 2:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                else:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1]) #for binary classification
            else:
                roc_auc = None
            model_results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "model": model,
            }
        except Exception as e:
            st.error(f"Error training or evaluating {name}: {e}")
            del models[name] # Remove the model if there's an error

    if model_results:
        # --- 4. Display Performance Metrics ---
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame(model_results).T
        metrics_df['y_prob'] = metrics_df['y_prob'].apply(lambda x: x[1] if isinstance(x, (list, tuple, np.ndarray)) else x)
        st.dataframe(metrics_df)

        # --- 5. Visualize Results ---
        st.subheader("Visualizations")

        # Bar chart of key metrics
        metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
        if all(metrics_df['roc_auc'].notna()):
            metrics_to_plot.append("roc_auc")

        metrics_data = metrics_df[metrics_to_plot].reset_index().rename(columns={'index': 'Model'})
        metrics_melted = pd.melt(metrics_data, id_vars=['Model'], var_name='Metric', value_name='Score')

        fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted, ax=ax_metrics)
        ax_metrics.set_title("Comparison of Performance Metrics")
        ax_metrics.set_ylabel("Score")
        ax_metrics.set_ylim(0, 1.05)
        st.pyplot(fig_metrics)

        # Confusion Matrices
        st.subheader("Confusion Matrices")
        num_cols = min(len(models), 3)
        num_rows = (len(models) + num_cols - 1) // num_cols
        if best_of_five == "Yes":
            fig, ax = plt.subplots()
            # And change your line to:
            # xticklabels=class_names, yticklabels=class_names, ax=ax
            cm = confusion_matrix(y_test, model_results["Best Of All"]["y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title("Confusion Matrix - Best Of All")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
        else:
            fig_cm, axes_cm = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            axes_cm = axes_cm.flatten()  # Flatten for easy indexing


            for i, (name, results) in enumerate(model_results.items()):
                cm = confusion_matrix(y_test, results["y_pred"])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=class_names, yticklabels=class_names, ax=axes_cm[i])
                axes_cm[i].set_title(f"Confusion Matrix - {name}")
                axes_cm[i].set_xlabel("Predicted Label")
                axes_cm[i].set_ylabel("True Label")

            for j in range(i + 1, num_rows * num_cols):
                fig_cm.delaxes(axes_cm[j]) # Remove empty subplots

            st.pyplot(fig_cm)

        # Classification Reports
        st.subheader("Classification Reports")
        for name, results in model_results.items():
            report = classification_report(y_test, results["y_pred"], target_names=class_names, zero_division=0)
            st.markdown(f"**{name} Classification Report:**\n```\n{report}\n```")

        # ROC Curves (if binary or multi-class with predict_proba)
        if all(results.get("y_prob") is not None for results in model_results.values()) and len(np.unique(y)) > 1:
            st.subheader("ROC Curves")
            fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
            for name, results in model_results.items():
                if len(np.unique(y)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, results["y_prob"][:, 1])
                    roc_auc = roc_auc_score(y_test, results["y_prob"][:, 1])
                    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
                # elif len(np.unique(y)) > 2:
                #     from sklearn.preprocessing import label_binarize
                #     y_test_bin = label_binarize(y_test, classes=np.unique(y))
                #     for i in range(len(class_names)):
                #         fpr, tpr, _ = roc_curve(y_test_bin[:, i], results["y_prob"][:, i])
                #         roc_auc = roc_auc_score(y_test_bin[:, i], results["y_prob"][:, i])
                #         ax_roc.plot(fpr, tpr, label=f"{name} - {class_names[i]} (AUC = {roc_auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], "k--", label="Baseline")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # Precision-Recall Curves
        if all(results.get("y_prob") is not None for results in model_results.values()) and len(np.unique(y)) > 1:
            st.subheader("Precision-Recall Curves")
            fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
            for name, results in model_results.items():
                precision, recall, _ = precision_recall_curve(y_test, results["y_prob"][:, 1])
                pr_auc = auc(recall, precision)
                ax_pr.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.2f})")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall Curves")
            ax_pr.legend()
            st.pyplot(fig_pr)

        # --- 6. Feature Importance (for tree-based models) ---
        st.subheader("Feature Importance")
        for name, results in model_results.items():
            if name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
                try:
                    if hasattr(results["model"], "feature_importances_"):
                        importances = results["model"].feature_importances_
                        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                        fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_fi)
                        ax_fi.set_title(f"Feature Importance - {name}")
                        st.pyplot(fig_fi)
                    else:
                        st.warning(f"{name} does not have feature importance attribute.")
                except Exception as e:
                    st.error(f"Error displaying feature importance for {name}: {e}")

        # --- 7. Model Comparison Highlights ---
        st.subheader("Key Highlights and Comparison")
        best_model = max(model_results, key=lambda k: model_results[k]['accuracy'])
        st.markdown(f"**Best Performing Model (based on accuracy):** {best_model}")
        st.markdown("---")
        st.markdown("### Model-specific Observations:")
        for name, results in model_results.items():
            st.markdown(f"**{name}:**")
            st.markdown(f"- Accuracy: {results['accuracy']:.4f}")
            st.markdown(f"- Precision: {results['precision']:.4f}")
            st.markdown(f"- Recall: {results['recall']:.4f}")
            st.markdown(f"- F1-Score: {results['f1']:.4f}")
            if results['roc_auc'] is not None:
                st.markdown(f"- ROC AUC: {results['roc_auc']:.4f}")
            st.markdown("---")

        st.info("This dashboard provides a visual and numerical comparison of the selected classification models. Consider the specific needs of your problem when choosing the best model.")