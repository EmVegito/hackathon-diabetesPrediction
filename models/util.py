import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def load_dataset(file_path):
    """
    Load the fraud detection dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df


def engineer_features(df):
    """
    Perform feature engineering on the dataset
    """
    print("\n=== FEATURE ENGINEERING ===")
    df_copy = df.copy()
    
    # Handle 'step' column - could represent time
    if 'step' in df_copy.columns:
        # Bin the step into time periods (e.g., hours of day assuming 24 hours)
        df_copy['time_period'] = df_copy['step'] % 24
        
        # Create day feature (assuming steps are hours)
        df_copy['day'] = df_copy['step'] // 24
        
        print("Added time period and day features based on step column")
    
    # Handle amount-related features
    if 'amount' in df_copy.columns:
        # Log transform for skewed amount distribution
        df_copy['log_amount'] = np.log1p(df_copy['amount'])
        
        # Amount binning
        df_copy['amount_bin'] = pd.qcut(df_copy['amount'], q=10, labels=False, duplicates='drop')
        
        print("Added log transformed amount and amount bins")
    
    # Handle oldbalanceOrg and newbalanceOrig if they exist
    if 'oldbalanceOrg' in df_copy.columns and 'newbalanceOrig' in df_copy.columns:
        # Calculate the difference in balance
        df_copy['orig_balance_diff'] = df_copy['newbalanceOrig'] - df_copy['oldbalanceOrg']
        
        # Check if balance becomes zero after transaction
        df_copy['orig_zero_after_transaction'] = (df_copy['newbalanceOrig'] == 0).astype(int)
        
        # Calculate transaction to balance ratio
        df_copy['orig_transaction_to_balance_ratio'] = df_copy['amount'] / (df_copy['oldbalanceOrg'] + 1)
        
        print("Added origin account balance difference features")
    
    # Handle oldbalanceDest and newbalanceDest if they exist
    if 'oldbalanceDest' in df_copy.columns and 'newbalanceDest' in df_copy.columns:
        # Calculate the difference in balance
        df_copy['dest_balance_diff'] = df_copy['newbalanceDest'] - df_copy['oldbalanceDest']
        
        # Check if balance becomes zero after transaction
        df_copy['dest_zero_after_transaction'] = (df_copy['newbalanceDest'] == 0).astype(int)
        
        # Calculate transaction to balance ratio
        df_copy['dest_transaction_to_balance_ratio'] = df_copy['amount'] / (df_copy['oldbalanceDest'] + 1)
        
        print("Added destination account balance difference features")
    
    # Check if the amount matches the balance difference
    if all(col in df_copy.columns for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig']):
        df_copy['orig_amount_matches_balance_diff'] = (abs(df_copy['oldbalanceOrg'] - df_copy['newbalanceOrig'] - df_copy['amount']) < 0.01).astype(int)
        print("Added feature to detect if origin amount matches balance difference")
    
    if all(col in df_copy.columns for col in ['amount', 'oldbalanceDest', 'newbalanceDest']):
        df_copy['dest_amount_matches_balance_diff'] = (abs(df_copy['newbalanceDest'] - df_copy['oldbalanceDest'] - df_copy['amount']) < 0.01).astype(int)
        print("Added feature to detect if destination amount matches balance difference")
    
    # replace names with there frequency
    frequency= df_copy['nameOrig'].value_counts()
    df_copy['nameOrig_freq'] = df_copy['nameOrig'].map(frequency)

    frequency_dest = df_copy['nameDest'].value_counts()
    df_copy['nameDest_freq'] = df_copy['nameDest'].map(frequency_dest)

    df_copy.drop(columns=['nameOrig', 'nameDest'], inplace=True)

    df_copy.drop(columns=['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], inplace=True)

    print(f"Feature engineering complete. New shape: {df_copy.shape}")

    return df_copy

def create_preprocessing_pipeline(df, target_column='isFraud'):
    """
    Create a preprocessing pipeline for the fraud detection dataset
    """
    print("\n=== PREPROCESSING PIPELINE ===")
    
    # Separate features and target
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print(f"Target variable '{target_column}' identified and separated")
    else:
        X = df
        y = None
        print(f"Target variable '{target_column}' not found in dataset. Proceeding with feature preprocessing only.")
    
    # Identify column types
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {categorical_columns}")
    print(f"Numeric columns: {len(numeric_columns)} columns identified")
    
    # Create preprocessing steps for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    
    # Create the preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    print("Preprocessing pipeline created")
    
    # Split the data
    if y is not None:
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Data split into train and test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return preprocessing_pipeline, X_train, X_test, y_train, y_test
    else:
        return preprocessing_pipeline, X, None, None, None

# 5. Pipeline execution function
def execute_pipeline(file_path):
    """
    Execute the full pipeline from loading to preprocessing
    """
    # Load data
    df = load_dataset(file_path)
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Create preprocessing pipeline
    pipeline, X_train, X_test, y_train, y_test = create_preprocessing_pipeline(df_engineered)
    
    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    print("The data is now ready for model training.")
    
    if y_train is not None:
        # Apply preprocessing
        print("\nApplying preprocessing to training data...")
        X_train_processed = pipeline.fit_transform(X_train)
        print(f"Processed training data shape: {X_train_processed.shape}")
        
        print("\nApplying preprocessing to test data...")
        X_test_processed = pipeline.transform(X_test)
        print(f"Processed test data shape: {X_test_processed.shape}")
        
        return pipeline, X_train_processed, X_test_processed, y_train, y_test
    else:
        X_processed = pipeline.fit_transform(X_train)
        print(f"Processed data shape: {X_processed.shape}")
        return pipeline, X_processed, None, None, None