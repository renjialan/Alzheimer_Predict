import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Preprocessing mappings
FEATURE_MAPPINGS = {
    'Gender': {'Female': 0, 'Male': 1},
    'Physical Activity Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Smoking Status': {'Never': 0, 'Former': 1, 'Current': 2},
    'Alcohol Consumption': {'Never': 0, 'Occasionally': 1, 'Regularly': 2},
    'Diabetes': {'No': 0, 'Yes': 1},
    'Cholesterol Level': {'Normal': 0, 'High': 1},
    'Family History of Alzheimer’s': {'No': 0, 'Yes': 1},
    'Depression Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Sleep Quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'Dietary Habits': {'Unhealthy': 0, 'Average': 1, 'Healthy': 2},
    'Genetic Risk Factor (APOE-ε4 allele)': {'No': 0, 'Yes': 1},
    'Stress Levels': {'Low': 0, 'Medium': 1, 'High': 2},
    'Urban vs Rural Living': {'Rural': 0, 'Urban': 1}
}

def preprocess_data(df):
    """Preprocess Alzheimer's dataset"""
    
    for col, mapping in FEATURE_MAPPINGS.items():
        df[col] = df[col].map(mapping)
        
    # Create interaction features
    df['Age_Alcohol'] = df['Age'] * df['Alcohol Consumption']
    df['Physical_Sleep'] = df['Physical Activity Level'] * df['Sleep Quality']
    df['Stress_Depression'] = df['Stress Levels'] * df['Depression Level']
    
    # Age binning
    df['Age Group'] = pd.cut(df['Age'], bins=[49, 59, 69, 79, 89, 99], labels=[0,1,2,3,4])
    df['Age Group'] = df['Age Group'].astype(int)
    
    return df

def train_and_save_model():
    # Load and preprocess data
    df = pd.read_csv('./data/raw/alzheimers_prediction_dataset.csv')
    df = preprocess_data(df)
    
    # Feature selection
    features = [
        'Age', 'Gender', 'Physical Activity Level', 'Sleep Quality',
        'Family History of Alzheimer’s', 'Genetic Risk Factor (APOE-ε4 allele)',
        'Alcohol Consumption', 'Age_Alcohol', 'Physical_Sleep', 'Stress_Depression'
    ]
    
    X = df[features]
    y = df['Alzheimer’s Diagnosis'].map({'No': 0, 'Yes': 1})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    model_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'objective': 'binary:logistic',
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**model_params)
       # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Overfitting check
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Enhanced evaluation report
    report = {
        'cross_val': {
            'scores': cv_scores.tolist(),
            'mean_accuracy': np.mean(cv_scores),
            'std_deviation': np.std(cv_scores)
        },
        'holdout_test': {
            'accuracy': accuracy_score(y_test, test_preds),
            'classification_report': classification_report(y_test, test_preds, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, test_preds).tolist()
        },
        'overfitting_check': {
            'train_accuracy': accuracy_score(y_train, train_preds),
            'test_accuracy': accuracy_score(y_test, test_preds),
            'accuracy_gap': accuracy_score(y_train, train_preds) - accuracy_score(y_test, test_preds)
        }
    }
    
    
    # Save artifacts
    joblib.dump(model, 'models/alzheimer_model.pkl')
    joblib.dump(FEATURE_MAPPINGS, 'models/feature_mappings.pkl')
    joblib.dump(features, 'models/selected_features.pkl')
    
    # Generate evaluation report
    
    joblib.dump(report, 'reports/evaluation_report.pkl')
    
    return model, report

if __name__ == "__main__":
    train_and_save_model()