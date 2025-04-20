import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

class AlzheimerPreprocessor:
    def __init__(self, n_features=20):
        """Initialize preprocessing pipeline"""
        self.n_features = n_features
        self._build_pipeline()
        
    def _build_pipeline(self):
        """Configure preprocessing steps"""
        # Define column types (modify these according to your dataset)
        numeric_features = ['Age', 'Neuro_Health_Index', 'Risk_Interaction_Factor']
        categorical_features = ['APOE_Genotype', 'Family_History']
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing 
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=self.n_features)
        
        # Store feature names
        self.feature_names_ = []

    def fit(self, df):
        """Fit preprocessing pipeline to data"""
        # Process data
        X_processed = self.preprocessor.fit_transform(df)
        
        # Feature selection
        y = df['Alzheimer’s Diagnosis'].map({'Yes': 1, 'No': 0})
        self.feature_selector.fit(X_processed, y)
        
        # Get feature names
        self._get_feature_names(df)
        return self

    def transform(self, df):
        """Apply preprocessing to new data"""
        if not hasattr(self.preprocessor, 'transform'):
            raise RuntimeError("Preprocessor not fitted - call fit() first")
            
        X_processed = self.preprocessor.transform(df)
        X_selected = self.feature_selector.transform(X_processed)
        return X_selected

    def _get_feature_names(self, df):
        """Extract feature names after preprocessing"""
        # Get numeric names
        numeric_features = self.preprocessor.transformers_[0][2]
        
        # Get categorical names
        ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_features = ohe.get_feature_names_out(
            self.preprocessor.transformers_[1][2]
        )
        
        # Combine all features
        all_features = np.concatenate([numeric_features, categorical_features])
        
        # Select final features
        self.feature_names_ = all_features[self.feature_selector.get_support()].tolist()

    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save complete processor state"""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load preprocessor with validation"""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise ValueError("Loaded object is not an AlzheimerPreprocessor")
        return obj

# Example usage
if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("alzheimer-app/data/raw/alzheimers_prediction_dataset.csv")
    
    # Initialize and fit preprocessor
    processor = AlzheimerPreprocessor()
    processor.fit(df)
    
    # Save processor
    processor.save_preprocessor()
    print(f"✅ Preprocessor saved with {processor.n_features} features")