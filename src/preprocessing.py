import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import joblib

class AlzheimerPreprocessor:
    def __init__(self):
        self.target_col = 'Alzheimer’s Diagnosis'
        self.numerical_features = ['Age', 'BMI', 'Cognitive Test Score']
        self.ordinal_features = {
            'Depression Level': ['Low', 'Medium', 'High'],
            'Stress Levels': ['Low', 'Medium', 'High'],
            'Physical Activity Level': ['Low', 'Medium', 'High']
        }
        self.categorical_features = [
            'Gender', 'Smoking Status', 'Diabetes',
            'Hypertension', 'Family History of Alzheimer’s',
            'Genetic Risk Factor (APOE-ε4 allele)'
        ]
        self.drop_columns = ['Country', 'Education Level']
        self.feature_selector = None
        self.preprocessor = None

    def clean_data(self, df):
        """Data cleaning pipeline"""
        # Convert Cognitive Test Score to numeric
        df['Cognitive Test Score'] = pd.to_numeric(
            df['Cognitive Test Score'], errors='coerce'
        )
        
        # Handle missing values
        df = df.dropna(subset=[self.target_col], how='any')
        
        # Clean ordinal features
        for col, categories in self.ordinal_features.items():
            df[col] = np.where(df[col].isin(categories), df[col], np.nan)
            
        return df

    def get_feature_names(self):
        """Get readable feature names after transformations"""
        # Get feature names from each transformer
        num_names = self.numerical_features
        ord_names = list(self.ordinal_features.keys())
        cat_names = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(self.categorical_features)
        
        # Combine all names
        all_names = np.concatenate([num_names, ord_names, cat_names])
        
        # Return selected names
        return all_names[self.feature_selector.get_support()]
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Proper serialization with feature names"""
        self.feature_names_ = self.get_feature_names()  # Store feature names
        joblib.dump(self, path)  # Save entire object


    def fit(self, df):
        """Full preprocessing pipeline"""
        # Clean and validate data
        df = self.clean_data(df)
        self._validate_dataset(df)
        
        X = df.drop(columns=[self.target_col]+self.drop_columns)
        y = df[self.target_col].map({'Yes': 1, 'No': 0})

        # Build preprocessing pipeline
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        ordinal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[
                self.ordinal_features[col] for col in self.ordinal_features
            ]))
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numerical_transformer, self.numerical_features),
            ('ord', ordinal_transformer, list(self.ordinal_features.keys())),
            ('cat', categorical_transformer, self.categorical_features)
        ])

        # Fit preprocessing
        X_processed = self.preprocessor.fit_transform(X)
        
        # Feature selection with validation
        self.feature_selector = SelectFromModel(
            LassoCV(cv=min(5, len(X_processed)-1))  # Dynamic CV folds
        )
        self.feature_selector.fit(X_processed, y)
        
        return self

    def _validate_dataset(self, df):
        """Ensure dataset meets requirements"""
        if len(df) < 10:
            raise ValueError(f"Dataset too small ({len(df)} samples). Minimum 10 samples required.")
            
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' missing")
            
        for col in self.numerical_features:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Numerical feature {col} contains non-numeric values")

    def transform(self, df):
        """Transform new data"""
        df = self.clean_data(df)
        X = df.drop(columns=[self.target_col]+self.drop_columns)
        return self.feature_selector.transform(self.preprocessor.transform(X))

    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save preprocessing pipeline"""
        joblib.dump({
            'preprocessor': self.preprocessor,
            'selector': self.feature_selector,
            'features': self.feature_selector.get_feature_names_out()
        }, path)

# Main execution with actual data
if __name__ == "__main__":
    try:
        # Load raw data
        df = pd.read_csv("alzheimer-app/data/raw/alzheimers_prediction_dataset.csv")
        
        # Process data
        processor = AlzheimerPreprocessor()
        processor.fit(df)
        processor.save_preprocessor()
        
        print("Data preprocessing completed successfully!")
        print("Selected Features:", processor.get_feature_names())  # Updated line
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTROUBLESHOOTING GUIDE:")
        print("1. Check file path to raw data exists")
        print("2. Verify dataset contains at least 10 samples")
        print("3. Ensure all numerical columns contain only numbers")
        print("4. Confirm ordinal features use only defined categories (Low/Medium/High)")
        print("5. Check target column exists with values 'Yes'/'No'")