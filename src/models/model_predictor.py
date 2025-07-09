"""
Model Predictor for Financial Fraud Detection
Loads trained models and makes predictions on new transaction data.
"""

import os
import joblib
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')


class FraudDetectionPredictor:
    """Predictor for fraud detection using trained models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.logger = logging.getLogger(__name__)
        
        # Load models and preprocessing
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessing objects."""
        self.logger.info("Loading trained models...")
        
        # Find model files
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        if not model_files:
            raise FileNotFoundError("No trained models found in models directory")
        
        # Load preprocessing objects
        preprocessing_files = [f for f in model_files if f.startswith('preprocessing_')]
        if preprocessing_files:
            latest_preprocessing = max(preprocessing_files)
            preprocessing_path = os.path.join(self.models_dir, latest_preprocessing)
            preprocessing_data = joblib.load(preprocessing_path)
            self.scalers = preprocessing_data['scalers']
            self.label_encoders = preprocessing_data['label_encoders']
        
        # Load models
        model_files = [f for f in model_files if not f.startswith('preprocessing_')]
        for model_file in model_files:
            model_name = model_file.split('_')[0]
            model_path = os.path.join(self.models_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            self.logger.info(f"Loaded {model_name} model")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        df = data.copy()
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Apply label encoding
        if self.label_encoders:
            for col in df.columns:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = df[col].unique()
                    for val in unique_values:
                        if val not in self.label_encoders[col].classes_:
                            df[col] = df[col].replace(val, self.label_encoders[col].classes_[0])
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Apply scaling
        if 'numerical' in self.scalers:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = self.scalers['numerical'].transform(df[numerical_cols])
        
        return df
    
    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make predictions using trained models.
        
        Args:
            data: Input data for prediction
            model_name: Specific model to use (if None, uses all models)
        
        Returns:
            Dictionary containing predictions and probabilities
        """
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        results = {}
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        for name, model in models_to_use.items():
            try:
                # Make predictions
                predictions = model.predict(processed_data)
                probabilities = model.predict_proba(processed_data)[:, 1]  # Probability of fraud
                
                results[name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'fraud_risk': self._calculate_fraud_risk(probabilities)
                }
                
            except Exception as e:
                self.logger.error(f"Error making predictions with {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _calculate_fraud_risk(self, probabilities: np.ndarray) -> List[str]:
        """Calculate fraud risk levels based on probabilities."""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append('Low')
            elif prob < 0.7:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        return risk_levels
    
    def predict_single_transaction(self, transaction_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction_data: Dictionary containing transaction features
            model_name: Specific model to use
        
        Returns:
            Prediction results for the transaction
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Make prediction
        results = self.predict(df, model_name)
        
        # Format results for single transaction
        formatted_results = {}
        for model_name, result in results.items():
            if 'error' not in result:
                formatted_results[model_name] = {
                    'is_fraud': bool(result['predictions'][0]),
                    'fraud_probability': float(result['probabilities'][0]),
                    'fraud_risk': result['fraud_risk'][0]
                }
            else:
                formatted_results[model_name] = {'error': result['error']}
        
        return formatted_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'models_loaded': list(self.models.keys()),
            'preprocessing_available': bool(self.scalers and self.label_encoders),
            'total_models': len(self.models)
        }
        
        # Add model-specific info
        for name, model in self.models.items():
            info[name] = {
                'type': type(model).__name__,
                'has_feature_importance': hasattr(model, 'feature_importances_')
            }
        
        return info


def main():
    """Main function to perform batch prediction on prediction_input.csv and save results."""
    # Setup logging to file and console
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"model_predictor_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing Fraud Detection Predictor...")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Initialize predictor
        predictor = FraudDetectionPredictor()
        
        # Get model info
        model_info = predictor.get_model_info()
        print("\n" + "="*50)
        print("MODEL PREDICTOR INITIALIZED")
        print("="*50)
        print(f"Models loaded: {model_info['models_loaded']}")
        print(f"Total models: {model_info['total_models']}")
        
        # Batch prediction on prediction_input.csv
        input_path = "data/prediction_input.csv"
        output_path = "data/prediction_output.csv"
        logger.info(f"Loading prediction input from: {input_path}")
        input_df = pd.read_csv(input_path)
        logger.info(f"Performing batch prediction on {len(input_df)} samples...")
        results = predictor.predict(input_df)
        
        # Prepare output DataFrame
        output_df = input_df.copy()
        for model_name, result in results.items():
            if 'error' not in result:
                output_df[f'{model_name}_is_fraud'] = result['predictions']
                output_df[f'{model_name}_fraud_probability'] = result['probabilities']
                output_df[f'{model_name}_fraud_risk'] = result['fraud_risk']
            else:
                logger.error(f"Prediction error for {model_name}: {result['error']}")
                output_df[f'{model_name}_error'] = result['error']
        
        output_df.to_csv(output_path, index=False)
        logger.info(f"Prediction output saved to: {output_path}")
        print(f"\nPrediction completed! Results saved to: {output_path}")
        print(f"Log file: {log_file}")
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        print(f"\n❌ Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 