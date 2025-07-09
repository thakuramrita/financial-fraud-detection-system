# Model Building Implementation Guide

## 📋 Quick Start Checklist

### Phase 1: Data Preparation (Days 1-2)
```python
# 1. Load and explore data
python -c "
import pandas as pd
df = pd.read_csv('data/transactions.csv')
print(df.info())
print(df['is_fraud'].value_counts())
"

# 2. Create feature engineering pipeline
python src/features/feature_engineering.py

# 3. Split data for training
python src/data/data_splitter.py
```

### Phase 2: Model Training (Days 3-5)
```python
# Train all models
python src/models/train_model.py --all

# Train specific model
python src/models/train_model.py --model xgboost

# Hyperparameter tuning
python src/models/hyperparameter_tuning.py
```

### Phase 3: Evaluation (Days 6-7)
```python
# Evaluate models
python src/models/model_evaluator.py

# Generate explainability reports
python src/models/explainability.py
```

### Phase 4: Deployment (Days 8-10)
```bash
# Start prediction API
uvicorn src.api.main:app --reload

# Test predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @data/sample_transaction.json
```

## 🏗️ File Structure to Create

```
src/
├── models/
│   ├── train_model.py          # Main training script
│   ├── model_evaluator.py      # Evaluation framework
│   ├── model_predictor.py      # Prediction service
│   ├── explainability.py       # SHAP/LIME integration
│   ├── model_registry.py       # Model versioning
│   └── hyperparameter_tuning.py # Hyperparameter optimization
├── features/
│   ├── feature_engineering.py  # Feature creation
│   ├── feature_selector.py     # Feature selection
│   └── feature_store.py        # Feature storage
├── data/
│   ├── data_splitter.py        # Train/val/test splits
│   ├── data_validator.py       # Data quality checks
│   └── data_preprocessor.py    # Preprocessing pipeline
└── api/
    ├── routes/
    │   ├── prediction.py       # Prediction endpoints
    │   ├── model_management.py # Model CRUD operations
    │   └── monitoring.py       # Health checks
    └── schemas/
        ├── prediction.py       # Request/response models
        └── model.py           # Model metadata schemas
```

## 💻 Sample Implementation Code

### 1. Feature Engineering Pipeline
```python
# src/features/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_features(self, df):
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Velocity features
        df['trans_count_1h'] = df.groupby('user_id')['transaction_id'].transform(
            lambda x: x.rolling('1H').count()
        )
        
        # Amount features
        df['amount_zscore'] = df.groupby('user_id')['amount'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        return df
```

### 2. Model Training Pipeline
```python
# src/models/train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

class ModelTrainer:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = self._get_model()
    
    def _get_model(self):
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10  # Handle imbalance
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced'
            )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    def save_model(self, path):
        joblib.dump(self.model, path)
```

### 3. Model Evaluation
```python
# src/models/model_evaluator.py
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate(self):
        # Predictions
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Cost-based evaluation
        fraud_cost = 1000  # Cost of missing fraud
        false_positive_cost = 50  # Cost of false alarm
        
        return {
            'roc_auc': roc_auc,
            'predictions': y_pred_proba
        }
```

### 4. FastAPI Prediction Endpoint
```python
# src/api/routes/prediction.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib

router = APIRouter()

class TransactionRequest(BaseModel):
    user_id: str
    amount: float
    merchant_category: str
    timestamp: str

@router.post("/predict")
async def predict_fraud(transaction: TransactionRequest):
    # Load model
    model = joblib.load('models/fraud_detector.pkl')
    
    # Prepare features
    features = prepare_features(transaction)
    
    # Predict
    fraud_probability = model.predict_proba(features)[0, 1]
    
    return {
        "fraud_probability": float(fraud_probability),
        "is_fraud": fraud_probability > 0.5,
        "confidence": abs(fraud_probability - 0.5) * 2
    }
```

## 🔧 Configuration Files

### 1. Model Configuration
```yaml
# config/model_config.yaml
models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    scale_pos_weight: 10
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    class_weight: balanced

evaluation:
  metrics:
    - roc_auc
    - precision
    - recall
    - f1
  
  thresholds:
    - 0.3
    - 0.5
    - 0.7
```

### 2. Training Configuration
```yaml
# config/training_config.yaml
data:
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  random_state: 42

preprocessing:
  handle_missing: true
  scale_features: true
  encode_categorical: true

hyperparameter_tuning:
  cv_folds: 5
  scoring: roc_auc
  n_iter: 50
```

## 📊 Expected Outputs

1. **Model Performance Report**
```
Model Performance Summary
========================
XGBoost:
  - ROC-AUC: 0.956
  - Precision@0.5: 0.85
  - Recall@0.5: 0.78
  - F1-Score: 0.81

Random Forest:
  - ROC-AUC: 0.942
  - Precision@0.5: 0.82
  - Recall@0.5: 0.75
  - F1-Score: 0.78
```

2. **API Response Example**
```json
{
  "fraud_probability": 0.823,
  "is_fraud": true,
  "confidence": 0.646,
  "explanation": {
    "top_factors": [
      "Unusual transaction amount",
      "High velocity of transactions",
      "Merchant category risk"
    ]
  }
}
```

## 🚀 Next Steps After Implementation

1. **Performance Optimization**
   - Implement model caching
   - Add batch prediction endpoints
   - Optimize feature computation

2. **Monitoring Setup**
   - Track prediction latency
   - Monitor model drift
   - Set up automated retraining

3. **Advanced Features**
   - Add real-time feature updates
   - Implement A/B testing framework
   - Integrate with fraud case management

## 📚 Additional Resources

- [Fraud Detection Best Practices](https://github.com/topics/fraud-detection)
- [XGBoost Parameter Tuning Guide](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [FastAPI Production Deployment](https://fastapi.tiangolo.com/deployment/)
- [Model Monitoring with Evidently](https://github.com/evidentlyai/evidently)