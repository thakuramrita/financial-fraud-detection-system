"""
Model Evaluator for Financial Fraud Detection
Provides comprehensive evaluation and interpretability analysis.
"""

import os
import joblib
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.inspection import permutation_importance
import shap

warnings.filterwarnings('ignore')


class FraudDetectionModelEvaluator:
    """Evaluator for fraud detection models with interpretability analysis."""
    
    def __init__(self, models_dir: str = "models", data_path: str = "data/features.csv"):
        """
        Initialize the model evaluator.
        
        Args:
            models_dir: Directory containing trained models
            data_path: Path to the features CSV file
        """
        self.models_dir = models_dir
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
        
        # Load models and data
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.load_models()
        
    def load_models(self):
        """Load trained models and preprocessing objects."""
        self.logger.info("Loading trained models...")
        
        # Find the latest model files
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
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for evaluation."""
        self.logger.info(f"Loading data from {self.data_path}")
        
        # Load features
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        target_col = 'is_fraud'
        exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud', 'fraud_reason']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Apply preprocessing
        if self.label_encoders:
            for col in X.columns:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        if 'numerical' in self.scalers:
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            X[numerical_cols] = self.scalers['numerical'].transform(X[numerical_cols])
        
        return X, y
    
    def evaluate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y).mean(),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba),
            'average_precision': average_precision_score(y, y_pred_proba)
        }
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def analyze_feature_importance(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance for a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Permutation importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Feature importance scores
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Model-specific feature importance (if available)
        model_importance = None
        if hasattr(model, 'feature_importances_'):
            model_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return {
            'permutation_importance': feature_importance,
            'model_importance': model_importance
        }
    
    def create_shap_analysis(self, model_name: str, X: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
        """Create SHAP analysis for model interpretability."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Sample data for SHAP analysis (to avoid memory issues)
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict_proba, X_sample[:100])
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, X_sample[:100])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, get SHAP values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'X_sample': X_sample
            }
        except Exception as e:
            self.logger.warning(f"SHAP analysis failed for {model_name}: {str(e)}")
            return None
    
    def plot_evaluation_results(self, results: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, save_path: str = "models"):
        """Plot comprehensive evaluation results."""
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Metrics comparison
        model_names = list(results.keys())
        metrics = ['auc', 'precision', 'recall', 'f1']
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[name]['metrics'][metric] for name in model_names]
            axes[0, 0].bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Metrics')
        axes[0, 0].set_xticks(x + width*1.5)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC curves
        for model_name in model_names:
            y_pred_proba = results[model_name]['probabilities']
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc_score = results[model_name]['metrics']['auc']
            axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})', linewidth=2)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall curves
        for model_name in model_names:
            y_pred_proba = results[model_name]['probabilities']
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            ap_score = results[model_name]['metrics']['average_precision']
            axes[0, 2].plot(recall, precision, label=f'{model_name} (AP={ap_score:.3f})', linewidth=2)
        
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confusion matrix for best model
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['auc'])[0]
        cm = results[best_model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # 5. Feature importance (for best model)
        if hasattr(self.models[best_model], 'feature_importances_'):
            importances = self.models[best_model].feature_importances_
            feature_names = X.columns
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            axes[1, 1].bar(range(len(indices)), importances[indices])
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Importance')
            axes[1, 1].set_title(f'Feature Importance - {best_model}')
            axes[1, 1].set_xticks(range(len(indices)))
            axes[1, 1].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        # 6. Model comparison radar chart
        metrics_radar = ['auc', 'precision', 'recall', 'f1']
        angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name in model_names:
            values = [results[model_name]['metrics'][metric] for metric in metrics_radar]
            values += values[:1]  # Complete the circle
            axes[1, 2].plot(angles, values, 'o-', linewidth=2, label=model_name)
            axes[1, 2].fill(angles, values, alpha=0.25)
        
        axes[1, 2].set_xticks(angles[:-1])
        axes[1, 2].set_xticklabels([m.upper() for m in metrics_radar])
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('Model Performance Radar')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, f"evaluation_results_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved evaluation plots to {plot_path}")
        return plot_path
    
    def generate_evaluation_report(self, results: Dict[str, Any], y: np.ndarray, save_path: str = "models"):
        """Generate a comprehensive evaluation report."""
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(save_path, f"evaluation_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FINANCIAL FRAUD DETECTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_path}\n\n")
            
            # Model rankings
            f.write("MODEL RANKINGS (by AUC):\n")
            f.write("-" * 40 + "\n")
            
            sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['auc'], reverse=True)
            
            for i, (model_name, result) in enumerate(sorted_models, 1):
                f.write(f"{i}. {model_name.upper()}\n")
                f.write(f"   AUC: {result['metrics']['auc']:.4f}\n")
                f.write(f"   Precision: {result['metrics']['precision']:.4f}\n")
                f.write(f"   Recall: {result['metrics']['recall']:.4f}\n")
                f.write(f"   F1-Score: {result['metrics']['f1']:.4f}\n")
                f.write(f"   Average Precision: {result['metrics']['average_precision']:.4f}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("=" * 40 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name.upper()}:\n")
                f.write("-" * 20 + "\n")
                
                f.write("Metrics:\n")
                for metric, value in result['metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
                
                f.write("Classification Report:\n")
                f.write(classification_report(y, result['predictions']))
                f.write("\n")
                
                f.write("Confusion Matrix:\n")
                f.write(str(result['confusion_matrix']))
                f.write("\n\n")
        
        self.logger.info(f"Generated evaluation report: {report_path}")
        return report_path


def main():
    """Main function to run model evaluation."""
    # Setup logging to file and console
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"model_evaluation_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Financial Fraud Detection Model Evaluation...")
    
    # Initialize evaluator
    evaluator = FraudDetectionModelEvaluator()
    
    try:
        # Prepare data
        X, y = evaluator.prepare_data()
        
        # Evaluate all models
        results = {}
        for model_name in evaluator.models.keys():
            logger.info(f"Evaluating {model_name}...")
            results[model_name] = evaluator.evaluate_model(model_name, X, y)
        
        # Generate plots and report
        evaluator.plot_evaluation_results(results, X, y)
        report_path = evaluator.generate_evaluation_report(results, y)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL EVALUATION COMPLETED")
        print("="*60)
        
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['auc'])
        print(f"Best Model: {best_model[0]}")
        print(f"Best AUC: {best_model[1]['metrics']['auc']:.4f}")
        print(f"Best F1: {best_model[1]['metrics']['f1']:.4f}")
        
        print(f"\nEvaluation report: {report_path}")
        print(f"Log file: {log_file}")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main() 