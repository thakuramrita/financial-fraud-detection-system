"""
SHAP Interpreter for Financial Fraud Detection Models
Provides comprehensive model interpretability using SHAP (SHapley Additive exPlanations)
"""

import os
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')


class SHAPInterpreter:
    """SHAP-based model interpreter for fraud detection models."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize SHAP interpreter with trained models."""
        self.models_dir = models_dir
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.explainers = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models from directory."""
        model_files = [f for f in os.listdir(self.models_dir) 
                      if f.endswith('.joblib') and not f.startswith('preprocessing_')]
        
        for model_file in model_files:
            model_name = model_file.split('_')[0]
            model_path = os.path.join(self.models_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            self.logger.info(f"Loaded {model_name} model for SHAP analysis")
    
    def create_explainer(self, model_name: str, X_background: pd.DataFrame) -> shap.Explainer:
        """Create appropriate SHAP explainer for a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Choose appropriate explainer based on model type
        if model_name in ['xgboost', 'randomforest']:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
        elif model_name == 'logisticregression':
            # Linear model
            explainer = shap.LinearExplainer(model, X_background)
        else:
            # Generic explainer for other models (SVM, Neural Network)
            # Use a sample of background data for efficiency
            background_sample = shap.sample(X_background, min(100, len(X_background)))
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                background_sample
            )
        
        self.explainers[model_name] = explainer
        return explainer
    
    def explain_transaction(
        self, 
        transaction_data: pd.DataFrame,
        model_name: str,
        plot: bool = True,
        plot_type: str = 'waterfall'
    ) -> Dict[str, Any]:
        """
        Explain a single transaction's fraud prediction.
        
        Args:
            transaction_data: Single transaction features (1 row DataFrame)
            model_name: Name of the model to use
            plot: Whether to generate visualization
            plot_type: Type of plot ('waterfall', 'force', 'bar')
        
        Returns:
            Dictionary with SHAP values and prediction details
        """
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for {model_name}. Run create_explainer first.")
        
        model = self.models[model_name]
        explainer = self.explainers[model_name]
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(transaction_data)[0, 1]
            prediction = model.predict(transaction_data)[0]
        else:
            prediction = model.predict(transaction_data)[0]
            prediction_proba = prediction
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(transaction_data)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Binary classification - use positive class
            shap_values = shap_values[1]
        
        if plot:
            if plot_type == 'waterfall':
                # Waterfall plot
                shap.waterfall_plot(
                    shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                    data=transaction_data.iloc[0],
                                    feature_names=transaction_data.columns.tolist())
                )
            elif plot_type == 'force':
                # Force plot
                shap.force_plot(
                    explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                    shap_values[0],
                    transaction_data.iloc[0],
                    matplotlib=True
                )
            elif plot_type == 'bar':
                # Bar plot
                shap.bar_plot(
                    shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                    data=transaction_data.iloc[0],
                                    feature_names=transaction_data.columns.tolist())
                )
        
        # Get feature contributions
        feature_contributions = pd.DataFrame({
            'feature': transaction_data.columns,
            'value': transaction_data.iloc[0].values,
            'shap_value': shap_values[0],
            'abs_shap_value': np.abs(shap_values[0])
        }).sort_values('abs_shap_value', ascending=False)
        
        return {
            'prediction': prediction,
            'prediction_probability': prediction_proba,
            'shap_values': shap_values[0],
            'feature_contributions': feature_contributions,
            'top_positive_factors': feature_contributions[feature_contributions['shap_value'] > 0].head(5),
            'top_negative_factors': feature_contributions[feature_contributions['shap_value'] < 0].head(5)
        }
    
    def global_feature_importance(
        self,
        model_name: str,
        X: pd.DataFrame,
        plot: bool = True,
        plot_type: str = 'summary'
    ) -> pd.DataFrame:
        """
        Calculate global feature importance using SHAP values.
        
        Args:
            model_name: Name of the model
            X: Feature data
            plot: Whether to generate visualization
            plot_type: Type of plot ('summary', 'bar', 'beeswarm')
        
        Returns:
            DataFrame with feature importance rankings
        """
        if model_name not in self.explainers:
            self.create_explainer(model_name, X)
        
        explainer = self.explainers[model_name]
        
        # Calculate SHAP values for dataset
        self.logger.info(f"Calculating SHAP values for {len(X)} samples...")
        shap_values = explainer.shap_values(X)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': shap_values.mean(axis=0),
            'std_shap': shap_values.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        if plot:
            if plot_type == 'summary':
                shap.summary_plot(shap_values, X, show=True)
            elif plot_type == 'bar':
                shap.summary_plot(shap_values, X, plot_type="bar", show=True)
            elif plot_type == 'beeswarm':
                shap.plots.beeswarm(
                    shap.Explanation(values=shap_values,
                                    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                    data=X.values,
                                    feature_names=X.columns.tolist())
                )
        
        return feature_importance
    
    def dependence_analysis(
        self,
        model_name: str,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None
    ):
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            model_name: Name of the model
            X: Feature data
            feature: Feature to analyze
            interaction_feature: Feature to color by (auto-selected if None)
        """
        if model_name not in self.explainers:
            self.create_explainer(model_name, X)
        
        explainer = self.explainers[model_name]
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create dependence plot
        shap.dependence_plot(
            feature,
            shap_values,
            X,
            interaction_index=interaction_feature,
            show=True
        )
    
    def generate_interpretability_report(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        sample_transactions: Optional[pd.DataFrame] = None,
        save_path: str = "reports"
    ) -> str:
        """
        Generate comprehensive interpretability report for a model.
        
        Args:
            model_name: Name of the model
            X: Feature data
            y: Target labels
            sample_transactions: Sample transactions to explain individually
            save_path: Directory to save report
        
        Returns:
            Path to generated report
        """
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(save_path, f"shap_report_{model_name}_{timestamp}.md")
        
        # Create explainer if needed
        if model_name not in self.explainers:
            self.create_explainer(model_name, X)
        
        # Get global feature importance
        feature_importance = self.global_feature_importance(model_name, X, plot=False)
        
        # Generate report
        with open(report_path, 'w') as f:
            f.write(f"# SHAP Interpretability Report - {model_name.upper()}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"This report provides interpretability analysis for the {model_name} fraud detection model using SHAP (SHapley Additive exPlanations).\n\n")
            
            f.write("## Global Feature Importance\n\n")
            f.write("Top 10 most important features based on mean absolute SHAP values:\n\n")
            f.write("| Rank | Feature | Mean |SHAP| | Mean SHAP | Std SHAP |\n")
            f.write("|------|---------|------------|-----------|----------|\n")
            
            for i, row in feature_importance.head(10).iterrows():
                f.write(f"| {i+1} | {row['feature']} | {row['mean_abs_shap']:.4f} | {row['mean_shap']:.4f} | {row['std_shap']:.4f} |\n")
            
            f.write("\n## Feature Impact Analysis\n\n")
            f.write("### Positive Impact Features (increase fraud probability):\n")
            positive_features = feature_importance[feature_importance['mean_shap'] > 0].head(5)
            for _, row in positive_features.iterrows():
                f.write(f"- **{row['feature']}**: Mean SHAP = {row['mean_shap']:.4f}\n")
            
            f.write("\n### Negative Impact Features (decrease fraud probability):\n")
            negative_features = feature_importance[feature_importance['mean_shap'] < 0].head(5)
            for _, row in negative_features.iterrows():
                f.write(f"- **{row['feature']}**: Mean SHAP = {row['mean_shap']:.4f}\n")
            
            if sample_transactions is not None:
                f.write("\n## Sample Transaction Explanations\n\n")
                for i in range(min(3, len(sample_transactions))):
                    transaction = sample_transactions.iloc[[i]]
                    explanation = self.explain_transaction(transaction, model_name, plot=False)
                    
                    f.write(f"### Transaction {i+1}\n")
                    f.write(f"- Prediction: {'Fraud' if explanation['prediction'] == 1 else 'Legitimate'}\n")
                    f.write(f"- Fraud Probability: {explanation['prediction_probability']:.2%}\n")
                    f.write("\nTop Contributing Factors:\n")
                    
                    for _, factor in explanation['top_positive_factors'].iterrows():
                        f.write(f"- {factor['feature']} = {factor['value']:.2f} (SHAP: +{factor['shap_value']:.4f})\n")
                    
                    for _, factor in explanation['top_negative_factors'].iterrows():
                        f.write(f"- {factor['feature']} = {factor['value']:.2f} (SHAP: {factor['shap_value']:.4f})\n")
                    f.write("\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the SHAP analysis:\n\n")
            f.write("1. **Feature Engineering**: Focus on the top contributing features for better fraud detection\n")
            f.write("2. **Model Monitoring**: Track changes in feature importance over time\n")
            f.write("3. **Business Rules**: Consider creating rules based on high-impact feature combinations\n")
            f.write("4. **Data Quality**: Ensure high quality data for the most important features\n")
        
        self.logger.info(f"Generated SHAP interpretability report: {report_path}")
        return report_path


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize interpreter
    interpreter = SHAPInterpreter()
    
    # Load some test data (placeholder)
    # In production, this would load actual feature data
    print("SHAP Interpreter initialized successfully!")
    print("Available models:", list(interpreter.models.keys()))
    print("\nTo use the interpreter:")
    print("1. Create explainer: interpreter.create_explainer('xgboost', X_train)")
    print("2. Explain transaction: interpreter.explain_transaction(transaction_df, 'xgboost')")
    print("3. Global importance: interpreter.global_feature_importance('xgboost', X_test)")