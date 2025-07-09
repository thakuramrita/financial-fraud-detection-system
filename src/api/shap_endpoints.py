"""
API endpoints for SHAP-based model explanations
Provides REST API access to model interpretability features
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import the SHAP interpreter (would be from the actual module)
# from src.models.shap_interpreter import SHAPInterpreter

logger = logging.getLogger(__name__)

# Create router for SHAP endpoints
router = APIRouter(
    prefix="/api/v1/explain",
    tags=["Model Explanations"],
    responses={404: {"description": "Not found"}},
)


class TransactionExplanationRequest(BaseModel):
    """Request model for transaction explanation"""
    transaction_id: str = Field(..., description="Transaction ID to explain")
    model_name: Optional[str] = Field(None, description="Specific model to use (default: best performing)")
    plot_type: Optional[str] = Field("waterfall", description="Type of visualization: waterfall, force, bar")
    include_visualization: bool = Field(True, description="Whether to include base64 encoded plot")


class FeatureContribution(BaseModel):
    """Feature contribution to prediction"""
    feature: str
    value: float
    shap_value: float
    percentage_impact: float


class TransactionExplanationResponse(BaseModel):
    """Response model for transaction explanation"""
    transaction_id: str
    model_used: str
    prediction: str
    fraud_probability: float
    confidence: str
    top_positive_factors: List[FeatureContribution]
    top_negative_factors: List[FeatureContribution]
    explanation_summary: str
    visualization: Optional[str] = None  # Base64 encoded plot
    timestamp: datetime


class GlobalImportanceRequest(BaseModel):
    """Request model for global feature importance"""
    model_name: Optional[str] = Field(None, description="Specific model to analyze")
    top_k: int = Field(10, description="Number of top features to return")
    include_statistics: bool = Field(True, description="Include detailed statistics")


class FeatureImportance(BaseModel):
    """Feature importance information"""
    rank: int
    feature: str
    mean_abs_shap: float
    mean_shap: float
    std_shap: float
    direction: str  # "positive" or "negative"


class GlobalImportanceResponse(BaseModel):
    """Response model for global feature importance"""
    model_name: str
    features_analyzed: int
    top_features: List[FeatureImportance]
    summary: str
    timestamp: datetime


@router.post("/transaction", response_model=TransactionExplanationResponse)
async def explain_transaction(request: TransactionExplanationRequest):
    """
    Explain why a specific transaction was classified as fraud/legitimate.
    
    This endpoint provides detailed SHAP-based explanations for individual
    transaction predictions, helping fraud analysts understand model decisions.
    """
    try:
        # In production, this would:
        # 1. Load the transaction data
        # 2. Get the SHAP interpreter
        # 3. Generate explanation
        # 4. Create visualization
        
        # Mock response for demonstration
        response = TransactionExplanationResponse(
            transaction_id=request.transaction_id,
            model_used=request.model_name or "xgboost",
            prediction="Fraudulent",
            fraud_probability=0.87,
            confidence="High",
            top_positive_factors=[
                FeatureContribution(
                    feature="transaction_amount",
                    value=5420.50,
                    shap_value=0.35,
                    percentage_impact=28.5
                ),
                FeatureContribution(
                    feature="merchant_risk_score",
                    value=0.89,
                    shap_value=0.28,
                    percentage_impact=22.8
                ),
                FeatureContribution(
                    feature="unusual_time",
                    value=1.0,
                    shap_value=0.21,
                    percentage_impact=17.1
                )
            ],
            top_negative_factors=[
                FeatureContribution(
                    feature="account_age_days",
                    value=1825,
                    shap_value=-0.15,
                    percentage_impact=-12.2
                ),
                FeatureContribution(
                    feature="previous_transactions",
                    value=342,
                    shap_value=-0.08,
                    percentage_impact=-6.5
                )
            ],
            explanation_summary=(
                "This transaction was flagged as fraudulent primarily due to: "
                "1) Unusually high transaction amount ($5,420.50) compared to user history, "
                "2) High-risk merchant score (0.89), and "
                "3) Transaction occurred at unusual time. "
                "Mitigating factors include long account history and many previous legitimate transactions."
            ),
            visualization=None,  # Would contain base64 encoded plot
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error explaining transaction {request.transaction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/importance", response_model=GlobalImportanceResponse)
async def get_global_feature_importance(
    model_name: Optional[str] = Query(None, description="Model to analyze"),
    top_k: int = Query(10, description="Number of top features", ge=1, le=50)
):
    """
    Get global feature importance based on SHAP values.
    
    This endpoint provides insights into which features are most important
    for fraud detection across all predictions.
    """
    try:
        # Mock response for demonstration
        features = []
        feature_names = [
            "transaction_amount", "merchant_risk_score", "unusual_time",
            "account_age_days", "previous_transactions", "velocity_1h",
            "country_risk", "device_fingerprint_match", "billing_shipping_match",
            "weekend_transaction"
        ]
        
        for i, feature in enumerate(feature_names[:top_k]):
            importance = FeatureImportance(
                rank=i + 1,
                feature=feature,
                mean_abs_shap=abs(np.random.normal(0.2, 0.1)),
                mean_shap=np.random.normal(0.1, 0.15),
                std_shap=abs(np.random.normal(0.05, 0.02)),
                direction="positive" if i < 5 else "negative"
            )
            features.append(importance)
        
        response = GlobalImportanceResponse(
            model_name=model_name or "xgboost",
            features_analyzed=47,  # Total number of features
            top_features=features,
            summary=(
                f"Analysis of {model_name or 'xgboost'} model shows that "
                f"transaction characteristics (amount, timing) and merchant risk "
                f"are the strongest predictors of fraud. Account history features "
                f"generally indicate legitimate behavior."
            ),
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def explain_batch_transactions(
    transaction_ids: List[str],
    model_name: Optional[str] = None
):
    """
    Explain multiple transactions in batch.
    
    Useful for investigating patterns across multiple flagged transactions.
    """
    try:
        # In production, this would process multiple transactions efficiently
        explanations = []
        
        for txn_id in transaction_ids[:10]:  # Limit to 10 for demo
            # Would call actual explanation logic here
            explanations.append({
                "transaction_id": txn_id,
                "fraud_probability": np.random.uniform(0.5, 0.95),
                "primary_factor": np.random.choice([
                    "high_amount", "risky_merchant", "unusual_pattern", "velocity"
                ])
            })
        
        return {
            "explanations": explanations,
            "summary": f"Analyzed {len(explanations)} transactions",
            "common_patterns": ["high_amounts", "risky_merchants"],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in batch explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{model_name}")
async def generate_interpretability_report(
    model_name: str,
    include_samples: bool = Query(True, description="Include sample explanations")
):
    """
    Generate comprehensive interpretability report for a model.
    
    This creates a detailed report with SHAP analysis, feature importance,
    and sample explanations suitable for regulatory compliance.
    """
    try:
        # In production, this would generate actual report
        report_info = {
            "model_name": model_name,
            "report_id": f"SHAP_REPORT_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "generated",
            "sections": [
                "Executive Summary",
                "Global Feature Importance",
                "Feature Impact Analysis",
                "Sample Transaction Explanations",
                "Model Behavior Insights",
                "Recommendations"
            ],
            "download_url": f"/api/v1/explain/report/download/{model_name}",
            "expires_at": datetime.now().isoformat(),
            "timestamp": datetime.now()
        }
        
        return report_info
        
    except Exception as e:
        logger.error(f"Error generating report for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def explanation_service_health():
    """Check if SHAP explanation service is healthy."""
    return {
        "status": "healthy",
        "service": "SHAP Explanation API",
        "models_available": ["xgboost", "randomforest", "logisticregression"],
        "version": "1.0.0",
        "timestamp": datetime.now()
    }


# Additional endpoints that could be implemented:

# @router.post("/features/interaction")
# async def analyze_feature_interactions(feature1: str, feature2: str, model_name: str = None):
#     """Analyze SHAP interaction values between two features."""
#     pass

# @router.get("/explain/realtime/{transaction_id}")
# async def get_realtime_explanation(transaction_id: str):
#     """Get cached real-time explanation for recent transaction."""
#     pass

# @router.post("/explain/compare")
# async def compare_model_explanations(transaction_id: str, models: List[str]):
#     """Compare explanations across different models for the same transaction."""
#     pass