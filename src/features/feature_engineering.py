"""
Feature Engineering Pipeline for Fraud Detection System

This module implements comprehensive feature engineering for financial transaction
fraud detection, including transaction-level features, user behavior patterns,
merchant risk scoring, and advanced ML features.

Issue: #1 - Feature Engineering Implementation for Fraud Detection System
Author: Financial Fraud Detection System
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for fraud detection.
    
    This class implements all feature engineering components outlined in issue #1:
    - Transaction-level features (amount, temporal, geographic)
    - User behavior features (historical patterns, deviations, velocity)
    - Merchant/category features (risk scoring, network analysis)
    - Advanced features (aggregations, interactions, anomaly detection)
    """
    
    def __init__(self, enable_advanced_features: bool = True):
        """
        Initialize the Feature Engineer.
        
        Args:
            enable_advanced_features: Whether to compute expensive advanced features
        """
        self.enable_advanced_features = enable_advanced_features
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.user_profiles = {}
        self.merchant_profiles = {}
        
        logger.info("FeatureEngineer initialized")
    
    def engineer_features(self, 
                         transactions_df: pd.DataFrame,
                         users_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            transactions_df: Transaction data
            users_df: User profile data (optional)
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering for {len(transactions_df)} transactions")
        
        # Make a copy to avoid modifying original data
        df = transactions_df.copy()
        
        # Ensure required columns exist
        df = self._validate_and_prepare_data(df)
        
        # 1. Transaction-level features
        df = self._add_transaction_level_features(df)
        
        # 2. Temporal features
        df = self._add_temporal_features(df)
        
        # 3. Amount-based features
        df = self._add_amount_features(df)
        
        # 4. User behavior features
        df = self._add_user_behavior_features(df)
        
        # 5. Merchant/category features
        df = self._add_merchant_features(df)
        
        # 6. Geographic features (if location data available)
        if 'location' in df.columns or ('lat' in df.columns and 'lon' in df.columns):
            df = self._add_geographic_features(df)
        
        # 7. Device/channel features
        if 'device_type' in df.columns:
            df = self._add_device_features(df)
        
        # 8. Advanced features (if enabled)
        if self.enable_advanced_features:
            df = self._add_aggregation_features(df)
            df = self._add_interaction_features(df)
            df = self._add_anomaly_detection_features(df)
        
        # 9. Velocity features
        df = self._add_velocity_features(df)
        
        # 10. Risk scoring features
        df = self._add_risk_scoring_features(df)
        
        logger.info(f"Feature engineering completed. Generated {len(df.columns)} total features")
        return df
    
    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare the input data."""
        required_columns = ['transaction_id', 'user_id', 'amount', 'timestamp']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # Create dummy columns if missing
            for col in missing_columns:
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                elif col in ['amount']:
                    df[col] = 100.0
                else:
                    df[col] = 'unknown'
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp for velocity calculations
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        return df
    
    def _add_transaction_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic transaction-level features."""
        logger.debug("Adding transaction-level features")
        
        # Transaction amount features
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_sqrt'] = np.sqrt(df['amount'])
            
            # Amount percentile ranks
            df['amount_percentile'] = df['amount'].rank(pct=True)
            
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from timestamp."""
        logger.debug("Adding temporal features")
        
        if 'timestamp' not in df.columns:
            return df
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        
        # Time-based indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add amount-based features and transformations."""
        logger.debug("Adding amount features")
        
        if 'amount' not in df.columns:
            return df
        
        # Amount statistics
        amount_stats = df['amount'].describe()
        df['amount_zscore'] = (df['amount'] - amount_stats['mean']) / amount_stats['std']
        
        # Amount buckets
        amount_quantiles = df['amount'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        df['amount_bucket'] = pd.cut(df['amount'], 
                                   bins=[0] + amount_quantiles.tolist() + [float('inf')],
                                   labels=['very_small', 'small', 'medium', 'large', 'very_large', 'extreme'])
        
        # Round amount features (psychological pricing)
        df['amount_is_round'] = (df['amount'] % 1 == 0).astype(int)
        df['amount_is_round_10'] = (df['amount'] % 10 == 0).astype(int)
        df['amount_is_round_100'] = (df['amount'] % 100 == 0).astype(int)
        
        return df
    
    def _add_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user behavior and historical pattern features."""
        logger.debug("Adding user behavior features")
        
        if 'user_id' not in df.columns:
            return df
        
        # User transaction statistics
        user_stats = df.groupby('user_id').agg({
            'amount': ['count', 'mean', 'std', 'min', 'max', 'sum'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.add_prefix('user_')
        
        # Merge back to main dataframe
        df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        # User deviation features
        if 'amount' in df.columns:
            df['amount_vs_user_mean'] = df['amount'] / (df['user_amount_mean'] + 1e-6)
            df['amount_deviation_from_user'] = abs(df['amount'] - df['user_amount_mean'])
            df['amount_zscore_user'] = ((df['amount'] - df['user_amount_mean']) / 
                                      (df['user_amount_std'] + 1e-6))
        
        # User account age
        if 'timestamp' in df.columns:
            df['user_account_age_days'] = (df['timestamp'] - df['user_timestamp_min']).dt.days
            df['user_days_since_last'] = (df['user_timestamp_max'] - df['timestamp']).dt.days
        
        return df
    
    def _add_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add merchant and category-based features."""
        logger.debug("Adding merchant features")
        
        # Merchant statistics (if merchant_id exists)
        if 'merchant_id' in df.columns:
            merchant_stats = df.groupby('merchant_id').agg({
                'amount': ['count', 'mean', 'std', 'sum'],
                'user_id': 'nunique'
            }).round(2)
            
            merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
            merchant_stats = merchant_stats.add_prefix('merchant_')
            
            df = df.merge(merchant_stats, left_on='merchant_id', right_index=True, how='left')
            
            # Merchant risk indicators
            df['merchant_popularity'] = df['merchant_user_id_nunique']
            df['amount_vs_merchant_mean'] = df['amount'] / (df['merchant_amount_mean'] + 1e-6)
        
        # Category statistics (if category exists)
        if 'category' in df.columns:
            category_stats = df.groupby('category').agg({
                'amount': ['count', 'mean', 'std'],
                'user_id': 'nunique'
            }).round(2)
            
            category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
            category_stats = category_stats.add_prefix('category_')
            
            df = df.merge(category_stats, left_on='category', right_index=True, how='left')
        
        return df
    
    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geographic and location-based features."""
        logger.debug("Adding geographic features")
        
        # If lat/lon coordinates are available
        if 'lat' in df.columns and 'lon' in df.columns:
            # User's home location (most frequent location)
            user_locations = df.groupby('user_id')[['lat', 'lon']].agg('mean')
            user_locations.columns = ['user_home_lat', 'user_home_lon']
            
            df = df.merge(user_locations, left_on='user_id', right_index=True, how='left')
            
            # Distance from home
            df['distance_from_home'] = self._haversine_distance(
                df['lat'], df['lon'], 
                df['user_home_lat'], df['user_home_lon']
            )
            
        # Location frequency features
        if 'location' in df.columns:
            location_stats = df.groupby('location').size().reset_index(name='location_frequency')
            df = df.merge(location_stats, on='location', how='left')
            
            # User location diversity
            user_location_counts = df.groupby('user_id')['location'].nunique().reset_index()
            user_location_counts.columns = ['user_id', 'user_location_diversity']
            df = df.merge(user_location_counts, on='user_id', how='left')
        
        return df
    
    def _add_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add device and channel-based features."""
        logger.debug("Adding device features")
        
        if 'device_type' not in df.columns:
            return df
        
        # Device usage statistics
        device_stats = df.groupby('device_type').agg({
            'amount': ['count', 'mean'],
            'user_id': 'nunique'
        }).round(2)
        
        device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns]
        device_stats = device_stats.add_prefix('device_')
        
        df = df.merge(device_stats, left_on='device_type', right_index=True, how='left')
        
        # User device consistency
        user_devices = df.groupby('user_id')['device_type'].nunique().reset_index()
        user_devices.columns = ['user_id', 'user_device_diversity']
        df = df.merge(user_devices, on='user_id', how='left')
        
        return df
    
    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity-based features (transaction frequency and patterns)."""
        logger.debug("Adding velocity features")
        
        if 'timestamp' not in df.columns or 'user_id' not in df.columns:
            return df
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Time since last transaction
        df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        
        # Rolling window velocity features (last 1h, 6h, 24h)
        time_windows = {
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '24h': timedelta(hours=24)
        }
        
        for window_name, window_size in time_windows.items():
            # Count transactions in window
            df[f'transaction_count_{window_name}'] = df.groupby('user_id').apply(
                lambda x: x.set_index('timestamp').rolling(window_size)['amount'].count()
            ).values
            
            # Sum amount in window
            df[f'amount_sum_{window_name}'] = df.groupby('user_id').apply(
                lambda x: x.set_index('timestamp').rolling(window_size)['amount'].sum()
            ).values
            
            # Unique merchants in window (if available)
            if 'merchant_id' in df.columns:
                df[f'unique_merchants_{window_name}'] = df.groupby('user_id').apply(
                    lambda x: x.set_index('timestamp').rolling(window_size)['merchant_id'].nunique()
                ).values
        
        return df
    
    def _add_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced aggregation features."""
        logger.debug("Adding aggregation features")
        
        if 'amount' not in df.columns:
            return df
        
        # Rolling statistics for amounts
        for window in [3, 5, 10]:
            df[f'amount_rolling_mean_{window}'] = df.groupby('user_id')['amount'].rolling(window, min_periods=1).mean().values
            df[f'amount_rolling_std_{window}'] = df.groupby('user_id')['amount'].rolling(window, min_periods=1).std().values
            df[f'amount_rolling_max_{window}'] = df.groupby('user_id')['amount'].rolling(window, min_periods=1).max().values
            df[f'amount_rolling_min_{window}'] = df.groupby('user_id')['amount'].rolling(window, min_periods=1).min().values
        
        # Percentile features
        for percentile in [25, 50, 75, 90, 95]:
            df[f'amount_percentile_{percentile}'] = df.groupby('user_id')['amount'].rolling(10, min_periods=1).quantile(percentile/100).values
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different variables."""
        logger.debug("Adding interaction features")
        
        # Amount and time interactions
        if 'amount' in df.columns and 'hour' in df.columns:
            df['amount_hour_interaction'] = df['amount'] * df['hour']
            df['amount_weekend_interaction'] = df['amount'] * df['is_weekend']
        
        # User and merchant interactions
        if 'user_amount_mean' in df.columns and 'merchant_amount_mean' in df.columns:
            df['user_merchant_amount_ratio'] = df['user_amount_mean'] / (df['merchant_amount_mean'] + 1e-6)
        
        return df
    
    def _add_anomaly_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly detection scores."""
        logger.debug("Adding anomaly detection features")
        
        # Select numerical features for anomaly detection
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variables and IDs
        exclude_features = ['is_fraud', 'fraud_reason', 'transaction_id', 'user_id']
        numerical_features = [col for col in numerical_features if col not in exclude_features]
        
        if len(numerical_features) < 2:
            return df
        
        # Isolation Forest for amount-based anomalies
        if 'amount' in df.columns:
            try:
                iso_forest_amount = IsolationForest(contamination=0.1, random_state=42)
                df['amount_anomaly_score'] = iso_forest_amount.fit_predict(df[['amount']].fillna(0))
            except Exception as e:
                logger.warning(f"Could not compute amount anomaly score: {e}")
                df['amount_anomaly_score'] = 0
        
        # Statistical outliers (Z-score based)
        for feature in ['amount', 'time_since_last_transaction']:
            if feature in df.columns:
                feature_mean = df[feature].mean()
                feature_std = df[feature].std()
                if feature_std > 0:
                    df[f'{feature}_is_outlier'] = (abs(df[feature] - feature_mean) > 3 * feature_std).astype(int)
        
        return df
    
    def _add_risk_scoring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk scoring features."""
        logger.debug("Adding risk scoring features")
        
        # Fraud rate by category (if fraud labels available)
        if 'is_fraud' in df.columns:
            # Merchant fraud rate
            if 'merchant_id' in df.columns:
                merchant_fraud_rate = df.groupby('merchant_id')['is_fraud'].mean()
                df = df.merge(merchant_fraud_rate.rename('merchant_fraud_rate'), 
                            left_on='merchant_id', right_index=True, how='left')
            
            # Category fraud rate
            if 'category' in df.columns:
                category_fraud_rate = df.groupby('category')['is_fraud'].mean()
                df = df.merge(category_fraud_rate.rename('category_fraud_rate'), 
                            left_on='category', right_index=True, how='left')
            
            # Hour fraud rate
            if 'hour' in df.columns:
                hour_fraud_rate = df.groupby('hour')['is_fraud'].mean()
                df = df.merge(hour_fraud_rate.rename('hour_fraud_rate'), 
                            left_on='hour', right_index=True, how='left')
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1: pd.Series, lon1: pd.Series, 
                          lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """Calculate haversine distance between coordinates."""
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * 
             np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of generated features."""
        feature_categories = {
            'transaction_level': [col for col in df.columns if any(x in col for x in ['amount', 'log', 'sqrt', 'percentile'])],
            'temporal': [col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'weekend', 'night', 'sin', 'cos'])],
            'user_behavior': [col for col in df.columns if col.startswith('user_')],
            'merchant': [col for col in df.columns if col.startswith('merchant_') or col.startswith('category_')],
            'geographic': [col for col in df.columns if any(x in col for x in ['location', 'distance', 'lat', 'lon'])],
            'device': [col for col in df.columns if col.startswith('device_')],
            'velocity': [col for col in df.columns if any(x in col for x in ['time_since', 'count_', 'sum_', 'rolling'])],
            'risk_scoring': [col for col in df.columns if 'fraud_rate' in col],
            'anomaly': [col for col in df.columns if any(x in col for x in ['anomaly', 'outlier'])],
            'interaction': [col for col in df.columns if 'interaction' in col or 'ratio' in col]
        }
        
        summary = {
            'total_features': len(df.columns),
            'feature_categories': {k: len(v) for k, v in feature_categories.items()},
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().sum()
        }
        
        return summary


# Convenience function for quick feature engineering
def engineer_fraud_features(transactions_df: pd.DataFrame,
                          users_df: Optional[pd.DataFrame] = None,
                          enable_advanced: bool = True) -> pd.DataFrame:
    """
    Convenience function to perform complete feature engineering.
    
    Args:
        transactions_df: Transaction data
        users_df: User data (optional)
        enable_advanced: Whether to compute advanced features
        
    Returns:
        DataFrame with engineered features
    """
    feature_engineer = FeatureEngineer(enable_advanced_features=enable_advanced)
    return feature_engineer.engineer_features(transactions_df, users_df)


if __name__ == "__main__":
    # Example usage and testing
    print("Feature Engineering Pipeline for Fraud Detection")
    print("=" * 50)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'transaction_id': range(1000),
        'user_id': np.random.randint(1, 100, 1000),
        'amount': np.random.lognormal(4, 1, 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'merchant_id': np.random.randint(1, 50, 1000),
        'category': np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail'], 1000),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    
    # Test feature engineering
    feature_engineer = FeatureEngineer()
    engineered_df = feature_engineer.engineer_features(sample_data)
    
    # Print summary
    summary = feature_engineer.get_feature_summary(engineered_df)
    print(f"Generated {summary['total_features']} total features:")
    for category, count in summary['feature_categories'].items():
        if count > 0:
            print(f"  - {category}: {count} features")
    
    print(f"\nFeature engineering completed successfully!")
    print(f"Original features: {len(sample_data.columns)}")
    print(f"Engineered features: {len(engineered_df.columns)}")