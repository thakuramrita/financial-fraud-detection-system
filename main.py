"""
Financial Fraud Detection System - Main Entry Point
Hybrid approach combining traditional ML models with LLM/Agent-based AI
"""

import sys
import os
import argparse

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_generator import FraudDataGenerator, logger
from src.features.feature_generator import generate_features
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(run_feature_engineering: bool = False):
    """Main function to run the fraud detection system."""
    logger.info("Starting Financial Fraud Detection System...")
    
    try:
        # Initialize data generator
        generator = FraudDataGenerator(seed=42)
        
        # Generate dataset
        logger.info("Generating synthetic fraud detection dataset...")
        users_df, transactions_df, features_df = generator.generate_dataset(
            num_users=10000,
            num_transactions=100000
        )
        
        # Save dataset (use the original transactions DataFrame, not features)
        generator.save_dataset(users_df, transactions_df, "data")
        
        logger.info("Dataset generation completed!")
        logger.info(f"Generated {len(users_df)} users and {len(transactions_df)} transactions")
        
        # Print sample statistics
        fraud_rate = transactions_df['is_fraud'].mean()
        logger.info(f"Overall fraud rate: {fraud_rate:.2%}")
        
        # Optionally run feature engineering
        if run_feature_engineering:
            logger.info("Running feature engineering...")
            try:
                features_df = generate_features(transactions_df, logger)
                
                # Save engineered features
                os.makedirs("data", exist_ok=True)
                features_df.to_csv("data/features.csv", index=False)
                logger.info(f"Generated {len(features_df.columns)} features and saved to data/features.csv")
                
                print(f"\n✅ Feature Engineering Completed!")
                print(f"Generated {len(features_df.columns)} features from {len(features_df)} transactions")
                
            except Exception as e:
                logger.error(f"Feature engineering failed: {e}")
                print(f"❌ Feature engineering failed: {e}")
        
        print("\n" + "="*50)
        print("FRAUD DETECTION SYSTEM - DATASET GENERATED")
        print("="*50)
        print(f"Users: {len(users_df):,}")
        print(f"Transactions: {len(transactions_df):,}")
        print(f"Fraud Rate: {fraud_rate:.2%}")
        print("\nFraud Reasons Distribution:")
        print(transactions_df['fraud_reason'].value_counts())
        
        if not run_feature_engineering:
            print("\nNext steps:")
            print("1. Run feature engineering: python src/features/feature_generator.py --input data/transactions.csv --output data/features.csv")
            print("   Or run: python main.py --features")
            print("2. Train ML model: python src/models/train_model.py")
            print("3. Start API server: python src/api/main.py")
        else:
            print("\nCompleted steps:")
            print("✅ 1. Data generation")
            print("✅ 2. Feature engineering")
            print("\nNext steps:")
            print("3. Train ML model: python src/models/train_model.py")
            print("4. Start API server: python src/api/main.py")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"❌ Data generator not available. Error: {e}")
        print("Please ensure the data generator module is properly set up.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Fraud Detection System")
    parser.add_argument('--features', action='store_true', 
                       help='Run feature engineering after data generation')
    args = parser.parse_args()
    
    main(run_feature_engineering=args.features)