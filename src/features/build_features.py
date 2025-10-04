#!/usr/bin/env python3
"""
Centralized feature engineering module for SafeLend project.
Contains all feature creation, transformation, and engineering logic.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SafeLendFeatureEngineer:
    """Centralized feature engineering for SafeLend credit risk assessment."""
    
    def __init__(self):
        self.SENTINEL_DAY = 365243
        self.feature_groups = {
            'application_ratios': [],
            'temporal_features': [],
            'categorical_encodings': [],
            'aggregated_features': [],
            'interaction_features': []
        }
    
    def create_application_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features from application data."""
        print("Creating application ratio features...")
        
        eps = 1e-6
        df = df.copy()
        
        # Credit-to-Income ratio
        if {'AMT_CREDIT', 'AMT_INCOME_TOTAL'}.issubset(df.columns):
            df['APP_CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + eps)
            self.feature_groups['application_ratios'].append('APP_CREDIT_TO_INCOME')
        
        # Annuity-to-Income ratio
        if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(df.columns):
            df['APP_ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + eps)
            self.feature_groups['application_ratios'].append('APP_ANNUITY_TO_INCOME')
        
        # Annuity-to-Credit ratio
        if {'AMT_ANNUITY', 'AMT_CREDIT'}.issubset(df.columns):
            df['APP_ANNUITY_TO_CREDIT'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + eps)
            self.feature_groups['application_ratios'].append('APP_ANNUITY_TO_CREDIT')
        
        # Goods Price-to-Credit ratio
        if {'AMT_GOODS_PRICE', 'AMT_CREDIT'}.issubset(df.columns):
            df['APP_GOODS_TO_CREDIT'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + eps)
            self.feature_groups['application_ratios'].append('APP_GOODS_TO_CREDIT')
        
        # Income per family member
        if {'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS'}.issubset(df.columns):
            df['APP_INCOME_PER_FAMILY'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + eps)
            self.feature_groups['application_ratios'].append('APP_INCOME_PER_FAMILY')
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date-related columns."""
        print("Creating temporal features...")
        
        df = df.copy()
        
        # Age in years
        if 'DAYS_BIRTH' in df.columns:
            df['APP_AGE_YEARS'] = (-df['DAYS_BIRTH']).clip(lower=0) / 365.0
            self.feature_groups['temporal_features'].append('APP_AGE_YEARS')
        
        # Employment years
        if 'DAYS_EMPLOYED' in df.columns:
            df['APP_EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED']).clip(lower=0) / 365.0
            self.feature_groups['temporal_features'].append('APP_EMPLOYED_YEARS')
        
        # Registration days
        if 'DAYS_REGISTRATION' in df.columns:
            df['APP_REGISTRATION_YEARS'] = df['DAYS_REGISTRATION'] / 365.0
            self.feature_groups['temporal_features'].append('APP_REGISTRATION_YEARS')
        
        # ID publish days
        if 'DAYS_ID_PUBLISH' in df.columns:
            df['APP_ID_PUBLISH_YEARS'] = df['DAYS_ID_PUBLISH'] / 365.0
            self.feature_groups['temporal_features'].append('APP_ID_PUBLISH_YEARS')
        
        # Car age
        if 'OWN_CAR_AGE' in df.columns:
            df['APP_CAR_AGE_YEARS'] = df['OWN_CAR_AGE'].clip(lower=0)
            self.feature_groups['temporal_features'].append('APP_CAR_AGE_YEARS')
        
        # Employment ratio (employed years / age)
        if {'APP_EMPLOYED_YEARS', 'APP_AGE_YEARS'}.issubset(df.columns):
            df['APP_EMPLOYMENT_RATIO'] = df['APP_EMPLOYED_YEARS'] / (df['APP_AGE_YEARS'] + 1e-6)
            self.feature_groups['temporal_features'].append('APP_EMPLOYMENT_RATIO')
        
        return df
    
    def create_document_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to document counts and flags."""
        print("Creating document features...")
        
        df = df.copy()
        
        # Total document count
        flag_doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if flag_doc_cols:
            df['APP_DOC_COUNT'] = df[flag_doc_cols].sum(axis=1)
            self.feature_groups['categorical_encodings'].append('APP_DOC_COUNT')
        
        # Phone and email flags
        if 'FLAG_PHONE' in df.columns:
            df['APP_HAS_PHONE'] = df['FLAG_PHONE'].astype(int)
            self.feature_groups['categorical_encodings'].append('APP_HAS_PHONE')
        
        if 'FLAG_EMAIL' in df.columns:
            df['APP_HAS_EMAIL'] = df['FLAG_EMAIL'].astype(int)
            self.feature_groups['categorical_encodings'].append('APP_HAS_EMAIL')
        
        # Own car and realty flags
        if 'FLAG_OWN_CAR' in df.columns:
            df['APP_OWNS_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
            self.feature_groups['categorical_encodings'].append('APP_OWNS_CAR')
        
        if 'FLAG_OWN_REALTY' in df.columns:
            df['APP_OWNS_REALTY'] = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
            self.feature_groups['categorical_encodings'].append('APP_OWNS_REALTY')
        
        return df
    
    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create family-related features."""
        print("Creating family features...")
        
        df = df.copy()
        
        # Children ratio
        if {'CNT_CHILDREN', 'CNT_FAM_MEMBERS'}.issubset(df.columns):
            df['APP_CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1e-6)
            self.feature_groups['categorical_encodings'].append('APP_CHILDREN_RATIO')
        
        # Income per child
        if {'AMT_INCOME_TOTAL', 'CNT_CHILDREN'}.issubset(df.columns):
            df['APP_INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
            self.feature_groups['categorical_encodings'].append('APP_INCOME_PER_CHILD')
        
        # Has children flag
        if 'CNT_CHILDREN' in df.columns:
            df['APP_HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
            self.feature_groups['categorical_encodings'].append('APP_HAS_CHILDREN')
        
        return df
    
    def create_income_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create income-related features."""
        print("Creating income features...")
        
        df = df.copy()
        
        # Income categories
        if 'AMT_INCOME_TOTAL' in df.columns:
            # Log income for better distribution
            df['APP_LOG_INCOME'] = np.log1p(df['AMT_INCOME_TOTAL'])
            self.feature_groups['application_ratios'].append('APP_LOG_INCOME')
            
            # Income percentiles
            df['APP_INCOME_PERCENTILE'] = df['AMT_INCOME_TOTAL'].rank(pct=True)
            self.feature_groups['application_ratios'].append('APP_INCOME_PERCENTILE')
            
            # Income bins
            income_bins = [0, 100000, 200000, 300000, float('inf')]
            income_labels = ['Low', 'Medium', 'High', 'Very High']
            df['APP_INCOME_BIN'] = pd.cut(df['AMT_INCOME_TOTAL'], bins=income_bins, labels=income_labels)
            self.feature_groups['categorical_encodings'].append('APP_INCOME_BIN')
        
        return df
    
    def create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create credit-related features."""
        print("Creating credit features...")
        
        df = df.copy()
        
        # Credit amount features
        if 'AMT_CREDIT' in df.columns:
            # Log credit for better distribution
            df['APP_LOG_CREDIT'] = np.log1p(df['AMT_CREDIT'])
            self.feature_groups['application_ratios'].append('APP_LOG_CREDIT')
            
            # Credit percentiles
            df['APP_CREDIT_PERCENTILE'] = df['AMT_CREDIT'].rank(pct=True)
            self.feature_groups['application_ratios'].append('APP_CREDIT_PERCENTILE')
        
        # Annuity features
        if 'AMT_ANNUITY' in df.columns:
            df['APP_LOG_ANNUITY'] = np.log1p(df['AMT_ANNUITY'])
            self.feature_groups['application_ratios'].append('APP_LOG_ANNUITY')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        print("Creating interaction features...")
        
        df = df.copy()
        
        # Age * Credit interaction
        if {'APP_AGE_YEARS', 'AMT_CREDIT'}.issubset(df.columns):
            df['APP_AGE_CREDIT_INTERACTION'] = df['APP_AGE_YEARS'] * df['AMT_CREDIT']
            self.feature_groups['interaction_features'].append('APP_AGE_CREDIT_INTERACTION')
        
        # Income * Employment interaction
        if {'AMT_INCOME_TOTAL', 'APP_EMPLOYED_YEARS'}.issubset(df.columns):
            df['APP_INCOME_EMPLOYMENT_INTERACTION'] = df['AMT_INCOME_TOTAL'] * df['APP_EMPLOYED_YEARS']
            self.feature_groups['interaction_features'].append('APP_INCOME_EMPLOYMENT_INTERACTION')
        
        # Credit * Family size interaction
        if {'AMT_CREDIT', 'CNT_FAM_MEMBERS'}.issubset(df.columns):
            df['APP_CREDIT_FAMILY_INTERACTION'] = df['AMT_CREDIT'] * df['CNT_FAM_MEMBERS']
            self.feature_groups['interaction_features'].append('APP_CREDIT_FAMILY_INTERACTION')
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame, agg_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Integrate aggregated features from other tables."""
        print("Integrating aggregated features...")
        
        df = df.copy()
        
        # Bureau aggregated features
        if 'bureau_agg' in agg_features:
            bureau_df = agg_features['bureau_agg']
            df = df.merge(bureau_df, on='SK_ID_CURR', how='left', suffixes=('', '_BUR'))
            self.feature_groups['aggregated_features'].extend([col for col in bureau_df.columns if col != 'SK_ID_CURR'])
        
        # Previous application aggregated features
        if 'previous_agg' in agg_features:
            prev_df = agg_features['previous_agg']
            df = df.merge(prev_df, on='SK_ID_CURR', how='left', suffixes=('', '_PREV'))
            self.feature_groups['aggregated_features'].extend([col for col in prev_df.columns if col != 'SK_ID_CURR'])
        
        # Credit card aggregated features
        if 'credit_card_agg' in agg_features:
            cc_df = agg_features['credit_card_agg']
            df = df.merge(cc_df, on='SK_ID_CURR', how='left', suffixes=('', '_CC'))
            self.feature_groups['aggregated_features'].extend([col for col in cc_df.columns if col != 'SK_ID_CURR'])
        
        # POS aggregated features
        if 'pos_agg' in agg_features:
            pos_df = agg_features['pos_agg']
            df = df.merge(pos_df, on='SK_ID_CURR', how='left', suffixes=('', '_POS'))
            self.feature_groups['aggregated_features'].extend([col for col in pos_df.columns if col != 'SK_ID_CURR'])
        
        # Installments aggregated features
        if 'installments_agg' in agg_features:
            inst_df = agg_features['installments_agg']
            df = df.merge(inst_df, on='SK_ID_CURR', how='left', suffixes=('', '_INST'))
            self.feature_groups['aggregated_features'].extend([col for col in inst_df.columns if col != 'SK_ID_CURR'])
        
        # Bureau balance aggregated features
        if 'bureau_balance_agg' in agg_features:
            bb_df = agg_features['bureau_balance_agg']
            df = df.merge(bb_df, on='SK_ID_CURR', how='left', suffixes=('', '_BB'))
            self.feature_groups['aggregated_features'].extend([col for col in bb_df.columns if col != 'SK_ID_CURR'])
        
        return df
    
    def handle_categorical_encoding(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Handle categorical variable encoding."""
        print("Handling categorical encoding...")
        
        df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                # Label encoding for tree models
                df[col] = pd.Categorical(df[col]).codes
                self.feature_groups['categorical_encodings'].append(col)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, agg_features: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Create all engineered features."""
        print("Creating all engineered features...")
        
        if agg_features is None:
            agg_features = {}
        
        # Apply all feature engineering steps
        df = self.create_application_ratios(df)
        df = self.create_temporal_features(df)
        df = self.create_document_features(df)
        df = self.create_family_features(df)
        df = self.create_income_features(df)
        df = self.create_credit_features(df)
        df = self.create_interaction_features(df)
        df = self.create_aggregated_features(df, agg_features)
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis."""
        return self.feature_groups
    
    def get_feature_summary(self) -> Dict[str, int]:
        """Return summary of features by group."""
        return {group: len(features) for group, features in self.feature_groups.items()}

def load_aggregated_features(interim_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all aggregated feature files."""
    agg_features = {}
    
    agg_files = [
        ('bureau_agg', 'bureau_agg.parquet'),
        ('previous_agg', 'previous_agg.parquet'),
        ('credit_card_agg', 'credit_card_agg.parquet'),
        ('pos_agg', 'pos_agg.parquet'),
        ('installments_agg', 'installments_agg.parquet'),
        ('bureau_balance_agg', 'bureau_balance_agg.parquet')
    ]
    
    for name, filename in agg_files:
        file_path = interim_dir / filename
        if file_path.exists():
            agg_features[name] = pd.read_parquet(file_path)
            print(f"Loaded {name}: {agg_features[name].shape}")
        else:
            print(f"Warning: {filename} not found")
    
    return agg_features

def main():
    """Main function to demonstrate feature engineering."""
    from pathlib import Path
    
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base application data
    train_df = pd.read_parquet(interim_dir / "application_train_clean.parquet")
    test_df = pd.read_parquet(interim_dir / "application_test_clean.parquet")
    
    # Load aggregated features
    agg_features = load_aggregated_features(interim_dir)
    
    # Initialize feature engineer
    engineer = SafeLendFeatureEngineer()
    
    # Create features for train and test
    train_features = engineer.create_all_features(train_df, agg_features)
    test_features = engineer.create_all_features(test_df, agg_features)
    
    # Handle categorical encoding
    categorical_cols = train_features.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['SK_ID_CURR', 'TARGET']]
    
    train_features = engineer.handle_categorical_encoding(train_features, categorical_cols)
    test_features = engineer.handle_categorical_encoding(test_features, categorical_cols)
    
    # Align columns
    train_cols = [col for col in train_features.columns if col != 'TARGET']
    test_cols = [col for col in test_features.columns if col != 'TARGET']
    
    # Ensure test has same columns as train (except TARGET)
    missing_in_test = [col for col in train_cols if col not in test_cols]
    for col in missing_in_test:
        test_features[col] = 0
    
    # Reorder test columns to match train
    test_features = test_features[train_cols + (['TARGET'] if 'TARGET' in test_features.columns else [])]
    
    # Save processed data
    train_features.to_parquet(processed_dir / "train_modeling.parquet", index=False)
    test_features.to_parquet(processed_dir / "test_modeling.parquet", index=False)
    
    # Print feature summary
    print("\nFeature Engineering Summary:")
    print("=" * 50)
    summary = engineer.get_feature_summary()
    for group, count in summary.items():
        print(f"{group}: {count} features")
    
    total_features = sum(summary.values())
    print(f"\nTotal engineered features: {total_features}")
    print(f"Final dataset shape - Train: {train_features.shape}, Test: {test_features.shape}")

if __name__ == "__main__":
    main()