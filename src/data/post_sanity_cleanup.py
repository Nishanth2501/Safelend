#!/usr/bin/env python3
"""
Post-sanity cleanup script for SafeLend datasets.
Removes constant columns and performs final data cleaning based on sanity check results.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    processed_dir = Path("data/processed")
    train_path = processed_dir / "train_ready.parquet"
    test_path = processed_dir / "test_ready.parquet"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    print("Loading datasets...")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    
    return train, test


def remove_constant_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove constant columns from both datasets."""
    print("\nRemoving constant columns...")
    
    # Find constant columns in test set (these are the ones we identified)
    test_constant_cols = []
    for col in test.columns:
        if test[col].nunique() <= 1:
            test_constant_cols.append(col)
    
    # Find constant columns in train set
    train_constant_cols = []
    for col in train.columns:
        if train[col].nunique() <= 1:
            train_constant_cols.append(col)
    
    # Remove constant columns from both datasets
    all_constant_cols = list(set(train_constant_cols + test_constant_cols))
    
    if all_constant_cols:
        print(f"Removing {len(all_constant_cols)} constant columns:")
        for col in sorted(all_constant_cols):
            print(f"  - {col}")
        
        train_clean = train.drop(columns=all_constant_cols)
        test_clean = test.drop(columns=all_constant_cols)
        
        print(f"Train: {train.shape[1]} → {train_clean.shape[1]} columns")
        print(f"Test:  {test.shape[1]} → {test_clean.shape[1]} columns")
    else:
        print("No constant columns found to remove.")
        train_clean = train.copy()
        test_clean = test.copy()
    
    return train_clean, test_clean


def remove_high_correlation_features(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove highly correlated features."""
    print(f"\nRemoving highly correlated features (threshold: {threshold})...")
    
    # Only consider numeric columns for correlation analysis
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if "TARGET" in numeric_cols:
        numeric_cols.remove("TARGET")
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return train, test
    
    # Calculate correlation matrix
    corr_matrix = train[numeric_cols].corr().abs()
    
    # Find pairs of highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if pd.notna(upper_tri.loc[idx, col]) and upper_tri.loc[idx, col] > threshold:
                high_corr_pairs.append((idx, col, upper_tri.loc[idx, col]))
    
    if high_corr_pairs:
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  - {feat1} ↔ {feat2}: {corr:.3f}")
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2, _ in high_corr_pairs:
            # Keep the feature that appears first in the column list
            if feat1 in numeric_cols and feat2 in numeric_cols:
                if numeric_cols.index(feat1) > numeric_cols.index(feat2):
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        print(f"Removing {len(features_to_remove)} features:")
        for feat in sorted(features_to_remove):
            print(f"  - {feat}")
        
        train_clean = train.drop(columns=features_to_remove)
        test_clean = test.drop(columns=features_to_remove)
        
        print(f"Train: {train.shape[1]} → {train_clean.shape[1]} columns")
        print(f"Test:  {test.shape[1]} → {test_clean.shape[1]} columns")
    else:
        print("No highly correlated features found.")
        train_clean = train.copy()
        test_clean = test.copy()
    
    return train_clean, test_clean


def save_cleaned_datasets(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save cleaned datasets."""
    print("\nSaving cleaned datasets...")
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned datasets
    train_path = processed_dir / "train_ready_clean.parquet"
    test_path = processed_dir / "test_ready_clean.parquet"
    
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    
    # Save sample CSVs
    train.head(2000).to_csv(processed_dir / "train_ready_clean_sample.csv", index=False)
    test.head(2000).to_csv(processed_dir / "test_ready_clean_sample.csv", index=False)
    
    print(f"✓ Saved cleaned datasets:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - Sample CSVs for quick inspection")
    
    print(f"\nFinal dataset shapes:")
    print(f"  Train: {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  Test:  {test.shape[0]:,} rows × {test.shape[1]} columns")


def main():
    """Main function to run post-sanity cleanup."""
    print("SafeLend Post-Sanity Cleanup")
    print("=" * 40)
    
    try:
        # Load datasets
        train, test = load_datasets()
        
        print(f"Original dataset shapes:")
        print(f"  Train: {train.shape[0]:,} rows × {train.shape[1]} columns")
        print(f"  Test:  {test.shape[0]:,} rows × {test.shape[1]} columns")
        
        # Remove constant columns
        train, test = remove_constant_columns(train, test)
        
        # Remove highly correlated features
        train, test = remove_high_correlation_features(train, test)
        
        # Save cleaned datasets
        save_cleaned_datasets(train, test)
        
        print("\n" + "=" * 40)
        print("POST-SANITY CLEANUP COMPLETED")
        print("=" * 40)
        print("✓ Datasets cleaned and saved successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
