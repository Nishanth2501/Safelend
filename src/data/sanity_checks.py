#!/usr/bin/env python3
"""
Sanity checks for SafeLend processed datasets.
Performs comprehensive data quality checks on train_ready.parquet and test_ready.parquet.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def print_basic_info(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print basic dataset information."""
    print("\n" + "="*60)
    print("BASIC DATASET INFORMATION")
    print("="*60)
    
    print(f"Train dataset: {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"Test dataset:  {test.shape[0]:,} rows × {test.shape[1]} columns")
    
    # Check if test has one fewer column (no TARGET)
    col_diff = train.shape[1] - test.shape[1]
    if col_diff == 1:
        print("✓ Test has 1 fewer column than train (expected: no TARGET column)")
    else:
        print(f"⚠ Warning: Column count difference is {col_diff} (expected: 1)")


def analyze_target_distribution(train: pd.DataFrame) -> None:
    """Analyze TARGET variable distribution."""
    print("\n" + "="*60)
    print("TARGET DISTRIBUTION (Train Only)")
    print("="*60)
    
    if "TARGET" not in train.columns:
        print("❌ ERROR: TARGET column not found in train dataset!")
        return
    
    target_counts = train["TARGET"].value_counts().sort_index()
    target_pct = train["TARGET"].value_counts(normalize=True).sort_index() * 100
    
    print("Counts:")
    for value, count in target_counts.items():
        pct = target_pct[value]
        print(f"  {value}: {count:,} ({pct:.2f}%)")
    
    # Check for class imbalance
    min_pct = target_pct.min()
    if min_pct < 5:
        print(f"⚠ Warning: Severe class imbalance detected (minority class: {min_pct:.2f}%)")
    elif min_pct < 20:
        print(f"⚠ Note: Moderate class imbalance (minority class: {min_pct:.2f}%)")
    else:
        print("✓ Balanced dataset")


def find_missing_values(df: pd.DataFrame, dataset_name: str) -> None:
    """Find and display columns with highest missing value percentages."""
    print(f"\n" + "="*60)
    print(f"MISSING VALUES - {dataset_name.upper()}")
    print("="*60)
    
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]  # Only show columns with missing values
    
    if len(missing_pct) == 0:
        print("✓ No missing values found!")
        return
    
    print("Top 15 columns with highest % missing values:")
    for i, (col, pct) in enumerate(missing_pct.head(15).items(), 1):
        print(f"  {i:2d}. {col:<30} {pct:6.2f}%")
    
    if len(missing_pct) > 15:
        print(f"  ... and {len(missing_pct) - 15} more columns with missing values")


def find_constant_columns(df: pd.DataFrame, dataset_name: str) -> None:
    """Find constant columns (single unique value)."""
    print(f"\n" + "="*60)
    print(f"CONSTANT COLUMNS - {dataset_name.upper()}")
    print("="*60)
    
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Found {len(constant_cols)} constant columns:")
        for col in constant_cols:
            unique_val = df[col].iloc[0] if len(df) > 0 else "N/A"
            print(f"  - {col}: '{unique_val}'")
        print("⚠ Consider removing these columns for modeling")
    else:
        print("✓ No constant columns found")


def analyze_key_features(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Analyze key feature statistics."""
    print("\n" + "="*60)
    print("KEY FEATURE STATISTICS")
    print("="*60)
    
    key_features = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "APP_CREDIT_TO_INCOME", "APP_AGE_YEARS"]
    
    for feature in key_features:
        if feature in train.columns:
            print(f"\n{feature}:")
            train_stats = train[feature].describe()
            print(f"  Train - Mean: {train_stats['mean']:.2f}, Std: {train_stats['std']:.2f}, "
                  f"Min: {train_stats['min']:.2f}, Max: {train_stats['max']:.2f}")
            
            if feature in test.columns:
                test_stats = test[feature].describe()
                print(f"  Test  - Mean: {test_stats['mean']:.2f}, Std: {test_stats['std']:.2f}, "
                      f"Min: {test_stats['min']:.2f}, Max: {test_stats['max']:.2f}")
                
                # Check for distribution shift
                mean_diff = abs(train_stats['mean'] - test_stats['mean'])
                if mean_diff > train_stats['std'] * 0.5:  # More than 0.5 std devs difference
                    print(f"  ⚠ Warning: Significant distribution shift detected (mean diff: {mean_diff:.2f})")
            else:
                print(f"  ❌ Feature missing in test set")
        else:
            print(f"\n{feature}: ❌ Not found in train set")


def check_target_presence(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check TARGET column presence."""
    print("\n" + "="*60)
    print("TARGET COLUMN VERIFICATION")
    print("="*60)
    
    train_has_target = "TARGET" in train.columns
    test_has_target = "TARGET" in test.columns
    
    if train_has_target and not test_has_target:
        print("✓ Correct: TARGET present in train, absent in test")
    elif train_has_target and test_has_target:
        print("❌ ERROR: TARGET should not be present in test set!")
    elif not train_has_target and not test_has_target:
        print("❌ ERROR: TARGET missing from train set!")
    else:
        print("❌ ERROR: Unexpected TARGET column configuration!")


def generate_report(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Generate and save a text report."""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    report_path = Path("data/processed/sanity_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("SafeLend Dataset Sanity Check Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic info
        f.write(f"Train dataset: {train.shape[0]:,} rows × {train.shape[1]} columns\n")
        f.write(f"Test dataset:  {test.shape[0]:,} rows × {test.shape[1]} columns\n\n")
        
        # Target distribution
        if "TARGET" in train.columns:
            target_counts = train["TARGET"].value_counts().sort_index()
            target_pct = train["TARGET"].value_counts(normalize=True).sort_index() * 100
            f.write("TARGET Distribution:\n")
            for value, count in target_counts.items():
                pct = target_pct[value]
                f.write(f"  {value}: {count:,} ({pct:.2f}%)\n")
            f.write("\n")
        
        # Missing values
        train_missing = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
        train_missing = train_missing[train_missing > 0]
        f.write(f"Train missing values: {len(train_missing)} columns have missing values\n")
        if len(train_missing) > 0:
            f.write("Top 10 missing value columns:\n")
            for col, pct in train_missing.head(10).items():
                f.write(f"  {col}: {pct:.2f}%\n")
        
        test_missing = (test.isnull().sum() / len(test) * 100).sort_values(ascending=False)
        test_missing = test_missing[test_missing > 0]
        f.write(f"\nTest missing values: {len(test_missing)} columns have missing values\n")
        if len(test_missing) > 0:
            f.write("Top 10 missing value columns:\n")
            for col, pct in test_missing.head(10).items():
                f.write(f"  {col}: {pct:.2f}%\n")
        
        # Constant columns
        train_constants = [col for col in train.columns if train[col].nunique() <= 1]
        test_constants = [col for col in test.columns if test[col].nunique() <= 1]
        f.write(f"\nConstant columns - Train: {len(train_constants)}, Test: {len(test_constants)}\n")
        
        # Key features
        f.write("\nKey Feature Summary:\n")
        key_features = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "APP_CREDIT_TO_INCOME", "APP_AGE_YEARS"]
        for feature in key_features:
            if feature in train.columns:
                train_mean = train[feature].mean()
                train_std = train[feature].std()
                f.write(f"  {feature} - Train mean: {train_mean:.2f}, std: {train_std:.2f}\n")
                if feature in test.columns:
                    test_mean = test[feature].mean()
                    f.write(f"    Test mean: {test_mean:.2f}\n")
    
    print(f"✓ Report saved to: {report_path}")


def main():
    """Main function to run all sanity checks."""
    print("SafeLend Dataset Sanity Checks")
    print("=" * 50)
    
    try:
        # Load datasets
        train, test = load_datasets()
        
        # Run all checks
        print_basic_info(train, test)
        analyze_target_distribution(train)
        find_missing_values(train, "Train")
        find_missing_values(test, "Test")
        find_constant_columns(train, "Train")
        find_constant_columns(test, "Test")
        analyze_key_features(train, test)
        check_target_presence(train, test)
        generate_report(train, test)
        
        print("\n" + "="*60)
        print("SANITY CHECKS COMPLETED")
        print("="*60)
        print("✓ All checks completed successfully!")
        print("✓ Report saved to data/processed/sanity_report.txt")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
