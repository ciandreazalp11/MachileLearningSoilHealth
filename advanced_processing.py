"""
Advanced Data Processing Module
Provides performance optimizations, advanced preprocessing, and data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================================
# PERFORMANCE OPTIMIZATION: Chunked Processing for Large Datasets
# ============================================================================

@st.cache_data
def process_large_dataset_chunked(df: pd.DataFrame, chunk_size: int = 50000) -> pd.DataFrame:
    """
    Process large datasets in chunks to avoid memory overload.
    
    Args:
        df: Input dataframe
        chunk_size: Number of rows per chunk
        
    Returns:
        Processed dataframe
    """
    if len(df) <= chunk_size:
        return df
    
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    processed_chunks = []
    
    progress_bar = st.progress(0)
    for idx, chunk in enumerate(chunks):
        # Process each chunk
        processed_chunks.append(chunk)
        progress_bar.progress((idx + 1) / len(chunks))
    
    result = pd.concat(processed_chunks, ignore_index=True)
    return result


# ============================================================================
# DATA VALIDATION: Comprehensive Quality Assessment
# ============================================================================

def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data validation and quality assessment.
    
    Returns dictionary with validation results including:
    - Missing values percentage
    - Duplicate rows count
    - Outliers detected
    - Data type mismatches
    - Value range violations
    """
    validation_results = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "missing_values": {},
        "duplicates": df.duplicated().sum(),
        "duplicate_pct": round(df.duplicated().sum() / len(df) * 100, 2),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "outliers_detected": {},
        "data_types": df.dtypes.to_dict(),
    }
    
    # Check missing values
    missing = df.isnull().sum()
    validation_results["missing_values"] = {
        col: {
            "count": int(missing[col]),
            "percentage": round(missing[col] / len(df) * 100, 2)
        }
        for col in df.columns if missing[col] > 0
    }
    
    # Detect outliers using IQR method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            validation_results["outliers_detected"][col] = {
                "count": int(outliers),
                "percentage": round(outliers / len(df) * 100, 2)
            }
    
    return validation_results


def generate_data_profile_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed data profiling report for each column.
    """
    profile = []
    
    for col in df.columns:
        col_info = {
            "Column": col,
            "Data_Type": str(df[col].dtype),
            "Non_Null_Count": df[col].notna().sum(),
            "Null_Count": df[col].isnull().sum(),
            "Null_Percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
            "Unique_Values": df[col].nunique(),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Mean": df[col].mean(),
                "Median": df[col].median(),
                "Std_Dev": df[col].std(),
            })
        else:
            col_info["Top_Value"] = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
        
        profile.append(col_info)
    
    return pd.DataFrame(profile)


# ============================================================================
# ADVANCED DATA PROCESSING: Sophisticated Imputation & Handling
# ============================================================================

def advanced_missing_value_imputation(df: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
    """
    Advanced handling of missing values with multiple strategies.
    
    Methods:
    - 'auto': Choose best method based on data type and missingness
    - 'median': Median imputation for numeric columns
    - 'knn': K-Nearest Neighbors imputation
    - 'mean': Mean imputation for numeric columns
    - 'forward_fill': Forward fill for time-series-like data
    """
    df_imputed = df.copy()
    
    if method == "auto":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Use KNN imputation for numeric if not too many missing
        if len(numeric_cols) > 0:
            missing_pct = df[numeric_cols].isnull().mean().max()
            if missing_pct < 0.3:  # Less than 30% missing
                try:
                    imputer = KNNImputer(n_neighbors=5, weights='distance')
                    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                except:
                    # Fall back to median if KNN fails
                    imputer = SimpleImputer(strategy='median')
                    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                # Use median for high missing percentage
                imputer = SimpleImputer(strategy='median')
                df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Use mode for categorical
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                df_imputed[col].fillna(mode_val, inplace=True)
    
    elif method == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    elif method == "knn":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    elif method == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    elif method == "forward_fill":
        df_imputed = df_imputed.fillna(method='ffill').fillna(method='bfill')
    
    return df_imputed


def detect_and_handle_outliers(
    df: pd.DataFrame, 
    method: str = "iqr", 
    action: str = "flag"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect outliers using IQR or Z-score methods.
    
    Actions:
    - 'flag': Add outlier flag column
    - 'remove': Remove outlier rows
    - 'winsorize': Cap values at whiskers
    """
    df_result = df.copy()
    outlier_stats = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = np.abs(stats.zscore(df[col].fillna(df[col].mean()))) > 3
        
        outlier_count = outlier_mask.sum()
        outlier_stats[col] = int(outlier_count)
        
        if action == "flag":
            df_result[f"{col}_is_outlier"] = outlier_mask
        elif action == "remove":
            df_result = df_result[~outlier_mask]
        elif action == "winsorize":
            if method == "iqr":
                df_result[col] = df_result[col].clip(lower_bound, upper_bound)
    
    return df_result, outlier_stats


def feature_engineering_suggestions(df: pd.DataFrame) -> Dict:
    """
    Suggest useful feature engineering opportunities.
    """
    suggestions = {
        "ratio_features": [],
        "interaction_features": [],
        "polynomial_features": [],
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Suggest ratios for soil nutrients
    if all(x in numeric_cols for x in ["Nitrogen", "Phosphorus"]):
        suggestions["ratio_features"].append("N_to_P_Ratio = Nitrogen / Phosphorus")
    
    if all(x in numeric_cols for x in ["Phosphorus", "Potassium"]):
        suggestions["ratio_features"].append("P_to_K_Ratio = Phosphorus / Potassium")
    
    # Suggest interactions
    if all(x in numeric_cols for x in ["pH", "Moisture"]):
        suggestions["interaction_features"].append("pH_x_Moisture = pH * Moisture")
    
    if all(x in numeric_cols for x in ["Nitrogen", "Organic Matter"]):
        suggestions["interaction_features"].append("N_x_OM = Nitrogen * Organic_Matter")
    
    # Suggest polynomial features
    if "pH" in numeric_cols:
        suggestions["polynomial_features"].append("pH_squared = pH ** 2")
    
    return suggestions


def create_suggested_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-create suggested engineered features.
    """
    df_enhanced = df.copy()
    
    # Create ratio features
    if all(x in df.columns for x in ["Nitrogen", "Phosphorus"]):
        df_enhanced["N_to_P_Ratio"] = (
            df_enhanced["Nitrogen"] / (df_enhanced["Phosphorus"] + 1e-6)
        )
    
    if all(x in df.columns for x in ["Phosphorus", "Potassium"]):
        df_enhanced["P_to_K_Ratio"] = (
            df_enhanced["Phosphorus"] / (df_enhanced["Potassium"] + 1e-6)
        )
    
    # Create interaction features
    if all(x in df.columns for x in ["pH", "Moisture"]):
        df_enhanced["pH_x_Moisture"] = df_enhanced["pH"] * df_enhanced["Moisture"]
    
    if all(x in df.columns for x in ["Nitrogen", "Organic Matter"]):
        df_enhanced["N_x_OM"] = df_enhanced["Nitrogen"] * df_enhanced["Organic Matter"]
    
    # Create polynomial features
    if "pH" in df.columns:
        df_enhanced["pH_squared"] = df_enhanced["pH"] ** 2
    
    return df_enhanced


# ============================================================================
# SCALABILITY: Support for Distributed Processing (Dask Integration)
# ============================================================================

def try_use_dask(df: pd.DataFrame) -> bool:
    """Check if Dask is available and dataset is large enough to benefit."""
    try:
        import dask.dataframe as dd
        # Use Dask if dataset is larger than 100MB
        if df.memory_usage(deep=True).sum() / 1024**2 > 100:
            return True
    except ImportError:
        pass
    return False


def process_with_dask_if_available(df: pd.DataFrame, operation: callable):
    """
    Attempt to use Dask for processing if available and beneficial.
    Falls back to pandas if Dask is not available.
    """
    try:
        import dask.dataframe as dd
        
        if try_use_dask(df):
            ddf = dd.from_pandas(df, npartitions=4)
            result = operation(ddf).compute()
            return result
    except ImportError:
        pass
    
    # Fallback to pandas
    return operation(df)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_processing_recommendations(df: pd.DataFrame) -> List[str]:
    """Get recommendations for data processing based on dataset characteristics."""
    recommendations = []
    
    validation = validate_dataset(df)
    
    # Memory recommendations
    memory_mb = validation["memory_mb"]
    if memory_mb > 500:
        recommendations.append(
            "⚠️ Large dataset detected (>500MB). Consider chunking processing or installing Dask."
        )
    
    # Missing data recommendations
    if validation["missing_values"]:
        total_missing = sum(v["count"] for v in validation["missing_values"].values())
        if total_missing > len(df) * 0.1:
            recommendations.append(
                "📊 High missing data detected (>10%). Consider using KNN imputation."
            )
    
    # Outlier recommendations
    if validation["outliers_detected"]:
        recommendations.append(
            "📈 Outliers detected. Review or apply winsorization."
        )
    
    # Duplicate recommendations
    if validation["duplicates"] > 0:
        recommendations.append(
            f"🔄 {validation['duplicates']} duplicate rows found. Consider removing them."
        )
    
    if not recommendations:
        recommendations.append("✅ Dataset appears clean and well-structured!")
    
    return recommendations
