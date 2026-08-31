import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import json
from scipy import stats as scipy_stats

LANG = os.environ.get("HG_MC_AUTO_LANG", "zh").lower()

TRANSLATIONS = {
    "zh": {
        "title": "数据分析与异常检测程序",
        "input_data": "请输入输入数据文件路径（Excel 格式）: ",
        "input_output": "请输入输出结果目录路径: ",
        "path_empty": "路径不能为空，请重新输入。",
        "not_exist": "文件不存在: {path}，请重新输入。",
        "excel_required": "请提供 Excel 文件（.xlsx 或 .xls 格式）",
        "mode_title": "分析模式选择",
        "mode_prompt": "请选择分析模式 (1, 2, 或 3): ",
        "mode1": "1. 经验模型（使用预定义范围）",
        "mode2": "2. 自定义数据质量范围",
        "mode3": "3. 使用上次自定义范围",
        "invalid_mode": "无效选择。请输入 1、2 或 3。",
        "config_title": "自定义数据质量范围配置",
    },
    "en": {
        "title": "Data Analysis and Anomaly Detection Program",
        "input_data": "Please enter the input data file path (Excel format): ",
        "input_output": "Please enter the output results directory path: ",
        "path_empty": "Path cannot be empty, please re-enter.",
        "not_exist": "File does not exist: {path}, please re-enter.",
        "excel_required": "Please provide an Excel file (.xlsx or .xls format)",
        "mode_title": "Analysis Mode Selection",
        "mode_prompt": "Please select analysis mode (1, 2, or 3): ",
        "mode1": "1. Empirical Model (use predefined ranges)",
        "mode2": "2. Custom Data Quality Ranges",
        "mode3": "3. Use Last Custom Ranges",
        "invalid_mode": "Invalid choice. Please enter 1, 2, or 3.",
        "config_title": "Custom Data Quality Range Configuration",
    },
}


def apply_lang_from_cli():
    global LANG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lang", choices=["zh", "en"], default=os.environ.get("HG_MC_AUTO_LANG", "zh"))
    args, _ = parser.parse_known_args()
    LANG = args.lang.lower()
    os.environ["HG_MC_AUTO_LANG"] = LANG


def text(key, **kwargs):
    template = TRANSLATIONS.get(LANG, TRANSLATIONS["zh"]).get(key, key)
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def get_user_paths():
    """Get input and output paths from user"""
    print("=" * 60)
    print(text("title"))
    print("=" * 60)
    
    while True:
        data_path = input(text("input_data")).strip()
        if not data_path:
            print(text("path_empty"))
            continue
        if not os.path.exists(data_path):
            print(text("not_exist", path=data_path))
            continue
        if not data_path.lower().endswith(('.xlsx', '.xls')):
            print(text("excel_required"))
            continue
        break
    
    while True:
        results_path = input(text("input_output")).strip()
        if not results_path:
            print(text("path_empty"))
            continue
        break
    
    return data_path, results_path


def get_analysis_mode():
    """Let user select analysis mode"""
    print("\n" + "=" * 60)
    print(text("mode_title"))
    print("=" * 60)
    print(text("mode1"))
    print(text("mode2"))
    print(text("mode3"))
    
    while True:
        choice = input(text("mode_prompt")).strip()
        if choice in ['1', '2', '3']:
            return choice
        print(text("invalid_mode"))

# Configuration file for custom ranges
CUSTOM_RANGES_FILE = "custom_ranges_config.json"

# Global variables to store configuration
current_label_ranges = None
current_internal_precision_ranges = None

def setup_directories(results_path):
    """Create output directory structure"""
    fig_path = os.path.join(results_path, "fig")
    report_path = os.path.join(results_path, "report")
    table_path = os.path.join(results_path, "tables")
    
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(report_path, exist_ok=True)
    os.makedirs(table_path, exist_ok=True)
    
    return fig_path, report_path, table_path

def save_custom_ranges(ranges):
    """Save custom ranges to configuration file"""
    try:
        with open(CUSTOM_RANGES_FILE, 'w') as f:
            json.dump(ranges, f, indent=4)
        print("Custom ranges saved to configuration file")
        return True
    except Exception as e:
        print(f"Error saving custom ranges: {str(e)}")
        return False

def load_custom_ranges():
    """Load custom ranges from configuration file"""
    try:
        if not os.path.exists(CUSTOM_RANGES_FILE):
            return None
            
        with open(CUSTOM_RANGES_FILE, 'r') as f:
            ranges = json.load(f)
        print("Custom ranges loaded from configuration file")
        return ranges
    except Exception as e:
        print(f"Error loading custom ranges: {str(e)}")
        return None

def configure_custom_ranges():
    """Configure custom data quality ranges"""
    print("\n" + "=" * 60)
    print("Custom Data Quality Range Configuration")
    print("=" * 60)
    
    custom_ranges = {}
    
    # External precision ranges for 3133, 3177, 8610, sample
    external_labels = ["3133", "3177", "8610", "sample"]
    external_columns = ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]
    
    print("\n--- External Precision Ranges ---")
    for label in external_labels:
        print(f"\nConfiguring ranges for {label}:")
        custom_ranges[label] = {}
        
        for col in external_columns:
            while True:
                try:
                    print(f"  {col}:")
                    mean = float(input(f"    Enter mean value for {col}: "))
                    range_val = float(input(f"    Enter range value for {col}: "))
                    custom_ranges[label][col] = {"mean": mean, "range": range_val}
                    break
                except ValueError:
                    print("    Invalid input. Please enter numeric values.")
    
    # R-THg(%) ranges for 3133, 3177, 8610
    print("\n--- R-THg(%) Ranges ---")
    rtHg_labels = ["3133", "3177", "8610"]
    custom_ranges["r_thg"] = {}
    
    for label in rtHg_labels:
        print(f"\nConfiguring R-THg(%) range for {label}:")
        while True:
            try:
                mean = float(input(f"  Enter mean value for R-THg(%): "))
                range_val = float(input(f"  Enter range value for R-THg(%): "))
                custom_ranges["r_thg"][label] = {"mean": mean, "range": range_val}
                break
            except ValueError:
                print("  Invalid input. Please enter numeric values.")
    
    # Internal precision ranges for 3133, 3177, 8610
    internal_labels = ["3133", "3177", "8610"]
    internal_columns = [
        "StdErr(abs)202Hg/198Hg",
        "StdErr(abs)201Hg/198Hg", 
        "StdErr(abs)200Hg/198Hg",
        "StdErr(abs)199Hg/198Hg"
    ]
    
    print("\n--- Internal Precision Ranges ---")
    custom_ranges["internal_precision"] = {}
    for label in internal_labels:
        print(f"\nConfiguring internal precision thresholds for {label}:")
        custom_ranges["internal_precision"][label] = {}
        
        for col in internal_columns:
            while True:
                try:
                    threshold = float(input(f"  Enter threshold for {col}: "))
                    custom_ranges["internal_precision"][label][col] = threshold
                    break
                except ValueError:
                    print("  Invalid input. Please enter numeric values.")
    
    # Show summary and confirm
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    
    print("\nExternal Precision Ranges:")
    for label in external_labels:
        print(f"\n{label}:")
        for col in external_columns:
            mean = custom_ranges[label][col]["mean"]
            range_val = custom_ranges[label][col]["range"]
            print(f"  {col}: {mean} ± {range_val}")
    
    print("\nR-THg(%) Ranges:")
    for label in rtHg_labels:
        mean = custom_ranges["r_thg"][label]["mean"]
        range_val = custom_ranges["r_thg"][label]["range"]
        print(f"  {label}: {mean} ± {range_val}")
    
    print("\nInternal Precision Thresholds:")
    for label in internal_labels:
        print(f"\n{label}:")
        for col in internal_columns:
            threshold = custom_ranges["internal_precision"][label][col]
            print(f"  {col}: {threshold}")
    
    # Confirm configuration
    while True:
        confirm = input("\nConfirm these settings? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            if save_custom_ranges(custom_ranges):
                return custom_ranges
            else:
                print("Failed to save configuration. Please try again.")
                return configure_custom_ranges()
        elif confirm in ['n', 'no']:
            print("Restarting configuration...")
            return configure_custom_ranges()
        else:
            print("Please enter 'y' or 'n'.")

def get_label_ranges(analysis_mode):
    """Get label ranges based on analysis mode"""
    global current_label_ranges, current_internal_precision_ranges
    
    if analysis_mode == '1':
        # Empirical model - predefined ranges
        current_label_ranges = {
            "3133": {
                "d202(‰)": {"mean": 0.00, "range": 0.12},
                "D199(‰)": {"mean": 0.00, "range": 0.05},
                "D200(‰)": {"mean": 0.00, "range": 0.05},
                "D201(‰)": {"mean": 0.00, "range": 0.05}
            },
            "3177": {
                "d202(‰)": {"mean": -0.55, "range": 0.14},
                "D199(‰)": {"mean": -0.02, "range": 0.07},
                "D200(‰)": {"mean": 0.01, "range": 0.05},
                "D201(‰)": {"mean": -0.04, "range": 0.09}
            },
            "8610": {
                "d202(‰)": {"mean": -0.51, "range": 0.09},
                "D199(‰)": {"mean": -0.02, "range": 0.07},
                "D200(‰)": {"mean": 0.02, "range": 0.05},
                "D201(‰)": {"mean": -0.03, "range": 0.06}
            },
            "sample": {
                "d202(‰)": {"mean": -1.00, "range": 3.00},
                "D199(‰)": {"mean": 0.50, "range": 1.00},
                "D200(‰)": {"mean": 0.25, "range": 0.55},
                "D201(‰)": {"mean": 0.50, "range": 0.70}
            }
        }
        current_internal_precision_ranges = None
        return current_label_ranges, current_internal_precision_ranges
    
    elif analysis_mode == '2':
        # Custom ranges - configure new
        custom_ranges = configure_custom_ranges()
        # Extract label_ranges (without internal_precision and r_thg keys)
        current_label_ranges = {k: v for k, v in custom_ranges.items() if k not in ["internal_precision", "r_thg"]}
        current_internal_precision_ranges = custom_ranges.get("internal_precision")
        return current_label_ranges, current_internal_precision_ranges
    
    elif analysis_mode == '3':
        # Use last custom ranges
        custom_ranges = load_custom_ranges()
        if custom_ranges:
            current_label_ranges = {k: v for k, v in custom_ranges.items() if k not in ["internal_precision", "r_thg"]}
            current_internal_precision_ranges = custom_ranges.get("internal_precision")
            return current_label_ranges, current_internal_precision_ranges
        else:
            print("No saved custom ranges found. Switching to empirical model.")
            return get_label_ranges('1')

def calculate_credibility(df):
    """Calculate confidence level"""
    print("\n" + "=" * 60)
    print("Calculating credibility levels...")
    print("=" * 60)
    
    # Add a 'Credibility' column, with an initial value of empty
    df['Credibility'] = ''
    
    # Get the index of all rows
    indices = df.index.tolist()
    
    # Iterate through each row
    for i, idx in enumerate(indices):
        current_label = df.at[idx, 'Label']
        current_status = df.at[idx, 'Check Status']
        
        # Calculate credibility only for rows with specific tags and a status of Normal
        if current_label in ["3177", "8610", "sample"] and current_status == "Normal":
            prev_3133_status = None
            next_3133_status = None
            
            # Find the previous 3133 tag
            for j in range(i-1, -1, -1):
                prev_idx = indices[j]
                if df.at[prev_idx, 'Label'] == "3133":
                    prev_3133_status = df.at[prev_idx, 'Check Status']
                    break
            
            # Find the next 3133 tag
            for j in range(i+1, len(indices)):
                next_idx = indices[j]
                if df.at[next_idx, 'Label'] == "3133":
                    next_3133_status = df.at[next_idx, 'Check Status']
                    break
            
            # Calculate reliability based on the state of 3133 before and after
            if prev_3133_status is not None and next_3133_status is not None:
                if prev_3133_status == "Normal" and next_3133_status == "Normal":
                    df.at[idx, 'Credibility'] = "High"
                elif (prev_3133_status == "Normal" and next_3133_status == "Abnormal") or \
                     (prev_3133_status == "Abnormal" and next_3133_status == "Normal"):
                    df.at[idx, 'Credibility'] = "Middle"
                elif prev_3133_status == "Abnormal" and next_3133_status == "Abnormal":
                    df.at[idx, 'Credibility'] = "Low"
                else:
                    df.at[idx, 'Credibility'] = "Not Calculated"
            else:
                # If the previous or next 3133 tag is not found
                if prev_3133_status is None and next_3133_status is None:
                    df.at[idx, 'Credibility'] = "No 3133 Found"
                elif prev_3133_status is None:
                    df.at[idx, 'Credibility'] = "No Previous 3133"
                elif next_3133_status is None:
                    df.at[idx, 'Credibility'] = "No Next 3133"
    
    # Statistical Confidence Distribution
    credibility_counts = df['Credibility'].value_counts()
    print("Credibility distribution:")
    for credibility, count in credibility_counts.items():
        if credibility:
            print(f"  {credibility}: {count}")
    
    return df

def detect_and_remove_outliers_by_label(df, label_col='Label', method='iqr', threshold=3.0):
    """
    Detect and remove outliers separately for each label type
    
    Parameters:
    df: Input DataFrame
    label_col: Column name containing labels
    method: Outlier detection method ('iqr', 'zscore')
    threshold: Outlier threshold
    
    Returns:
    df_clean: DataFrame with outliers removed
    outlier_stats: Statistics about removed outliers
    """
    print("\n" + "="*60)
    print("OUTLIER DETECTION AND REMOVAL BY LABEL")
    print("="*60)
    
    df_clean = df.copy()
    outlier_stats = {}
    total_outliers = 0
    columns_to_check = ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]
    
    # Get unique labels
    unique_labels = df[label_col].unique()
    print(f"Processing {len(unique_labels)} label types: {unique_labels}")
    
    # Process each label separately
    for label in unique_labels:
        print(f"\nProcessing label: {label}")
        
        # Get data for this label
        label_mask = df[label_col] == label
        label_data = df[label_mask].copy()
        
        if len(label_data) < 5:  # Too few samples for reliable outlier detection
            print(f"  Skipping - only {len(label_data)} samples")
            continue
        
        # Initialize outlier mask for this label
        outlier_mask = pd.Series(False, index=label_data.index)
        
        # Check each column
        for col in columns_to_check:
            if col not in label_data.columns:
                continue
                
            # Remove NaN values temporarily
            col_data = label_data[col].dropna()
            if len(col_data) < 3:  # Need at least 3 values for outlier detection
                continue
            
            if method == 'iqr':
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:  # If all values are the same
                    print(f"  {col}: IQR is 0, skipping outlier detection")
                    continue
                
                # Calculate bounds
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Mark outliers
                col_outliers = (label_data[col] < lower_bound) | (label_data[col] > upper_bound)
                outlier_mask |= col_outliers
                
                n_outliers = col_outliers.sum()
                if n_outliers > 0:
                    print(f"  {col}: {n_outliers} outliers detected")
                    print(f"    Range: [{lower_bound:.4f}, {upper_bound:.4f}]")
                    print(f"    Outlier values: [{label_data[col_outliers][col].min():.4f}, {label_data[col_outliers][col].max():.4f}]")
            
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(scipy_stats.zscore(col_data))
                outlier_indices = np.where(z_scores > threshold)[0]
                original_indices = col_data.index[outlier_indices]
                
                # Mark outliers
                temp_mask = pd.Series(False, index=label_data.index)
                temp_mask.loc[original_indices] = True
                outlier_mask |= temp_mask
                
                n_outliers = len(outlier_indices)
                if n_outliers > 0:
                    print(f"  {col}: {n_outliers} outliers detected (|Z| > {threshold})")
        
        # Remove outliers for this label
        n_label_outliers = outlier_mask.sum()
        total_outliers += n_label_outliers
        
        if n_label_outliers > 0:
            print(f"  Removing {n_label_outliers} outliers ({n_label_outliers/len(label_data)*100:.1f}% of label data)")
            # Remove from clean DataFrame
            df_clean = df_clean.drop(label_data[outlier_mask].index)
            
            # Record statistics
            outlier_stats[label] = {
                'total_samples': len(label_data),
                'outliers_removed': n_label_outliers,
                'percentage_removed': n_label_outliers/len(label_data)*100,
                'remaining_samples': len(label_data) - n_label_outliers
            }
        else:
            print(f"  No outliers detected")
            outlier_stats[label] = {
                'total_samples': len(label_data),
                'outliers_removed': 0,
                'percentage_removed': 0,
                'remaining_samples': len(label_data)
            }
    
    print(f"\nOutlier removal completed:")
    print(f"  Original data shape: {df.shape}")
    print(f"  Cleaned data shape: {df_clean.shape}")
    print(f"  Total outliers removed: {total_outliers} ({total_outliers/len(df)*100:.1f}% of all data)")
    
    # Display detailed statistics
    print("\nOutlier statistics by label:")
    for label, stats in outlier_stats.items():
        print(f"  {label}: {stats['outliers_removed']}/{stats['total_samples']} outliers "
              f"({stats['percentage_removed']:.1f}%) - {stats['remaining_samples']} samples remaining")
    
    return df_clean, outlier_stats

def smooth_noise_by_label(df, label_col='Label', columns=None, window_size=3):
    """
    Apply smoothing to reduce noise in time series data for each label separately
    
    Parameters:
    df: Input DataFrame
    label_col: Column name containing labels
    columns: List of columns to smooth (default: ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"])
    window_size: Size of moving average window
    
    Returns:
    df_smoothed: DataFrame with smoothed values
    """
    print("\n" + "="*60)
    print("NOISE SMOOTHING BY LABEL")
    print("="*60)
    
    if columns is None:
        columns = ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]
    
    df_smoothed = df.copy()
    unique_labels = df[label_col].unique()
    
    for label in unique_labels:
        print(f"\nSmoothing label: {label}")
        
        # Get data for this label
        label_mask = df[label_col] == label
        label_indices = df[label_mask].index
        
        if len(label_indices) < window_size:
            print(f"  Skipping - only {len(label_indices)} samples (need at least {window_size})")
            continue
        
        # Sort by index to maintain order
        sorted_indices = sorted(label_indices)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Extract values for this label
            values = df.loc[sorted_indices, col].values
            
            # Apply moving average smoothing
            if len(values) >= window_size:
                smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                
                # Pad the beginning and end to maintain original length
                padding = window_size - 1
                padded_values = np.empty_like(values)
                
                # Beginning: use original values
                padded_values[:padding//2] = values[:padding//2]
                
                # Middle: use smoothed values
                padded_values[padding//2:padding//2+len(smoothed_values)] = smoothed_values
                
                # End: use original values
                if len(values) > padding//2 + len(smoothed_values):
                    padded_values[padding//2+len(smoothed_values):] = values[-(len(values) - (padding//2 + len(smoothed_values))):]
                
                # Update smoothed values
                df_smoothed.loc[sorted_indices, col] = padded_values
                
                # Calculate smoothing effect
                noise_reduction = np.std(values) - np.std(padded_values)
                print(f"  {col}: Noise reduced by {noise_reduction:.4f} (std)")
    
    print(f"\nNoise smoothing completed for {len(unique_labels)} labels")
    
    return df_smoothed

def main_analysis(data_path, results_path, fig_path, report_path, table_path, analysis_mode):
    """Main data analysis function"""
    global current_label_ranges, current_internal_precision_ranges
    
    # Get label ranges based on analysis mode
    if current_label_ranges is None:
        current_label_ranges, current_internal_precision_ranges = get_label_ranges(analysis_mode)
    
    if current_label_ranges is None:
        print("Failed to get label ranges. Exiting.")
        return None

    # Set font support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Load data (only first sheet)
    try:
        df = pd.read_excel(data_path, sheet_name=0)
        print("Data loaded successfully!")
        print(f"Original data shape: {df.shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None

    # ==================================================================
    # STEP 1: OUTLIER DETECTION AND REMOVAL
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 1: PRE-PROCESSING - OUTLIER AND NOISE HANDLING")
    print("="*60)
    
    # Ask user about outlier handling
    while True:
        outlier_choice = input("\nDo you want to perform outlier detection and removal? (y/n): ").strip().lower()
        if outlier_choice in ['y', 'yes', 'n', 'no']:
            break
        print("Please enter 'y' or 'n'")
    
    df_processed = df.copy()
    outlier_stats = None
    
    if outlier_choice in ['y', 'yes']:
        # Detect and remove outliers
        df_processed, outlier_stats = detect_and_remove_outliers_by_label(
            df_processed, 
            label_col='Label', 
            method='iqr', 
            threshold=3.0
        )
        
        # Ask about noise smoothing
        while True:
            smooth_choice = input("\nDo you want to apply noise smoothing? (y/n): ").strip().lower()
            if smooth_choice in ['y', 'yes', 'n', 'no']:
                break
            print("Please enter 'y' or 'n'")
        
        if smooth_choice in ['y', 'yes']:
            # Get window size for smoothing
            while True:
                try:
                    window_size = int(input("Enter window size for smoothing (default 3): ").strip() or "3")
                    if window_size >= 2 and window_size <= 10:
                        break
                    print("Please enter a number between 2 and 10")
                except ValueError:
                    print("Please enter a valid number")
            
            df_processed = smooth_noise_by_label(
                df_processed,
                label_col='Label',
                window_size=window_size
            )
    
    # ==================================================================
    # STEP 2: DATA ANALYSIS WITH PROCESSED DATA
    # ==================================================================
    
    # Define labels for different analysis types
    all_labels = ["3133", "3177", "8610", "sample"]
    columns_of_interest = ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]

    # Initialize results dictionaries for both analysis types
    results_type1 = {}
    results_type2 = {}
    classified_data_type1 = {}
    classified_data_type2 = {}
    abnormal_data_list = []

    def check_D199_D201_condition(row):
        """Check if D199 and D201 meet the additional abnormal condition"""
        D199 = row["D199(‰)"]
        D201 = row["D201(‰)"]
        
        if pd.isna(D199) or pd.isna(D201):
            return False
        
        condition1 = abs(D199) > 0.1 and abs(D201) > 0.1
        condition2 = (D199 / D201) < 0 if D201 != 0 else False
        
        return condition1 and condition2

    def check_value_in_range(value, mean, range_val):
        """Check if value is within the specified range (mean ± range)"""
        lower_bound = mean - range_val
        upper_bound = mean + range_val
        return lower_bound <= value <= upper_bound

    def perform_analysis(labels_to_analyze, analysis_type="type1", input_df=None):
        """Perform statistical analysis for given labels and analysis type"""
        if input_df is None:
            input_df = df_processed
            
        results = {}
        classified_data = {}
        
        for label in labels_to_analyze:
            results[label] = {}
            label_mask = input_df.iloc[:, 0] == label
            label_df_full = input_df[label_mask].copy()
            label_data = label_df_full[columns_of_interest].copy()
            
            if len(label_data) == 0:
                print(f"Warning: No data found for label {label}")
                continue
            
            print(f"Processing {label} data for {analysis_type}, total {len(label_data)} records")
            
            status_list = []
            for idx, row in label_data.iterrows():
                if row[columns_of_interest].isnull().any():
                    status_list.append(np.nan)
                    continue
                
                if check_D199_D201_condition(row):
                    if label == "sample":
                        status_list.append("Potential anomaly detected; please pay attention to the source of the sample")
                    else:
                        status_list.append("Abnormal")
                    continue
                
                all_within_range = True
                for col in columns_of_interest:
                    value = row[col]
                    mean_val = current_label_ranges[label][col]["mean"]
                    range_val = current_label_ranges[label][col]["range"]
                    
                    if not check_value_in_range(value, mean_val, range_val):
                        all_within_range = False
                        break
                
                if all_within_range:
                    status_list.append("Normal")
                else:
                    if label == "sample":
                        status_list.append("Potential anomaly detected; please pay attention to the source of the sample")
                    else:
                        status_list.append("Abnormal")
            
            label_df_full = label_df_full.reset_index(drop=True)
            label_df_full["Check Status"] = status_list
            classified_data[label] = label_df_full
            
            if analysis_type == "type2":
                normal_mask = label_df_full["Check Status"] == "Normal"
                label_data_for_stats = label_df_full[normal_mask][columns_of_interest].copy()
            else:
                label_data_for_stats = label_data.copy()
            
            if len(label_data_for_stats) > 0:
                mean_values = label_data_for_stats.mean()
                std_values = label_data_for_stats.std()
                two_std_values = 2 * std_values
                results[label]["mean"] = mean_values
                results[label]["2*std"] = two_std_values
                results[label]["count"] = len(label_data_for_stats)
            else:
                results[label]["mean"] = pd.Series([np.nan] * len(columns_of_interest), index=columns_of_interest)
                results[label]["2*std"] = pd.Series([np.nan] * len(columns_of_interest), index=columns_of_interest)
                results[label]["count"] = 0
        
        return results, classified_data

    # Perform Type 1 analysis (all data)
    print("\n" + "="*60)
    print("STEP 2: TYPE 1 ANALYSIS (ALL DATA)")
    print("="*60)
    results_type1, classified_data_type1 = perform_analysis(all_labels, "type1", df_processed)

    # Perform Type 2 analysis (only Normal data)
    print("\n" + "="*60)
    print("STEP 3: TYPE 2 ANALYSIS (ONLY NORMAL DATA)")
    print("="*60)
    results_type2, classified_data_type2 = perform_analysis(all_labels, "type2", df_processed)

    # Extract abnormal data from Type 1 analysis
    for label in all_labels:
        if label in classified_data_type1:
            abnormal_mask = classified_data_type1[label]["Check Status"].isin(["Abnormal", "Potential anomaly detected; please pay attention to the source of the sample"])
            abnormal_data = classified_data_type1[label][abnormal_mask]
            if not abnormal_data.empty:
                abnormal_data_list.append(abnormal_data)

    # ==================================================================
    # STEP 3: VISUALIZATION AND REPORTING
    # ==================================================================
    
    # Plotting for both analysis types
    def create_plots(results_data, classified_data, analysis_type):
        """Create statistical plots for the given analysis type"""
        plot_path = os.path.join(fig_path, analysis_type)
        os.makedirs(plot_path, exist_ok=True)
        
        for label in all_labels:
            if label not in results_data or results_data[label].get("count", 0) == 0:
                continue
                
            if analysis_type == "type2" and label in classified_data:
                normal_mask = classified_data[label]["Check Status"] == "Normal"
                label_data = classified_data[label][normal_mask][columns_of_interest].copy()
            else:
                label_mask = df_processed.iloc[:, 0] == label
                label_data = df_processed[label_mask][columns_of_interest].copy()
            
            for col in columns_of_interest:
                if col not in label_data.columns or label_data[col].isnull().all():
                    continue
                    
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.hist(label_data[col].dropna(), bins=20, alpha=0.75, color='blue', edgecolor='black')
                plt.title(f"{label} {col} Histogram ({analysis_type})")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                
                plt.subplot(1, 2, 2)
                stats.probplot(label_data[col].dropna(), dist="norm", plot=plt)
                plt.title(f"{label} {col} Q-Q Plot ({analysis_type})")
                
                fig_file = os.path.join(plot_path, f"{label}_{col}.png")
                plt.tight_layout()
                plt.savefig(fig_file)
                plt.close()

    # Create plots for both analysis types
    print("\n" + "="*60)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("="*60)
    
    print("Creating plots for Type 1 analysis...")
    create_plots(results_type1, classified_data_type1, "type1")

    print("Creating plots for Type 2 analysis...")
    create_plots(results_type2, classified_data_type2, "type2")

    # Save all abnormal data to a separate table
    if abnormal_data_list:
        all_abnormal_data = pd.concat(abnormal_data_list, ignore_index=True)
        abnormal_file = os.path.join(table_path, "all_abnormal_data.xlsx")
        try:
            all_abnormal_data.to_excel(abnormal_file, index=False)
            print(f"All abnormal data saved to: {abnormal_file}")
        except Exception as e:
            print(f"Error saving abnormal data: {e}")
    else:
        print("No abnormal data found to save")
        abnormal_file = None

    # Save classified data for each category
    for label, data in classified_data_type1.items():
        if label in ["3133", "3177", "8610", "sample"]:
            table_file = os.path.join(table_path, f"{label}_classified_data.xlsx")
            try:
                with pd.ExcelWriter(table_file, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name='Data')
                    
                    workbook = writer.book
                    worksheet = writer.sheets['Data']
                    
                    if "THg(%)" in data.columns:
                        for idx, col_name in enumerate(data.columns):
                            if col_name == "THg(%)":
                                for row in range(2, len(data) + 2):
                                    cell = worksheet.cell(row=row, column=idx+1)
                                    if cell.value is not None:
                                        try:
                                            cell_value = float(cell.value)
                                            cell.value = cell_value
                                            cell.number_format = '0.00%'
                                        except (ValueError, TypeError):
                                            pass
                print(f"{label} classified data saved to: {table_file}")
            except Exception as e:
                print(f"Error saving {label} data: {e}")

    # Generate combined table (only 3133, 3177, 8610)
    if classified_data_type1:
        data_to_combine = []
        for label in ["3133", "3177", "8610"]:
            if label in classified_data_type1:
                data_to_combine.append(classified_data_type1[label])
        
        if data_to_combine:
            combined_df = pd.concat(data_to_combine, ignore_index=True)
            
            sort_col = None
            for candidate in ["Time", "time", "Cycle"]:
                if candidate in combined_df.columns:
                    sort_col = candidate
                    break
            if sort_col:
                try:
                    combined_df[sort_col] = pd.to_datetime(combined_df[sort_col], errors='coerce')
                    combined_df = combined_df.sort_values(by=sort_col).reset_index(drop=True)
                except Exception:
                    combined_df = combined_df.sort_values(by=sort_col, kind='stable').reset_index(drop=True)
            
            combined_file = os.path.join(table_path, "combined_classified_data.xlsx")
            try:
                with pd.ExcelWriter(combined_file, engine='openpyxl') as writer:
                    combined_df.to_excel(writer, index=False, sheet_name='Data')
                    
                    workbook = writer.book
                    worksheet = writer.sheets['Data']
                    
                    if "THg(%)" in combined_df.columns:
                        for idx, col_name in enumerate(combined_df.columns):
                            if col_name == "THg(%)":
                                for row in range(2, len(combined_df) + 2):
                                    cell = worksheet.cell(row=row, column=idx+1)
                                    if cell.value is not None:
                                        try:
                                            cell_value = float(cell.value)
                                            cell.value = cell_value
                                            cell.number_format = '0.00%'
                                        except (ValueError, TypeError):
                                            pass
                print(f"Combined classified data saved to: {combined_file}")
            except Exception as e:
                print(f"Error saving combined classified data: {e}")
    
    # Create a merged table of all category data (including samples)
    if classified_data_type1:
        try:
            # Merge data from all tags (including samples)
            all_combined = pd.concat(list(classified_data_type1.values()), ignore_index=True)
            
            # Try sorting by Time/Cycle
            sort_col = None
            for candidate in ["Time", "time", "Cycle"]:
                if candidate in all_combined.columns:
                    sort_col = candidate
                    break
            if sort_col:
                try:
                    all_combined[sort_col] = pd.to_datetime(all_combined[sort_col], errors='coerce')
                    all_combined = all_combined.sort_values(by=sort_col).reset_index(drop=True)
                except Exception:
                    all_combined = all_combined.sort_values(by=sort_col, kind='stable').reset_index(drop=True)

            # Determine the file name based on the analysis pattern
            if analysis_mode == '1':
                all_combined_file = os.path.join(table_path, "all_empirical_model_classified_data.xlsx")
                file_type = "Empirical Model"
            elif analysis_mode == '2':
                all_combined_file = os.path.join(table_path, "all_custom_model_classified_data.xlsx")
                file_type = "Custom Model"
            else:
                all_combined_file = os.path.join(table_path, "all_custom_model_classified_data.xlsx")
                file_type = "Custom Model (Last Saved)"
            
            print(f"\nCalculating credibility for {file_type} data...")
            
            # Calculate reliability (calculated for all modes)
            all_combined_with_credibility = calculate_credibility(all_combined)
                
            all_combined_with_credibility.to_excel(all_combined_file, index=False)
            print(f"All classified data (all labels) with credibility saved to: {all_combined_file}")
            
            # Display credibility analysis results
            print(f"\nCredibility analysis completed for {file_type}!")
            
        except Exception as e:
            print(f"Error saving all classified combined data: {e}")
    else:
        print("No classified data available to save as combined all-label table")

    # ==================================================================
    # STEP 5: GENERATE COMPREHENSIVE REPORT
    # ==================================================================
    
    # Generate comprehensive report
    report_file = os.path.join(report_path, "statistical_analysis_report.txt")
    try:
        with open(report_file, "w", encoding='utf-8') as f:
            f.write("Statistical Analysis Report with Outlier and Noise Handling\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Mode: {analysis_mode}\n")
            mode_description = {
                '1': 'Empirical Model (predefined ranges)',
                '2': 'Custom Data Quality Ranges',
                '3': 'Last Custom Ranges'
            }
            f.write(f"Mode Description: {mode_description.get(analysis_mode, 'Unknown')}\n\n")
            
            # Data processing summary
            f.write("DATA PROCESSING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Original data shape: {df.shape}\n")
            f.write(f"Processed data shape: {df_processed.shape}\n")
            
            if outlier_stats:
                f.write("\nOutlier Removal Statistics:\n")
                total_removed = sum(stats['outliers_removed'] for stats in outlier_stats.values())
                f.write(f"Total outliers removed: {total_removed} ({total_removed/len(df)*100:.1f}%)\n")
                for label, stats_info in outlier_stats.items():
                    f.write(f"  {label}: {stats_info['outliers_removed']} outliers removed "
                           f"({stats_info['percentage_removed']:.1f}%)\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
            
            f.write("Range Criteria for Each Label:\n")
            for label in all_labels:
                f.write(f"{label}:\n")
                for col in columns_of_interest:
                    mean_val = current_label_ranges[label][col]["mean"]
                    range_val = current_label_ranges[label][col]["range"]
                    f.write(f"  {col}: {mean_val} ± {range_val} (range: {mean_val - range_val:.3f} to {mean_val + range_val:.3f})\n")
                f.write("\n")
            f.write("\n")
            
            # Type 1 Analysis Results
            f.write("TYPE 1 ANALYSIS RESULTS (All Data)\n")
            f.write("=" * 40 + "\n\n")
            for label in all_labels:
                if label not in results_type1:
                    continue
                f.write(f"Label: {label}\n")
                if "count" in results_type1[label]:
                    f.write(f"Total Records: {results_type1[label]['count']}\n")
                if "mean" in results_type1[label]:
                    f.write(f"Mean Values:\n{results_type1[label]['mean']}\n")
                if "2*std" in results_type1[label]:
                    f.write(f"2 Standard Deviations:\n{results_type1[label]['2*std']}\n")
                
                if label in classified_data_type1:
                    if label == "sample":
                        non_empty_status = classified_data_type1[label]["Check Status"][classified_data_type1[label]["Check Status"].notna() & (classified_data_type1[label]["Check Status"] != "")]
                        status_counts = non_empty_status.value_counts()
                    else:
                        status_counts = classified_data_type1[label]["Check Status"].value_counts(dropna=True)
                    
                    f.write("Check Status Statistics:\n")
                    for status, count in status_counts.items():
                        f.write(f"  {status}: {count} records\n")
                f.write("-" * 50 + "\n\n")
            
            # Type 2 Analysis Results
            f.write("TYPE 2 ANALYSIS RESULTS (Only Normal Data)\n")
            f.write("=" * 40 + "\n\n")
            for label in all_labels:
                if label not in results_type2:
                    continue
                f.write(f"Label: {label}\n")
                if "count" in results_type2[label]:
                    f.write(f"Normal Records: {results_type2[label]['count']}\n")
                if "mean" in results_type2[label]:
                    f.write(f"Mean Values:\n{results_type2[label]['mean']}\n")
                if "2*std" in results_type2[label]:
                    f.write(f"2 Standard Deviations:\n{results_type2[label]['2*std']}\n")
                f.write("-" * 50 + "\n\n")
                
        print(f"Report saved to: {report_file}")
    except Exception as e:
        print(f"Error generating report: {e}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Classified data tables location: {table_path}")
    print(f"Image files location: {fig_path}")
    print(f"Report file location: {report_path}")
    
    return abnormal_file

def process_abnormal_data(abnormal_file_path, table_path):
    """Process abnormal data and add cause analysis"""
    global current_internal_precision_ranges
    
    print("\n" + "=" * 60)
    print("Starting abnormal data analysis...")
    print("=" * 60)
    
    try:
        # Load custom ranges to get R-THg(%) ranges if available
        custom_ranges = load_custom_ranges()
        r_thg_ranges = custom_ranges.get("r_thg") if custom_ranges else None
        
        # If no custom internal precision ranges, use defaults
        if current_internal_precision_ranges is None:
            current_internal_precision_ranges = {
                '3133': {
                    'StdErr(abs)202Hg/198Hg': 0.0010,
                    'StdErr(abs)201Hg/198Hg': 0.0004,
                    'StdErr(abs)200Hg/198Hg': 0.0009,
                    'StdErr(abs)199Hg/198Hg': 0.0007
                },
                '3177': {
                    'StdErr(abs)202Hg/198Hg': 0.0010,
                    'StdErr(abs)201Hg/198Hg': 0.0004,
                    'StdErr(abs)200Hg/198Hg': 0.0009,
                    'StdErr(abs)199Hg/198Hg': 0.0007
                },
                '8610': {
                    'StdErr(abs)202Hg/198Hg': 0.0010,
                    'StdErr(abs)201Hg/198Hg': 0.0004,
                    'StdErr(abs)200Hg/198Hg': 0.0009,
                    'StdErr(abs)199Hg/198Hg': 0.0007
                }
            }
        
        # Read Excel file
        df = pd.read_excel(abnormal_file_path)
        print(f"Abnormal data file loaded successfully: {abnormal_file_path}")

        df['Cause of the anomaly'] = ''
        target_labels = ['3133', '3177', '8610']

        processed_count = 0
        for index, row in df.iterrows():
            current_label = str(row['Label'])
            
            if current_label in target_labels:
                causes = []
                
                # First condition: Check if R-THg(%) is outside custom range or default 10%
                r_thg_value = row['R-THg(%)']
                
                if r_thg_ranges and current_label in r_thg_ranges:
                    # Use custom R-THg(%) range if available
                    mean_val = r_thg_ranges[current_label]["mean"]
                    range_val = r_thg_ranges[current_label]["range"]
                    lower_bound = mean_val - range_val
                    upper_bound = mean_val + range_val
                    
                    if not (lower_bound <= r_thg_value <= upper_bound):
                        causes.append("It might be an abnormal concentration")
                else:
                    # Use default threshold of 10%
                    if abs(r_thg_value) > 10:
                        causes.append("It might be an abnormal concentration")
                
                # Second condition: Check if 2 times StdErr values exceed thresholds
                stderr_exceeds_threshold = False
                thresholds = current_internal_precision_ranges[current_label]
                
                for stderr_col, threshold_value in thresholds.items():
                    if stderr_col in df.columns:
                        if 2 * row[stderr_col] > threshold_value:
                            stderr_exceeds_threshold = True
                            break
                    else:
                        print(f"Warning: Column '{stderr_col}' does not exist in the data")
                        stderr_exceeds_threshold = True
                        break
                
                if stderr_exceeds_threshold:
                    causes.append("Possible instrument instability")
                
                if not causes:
                    causes.append("Other reasons, retesting is recommended")
                
                df.at[index, 'Cause of the anomaly'] = '; '.join(causes)
                processed_count += 1

        # Save results
        output_path = os.path.join(table_path, "all_abnormal_data_processed.xlsx")
        df.to_excel(output_path, index=False)

        print(f"Abnormal data processing completed! Results saved to: {output_path}")
        print(f"\nProcessing statistics:")
        print(f"Total {processed_count} abnormal records processed")
        print(df['Cause of the anomaly'].value_counts())

        # Display sample of marked results
        print("\nSample of marked results:")
        marked_data = df[df['Cause of the anomaly'] != ''][['Label', 'R-THg(%)', 'Cause of the anomaly']]
        print(marked_data.head(10))

        # Display detailed breakdown by label
        print("\nDetailed breakdown by label:")
        for label in target_labels:
            label_data = df[df['Label'] == label]
            if not label_data.empty:
                print(f"\nLabel {label}:")
                print(label_data['Cause of the anomaly'].value_counts())
                
    except Exception as e:
        print(f"Error processing abnormal data: {e}")

def main():
    """Main function"""
    apply_lang_from_cli()
    # Get analysis mode
    analysis_mode = get_analysis_mode()
    
    # Get user input paths
    data_path, results_path = get_user_paths()
    
    # Create output directories
    fig_path, report_path, table_path = setup_directories(results_path)
    
    # Execute main analysis (this will handle all user input and both type1/type2 analysis)
    abnormal_file_path = main_analysis(data_path, results_path, fig_path, report_path, table_path, analysis_mode)
    
    # Execute abnormal data analysis
    if abnormal_file_path and os.path.exists(abnormal_file_path):
        process_abnormal_data(abnormal_file_path, table_path)
    else:
        print(f"No abnormal data file found or created.")
    
    print("\n" + "=" * 60)
    print("All processing completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()