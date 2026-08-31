import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

LANG = os.environ.get("HG_MC_AUTO_LANG", "zh").lower()

TRANSLATIONS = {
    "zh": {
        "title": "组合机器学习推断程序",
        "input_external": "请输入外部模型的输入数据文件路径（Excel 格式）: ",
        "input_external_dir": "请输入外部模型目录路径（保存外部模型 .pkl 文件的目录）: ",
        "input_internal": "请输入内部模型的输入数据文件路径（Excel 格式）: ",
        "input_internal_dir": "请输入内部模型目录路径（保存内部模型 .pkl 文件的目录）: ",
        "input_output": "请输入输出结果目录路径（建议放在 inference 文件夹中）: ",
        "path_empty": "路径不能为空，请重新输入。",
        "not_exist": "文件不存在: {path}，请重新输入。",
        "dir_missing": "目录不存在: {path}，请重新输入。",
        "excel_required": "请提供 Excel 文件（.xlsx 或 .xls 格式）",
    },
    "en": {
        "title": "Combined Machine Learning Inference Program",
        "input_external": "Please enter the input data file path for Normal/Abnormal classification (Excel format): ",
        "input_external_dir": "Please enter the EXTERNAL model directory path (where the external model .pkl files are saved): ",
        "input_internal": "Please enter the input data file path for Cause Analysis (Excel format): ",
        "input_internal_dir": "Please enter the INTERNAL model directory path (where the internal model .pkl files are saved): ",
        "input_output": "Please enter the output results directory path (It is recommended to place it in the inference folder) (): ",
        "path_empty": "Path cannot be empty, please re-enter.",
        "not_exist": "File does not exist: {path}, please re-enter.",
        "dir_missing": "Directory does not exist: {path}, please re-enter.",
        "excel_required": "Please provide an Excel file (.xlsx or .xls format)",
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
    
    print("\n--- External Model Input (Normal/Abnormal Classification) ---")
    while True:
        external_data_path = input(text("input_external")).strip()
        if not external_data_path:
            print(text("path_empty"))
            continue
        if not os.path.exists(external_data_path):
            print(text("not_exist", path=external_data_path))
            continue
        if not external_data_path.lower().endswith(('.xlsx', '.xls')):
            print(text("excel_required"))
            continue
        break
    
    print("\n--- External Model Directory ---")
    while True:
        external_model_dir = input(text("input_external_dir")).strip()
        if not external_model_dir:
            print(text("path_empty"))
            continue
        if not os.path.exists(external_model_dir):
            print(text("dir_missing", path=external_model_dir))
            continue
        break
    
    print("\n--- Internal Model Input (Cause Analysis) ---")
    while True:
        internal_data_path = input(text("input_internal")).strip()
        if not internal_data_path:
            print(text("path_empty"))
            continue
        if not os.path.exists(internal_data_path):
            print(text("not_exist", path=internal_data_path))
            continue
        if not internal_data_path.lower().endswith(('.xlsx', '.xls')):
            print(text("excel_required"))
            continue
        break
    
    print("\n--- Internal Model Directory ---")
    while True:
        internal_model_dir = input(text("input_internal_dir")).strip()
        if not internal_model_dir:
            print(text("path_empty"))
            continue
        if not os.path.exists(internal_model_dir):
            print(text("dir_missing", path=internal_model_dir))
            continue
        break
    
    while True:
        results_path = input(text("input_output")).strip()
        if not results_path:
            print(text("path_empty"))
            continue
        break
    
    return external_data_path, internal_data_path, external_model_dir, internal_model_dir, results_path

# ============================ SETUP AND UTILITY FUNCTIONS ============================

def check_model_directory(model_dir):
    """Check what files are in the model directory"""
    print(f"\n📁 Checking model directory: {model_dir}")
    if not os.path.exists(model_dir):
        print(f"❌ Directory does not exist: {model_dir}")
        return []
    
    files = os.listdir(model_dir)
    print(f"Found {len(files)} files:")
    for file in files:
        if file.endswith('.pkl'):
            print(f"  📄 {file}")
        elif file.endswith(('.xlsx', '.txt', '.png', '.json')):
            print(f"  📊 {file}")
        else:
            print(f"  📝 {file}")
    
    return files

def setup_directories(results_path):
    """Create output directory structure"""
    external_output_dir = os.path.join(results_path, "external_model_results")
    internal_output_dir = os.path.join(results_path, "internal_model_results")
    
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(external_output_dir, exist_ok=True)
    os.makedirs(internal_output_dir, exist_ok=True)
    
    return external_output_dir, internal_output_dir

# ============================ EXTERNAL MODEL FUNCTIONS ============================

def find_external_model_files(model_dir, pattern='top3_model_'):
    """Find all top model files for external model"""
    model_files = []
    for fn in os.listdir(model_dir):
        if fn.startswith(pattern) and fn.endswith('.pkl'):
            model_files.append(os.path.join(model_dir, fn))
    return sorted(model_files)

def load_external_artifacts(model_dir):
    """Load models and preprocessing artifacts for external model"""
    print(f"Loading models from directory: {model_dir}")
    
    # Define filename patterns
    EXTERNAL_MODEL_FILE_PATTERN = 'top3_model_'
    EXTERNAL_SCALER_FILE = os.path.join(model_dir, 'scaler.pkl')
    EXTERNAL_ENCODER_FILE = os.path.join(model_dir, 'target_encoder.pkl')
    EXTERNAL_PREPROCESSOR_FILE = os.path.join(model_dir, 'preprocessor.pkl')
    
    # Load all top models
    model_files = find_external_model_files(model_dir, EXTERNAL_MODEL_FILE_PATTERN)
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir} starting with {EXTERNAL_MODEL_FILE_PATTERN}")
    
    models = {}
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.pkl', '')
        models[model_name] = joblib.load(model_file)
        print(f"Loaded external model: {model_name}")
    
    # Load preprocessing artifacts
    scaler = None
    encoder = None
    preprocessor = None
    
    # Try to load preprocessor first
    if os.path.exists(EXTERNAL_PREPROCESSOR_FILE):
        preprocessor = joblib.load(EXTERNAL_PREPROCESSOR_FILE)
        print("✅ Loaded external preprocessor (includes scaler and encoder)")
    else:
        print("⚠️ Warning: No preprocessor.pkl found, trying to load scaler and encoder separately")
        # If no preprocessor, try to load separate scaler and encoder
        if os.path.exists(EXTERNAL_SCALER_FILE):
            scaler = joblib.load(EXTERNAL_SCALER_FILE)
            print("Loaded external scaler")
        else:
            print("Warning: No scaler.pkl found in model directory")
        
        if os.path.exists(EXTERNAL_ENCODER_FILE):
            encoder = joblib.load(EXTERNAL_ENCODER_FILE)
            print("Loaded external target encoder")
        else:
            print("Warning: No target_encoder.pkl found in model directory")
    
    return models, scaler, encoder, preprocessor

def preprocess_external_input(df, feature_columns):
    """Apply basic preprocessing consistent with external model training"""
    X = df.copy()
    
    # Check required columns: numerical features and Label column
    required_columns = ['d202(‰)', 'D199(‰)', 'D200(‰)', 'D201(‰)', 'Label']
    
    # Check if Label column exists
    if 'Label' not in X.columns:
        print("⚠️ Warning: 'Label' column not found in input data!")
        print("Adding 'Label' column with value 'sample' (default)")
        X['Label'] = 'sample'  # Default value
    
    # Ensure all required columns exist
    available_columns = [c for c in required_columns if c in X.columns]
    missing_columns = [c for c in required_columns if c not in X.columns]
    
    if missing_columns:
        print(f"❌ Error: Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    X = X[available_columns].copy()
    
    # Identify d-value columns, these should not have missing value imputation
    d_value_columns = ['d202(‰)', 'D199(‰)', 'D200(‰)', 'D201(‰)']
    
    # Check for missing values in d-value columns
    missing_d_values = X[d_value_columns].isnull().any(axis=1)
    if missing_d_values.any():
        print(f"⚠️ Warning: {missing_d_values.sum()} rows have missing d-values. These rows will be skipped from prediction.")
        # Store indices of rows with complete d-values
        complete_indices = ~missing_d_values
        X_complete = X[complete_indices].copy()
    else:
        X_complete = X.copy()
        complete_indices = pd.Series([True] * len(X), index=X.index)
    
    # If no rows have complete d-values, return empty
    if len(X_complete) == 0:
        print("❌ Error: No rows with complete d-values found")
        return X_complete, complete_indices, available_columns
    
    print(f"✅ Preprocessed data shape: {X_complete.shape}")
    print(f"✅ Available features: {available_columns}")
    
    return X_complete, complete_indices, available_columns

def run_external_ensemble_prediction(models, X_scaled):
    """Run prediction using all models and return ensemble results for external model"""
    all_predictions = []
    all_probabilities = []
    model_names = []
    
    for model_name, model in models.items():
        try:
            # Get predictions
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
            
            all_predictions.append(y_pred)
            if y_prob is not None:
                all_probabilities.append(y_prob)
            model_names.append(model_name)
            
        except Exception as e:
            print(f"Warning: Prediction failed for {model_name}: {e}")
            continue
    
    if not all_predictions:
        raise RuntimeError("All model predictions failed")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    
    # Ensemble prediction (majority voting)
    ensemble_pred = []
    for i in range(len(X_scaled)):
        votes = all_predictions[:, i]
        # Use mode (most frequent prediction)
        unique, counts = np.unique(votes, return_counts=True)
        ensemble_pred.append(unique[np.argmax(counts)])
    
    ensemble_pred = np.array(ensemble_pred)
    
    # Ensemble probabilities (average)
    ensemble_prob = None
    if all_probabilities:
        ensemble_prob = np.mean(all_probabilities, axis=0)
    
    # Individual model results for analysis
    individual_results = {}
    for i, model_name in enumerate(model_names):
        individual_results[model_name] = {
            'predictions': all_predictions[i],
            'probabilities': all_probabilities[i] if i < len(all_probabilities) else None
        }
    
    return ensemble_pred, ensemble_prob, individual_results

def run_external_individual_prediction(models, X_scaled):
    """Run prediction using all models and return individual results for external model (no voting)"""
    all_predictions = []
    all_probabilities = []
    model_names = []
    
    for model_name, model in models.items():
        try:
            # Get predictions
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
            
            all_predictions.append(y_pred)
            if y_prob is not None:
                all_probabilities.append(y_prob)
            model_names.append(model_name)
            
            print(f"Model {model_name} predictions completed")
            
        except Exception as e:
            print(f"Warning: Prediction failed for {model_name}: {e}")
            continue
    
    if not all_predictions:
        raise RuntimeError("All model predictions failed")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    
    # Individual model results for analysis
    individual_results = {}
    for i, model_name in enumerate(model_names):
        individual_results[model_name] = {
            'predictions': all_predictions[i],
            'probabilities': all_probabilities[i] if i < len(all_probabilities) else None
        }
    
    # For compatibility, we'll use the first model's predictions as the "main" prediction
    # but we'll store all individual predictions separately
    main_pred = all_predictions[0] if len(all_predictions) > 0 else None
    main_prob = all_probabilities[0] if len(all_probabilities) > 0 else None
    
    return main_pred, main_prob, individual_results

def calculate_normal_statistics(df, label_col='Predicted_Label', output_dir=None, model_name="Overall"):
    """Calculate statistics for Normal predictions by label type"""
    print(f"\nCalculating statistics for Normal predictions ({model_name})...")
    
    # First check the dataframe
    if df.empty:
        print("DataFrame is empty")
        return {}
    
    # Check if label_col exists
    if label_col not in df.columns:
        print(f"Error: '{label_col}' column not found in DataFrame")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    # Copy dataframe to avoid modifying original data
    df_processed = df.copy()
    
    # Debug info: check current values in label column
    print(f"Checking '{label_col}' column values...")
    unique_values = df_processed[label_col].dropna().unique()
    print(f"Unique values in '{label_col}': {unique_values}")
    
    # Convert labels: change 0/1 encoding to 'Normal'/'Abnormal'
    if set(unique_values).issubset({0, 1, '0', '1'}):
        print(f"Converting numeric labels to 'Normal'/'Abnormal'...")
        # Correction: 1->Normal, 0->Abnormal
        label_mapping = {0: 'Abnormal', 1: 'Normal', '0': 'Abnormal', '1': 'Normal'}
        df_processed[label_col] = df_processed[label_col].map(label_mapping)
        
        # Check converted values again
        new_unique = df_processed[label_col].dropna().unique()
        print(f"After conversion - Unique values: {new_unique}")
    
    # Check if there are 'Normal' labels
    normal_count = len(df_processed[df_processed[label_col] == 'Normal'])
    print(f"Found {normal_count} 'Normal' predictions")
    
    if normal_count == 0:
        print("No 'Normal' predictions found after conversion!")
        print("Current label distribution:")
        label_dist = df_processed[label_col].value_counts().to_dict()
        for label, count in label_dist.items():
            print(f"  {label}: {count}")
        return {}
    
    # Continue with existing logic
    # Define the labels we're interested in
    target_labels = ["3133", "3177", "8610", "sample"]
    d_value_columns = ['d202(‰)', 'D199(‰)', 'D200(‰)', 'D201(‰)']
    
    stats_results = {}
    
    for label in target_labels:
        # Filter data for this label and Normal predictions
        # Note: using processed df_processed instead of original df
        if 'Label' in df_processed.columns:
            label_normal_data = df_processed[(df_processed['Label'] == label) & (df_processed[label_col] == 'Normal')]
        else:
            # If no Label column, skip label-specific statistics
            print(f"Warning: 'Label' column not found, cannot filter by sample type")
            continue
        
        if len(label_normal_data) > 0:
            print(f"\n{label} - Normal predictions: {len(label_normal_data)} records")
            
            label_stats = {}
            for col in d_value_columns:
                if col in label_normal_data.columns:
                    values = label_normal_data[col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        two_sd = 2 * std_val
                        
                        label_stats[col] = {
                            'count': len(values),
                            'mean': mean_val,
                            'std': std_val,
                            '2sd': two_sd,
                            'range_lower': mean_val - two_sd,
                            'range_upper': mean_val + two_sd
                        }
                        
                        print(f"  {col}: {mean_val:.4f} ± {two_sd:.4f} (range: {mean_val - two_sd:.4f} to {mean_val + two_sd:.4f})")
                    else:
                        label_stats[col] = {'count': 0, 'mean': np.nan, 'std': np.nan, '2sd': np.nan}
                        print(f"  {col}: No data available")
                else:
                    label_stats[col] = {'count': 0, 'mean': np.nan, 'std': np.nan, '2sd': np.nan}
                    print(f"  {col}: Column not found")
            
            stats_results[label] = label_stats
        else:
            print(f"\n{label}: No Normal predictions found")
            stats_results[label] = {col: {'count': 0, 'mean': np.nan, 'std': np.nan, '2sd': np.nan} for col in d_value_columns}
    
    # Save statistics to report file if output_dir is provided
    if output_dir and stats_results:
        # Clean special characters from model_name to avoid filename issues
        safe_model_name = str(model_name).replace('/', '_').replace(':', '_').replace('\\', '_')
        report_file = os.path.join(output_dir, f"ml_prediction_statistics_report_{safe_model_name}.txt")
        try:
            with open(report_file, "w", encoding='utf-8') as f:
                f.write(f"External Model - Machine Learning Prediction Statistics Report ({model_name})\n")
                f.write("=" * 60 + "\n\n")
                f.write("Statistics for Normal predictions by label type\n")
                f.write("Format: Mean ± 2SD (Range: Lower to Upper)\n\n")
                
                for label in target_labels:
                    f.write(f"{label}:\n")
                    if label in stats_results:
                        for col in d_value_columns:
                            stats_data = stats_results[label][col]
                            if stats_data['count'] > 0:
                                f.write(f"  {col}: {stats_data['mean']:.4f} ± {stats_data['2sd']:.4f} ")
                                f.write(f"(range: {stats_data['range_lower']:.4f} to {stats_data['range_upper']:.4f})")
                                f.write(f" [n={stats_data['count']}]\n")
                            else:
                                f.write(f"  {col}: No Normal predictions available\n")
                    f.write("\n")
            
            print(f"\nStatistics report saved to: {report_file}")
        except Exception as e:
            print(f"Error saving statistics report: {e}")
    
    return stats_results

def create_individual_model_results(df, individual_results, encoder, complete_indices, output_dir):
    """Create separate result files for each individual model"""
    print("\nCreating individual model result files...")
    
    for model_name, results in individual_results.items():
        # Create a copy of the original dataframe
        model_df = df.copy()
        
        # Add this model's predictions
        pred_col = f'Predicted_Label_{model_name}'
        model_df[pred_col] = np.nan
        
        # Fill predictions only for rows with complete d-values
        if encoder is not None:
            try:
                individual_pred_labels = encoder.inverse_transform(results['predictions'])
                model_df.loc[complete_indices, pred_col] = individual_pred_labels
            except Exception as e:
                print(f"Warning: Could not inverse transform labels for {model_name}: {e}")
                model_df.loc[complete_indices, pred_col] = results['predictions']
        else:
            model_df.loc[complete_indices, pred_col] = results['predictions']
        
        # Add probabilities if available
        if results['probabilities'] is not None and results['probabilities'].shape[1] == 2:
            prob_normal_col = f'Prob_Normal_{model_name}'
            prob_abnormal_col = f'Prob_Abnormal_{model_name}'
            model_df[prob_normal_col] = np.nan
            model_df[prob_abnormal_col] = np.nan
            model_df.loc[complete_indices, prob_normal_col] = results['probabilities'][:, 0]
            model_df.loc[complete_indices, prob_abnormal_col] = results['probabilities'][:, 1]
        
        # Save individual model results
        model_file = os.path.join(output_dir, f'external_model_results_{model_name}.xlsx')
        model_df.to_excel(model_file, index=False)
        print(f"Individual model results saved to {model_file}")
        
        # Generate statistics for this individual model
        calculate_normal_statistics(model_df, pred_col, output_dir, model_name)

def filter_normal_predictions(df, label_col='Predicted_Label', 
                              prob_col='Predicted_Prob_Normal', 
                              min_confidence=0.9):
    """Filter Normal predictions to keep only high-confidence samples"""
    # 1. Get all samples predicted as Normal
    normal_data = df[df[label_col] == 'Normal'].copy()
    
    if len(normal_data) == 0:
        print("No Normal predictions found.")
        return normal_data
    
    print(f"Total Normal predictions: {len(normal_data)}")
    
    # 2. If there is confidence information, perform confidence filtering
    if prob_col in normal_data.columns:
        before_filter = len(normal_data)
        high_conf_normal = normal_data[normal_data[prob_col] > min_confidence]
        after_filter = len(high_conf_normal)
        print(f"High confidence (> {min_confidence}) Normal predictions: {after_filter}")
        print(f"High confidence percentage: {after_filter/before_filter*100:.1f}%")
        return high_conf_normal
    
    return normal_data

def calculate_filtered_statistics(df, label_col='Predicted_Label', 
                                  prob_col='Predicted_Prob_Normal',
                                  output_dir=None, model_name="Overall",
                                  min_confidence=0.9):
    """Calculate filtered statistical information"""
    print(f"\nCalculating statistics for high confidence Normal predictions ({model_name})...")
    print(f"Confidence threshold: > {min_confidence}")
    
    # Apply filtering
    filtered_normal_data = filter_normal_predictions(df, label_col, prob_col, min_confidence)
    
    if len(filtered_normal_data) == 0:
        print("No high confidence Normal predictions found.")
        return {}
    
    # Calculate statistics using filtered data
    return calculate_normal_statistics(filtered_normal_data, label_col, output_dir, f"High_Confidence_{model_name}")

def generate_high_confidence_report(high_conf_df, high_conf_stats, total_normal_count, output_dir):
    """Generate detailed report for high-confidence Normal predictions"""
    report_file = os.path.join(output_dir, 'high_confidence_normal_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("HIGH CONFIDENCE NORMAL PREDICTIONS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Confidence threshold: > 0.9\n")
        f.write(f"High confidence samples: {len(high_conf_df)}\n")
        f.write(f"Total Normal predictions: {total_normal_count}\n")
        f.write(f"High confidence percentage: {len(high_conf_df)/max(1, total_normal_count)*100:.1f}%\n")
        f.write("=" * 60 + "\n\n")
        
        # Statistics by Label type
        if 'Label' in high_conf_df.columns:
            f.write("DISTRIBUTION BY SAMPLE TYPE:\n")
            f.write("-" * 40 + "\n")
            label_counts = high_conf_df['Label'].value_counts()
            for label, count in label_counts.items():
                f.write(f"{label}: {count} samples\n")
            f.write("\n")
        
        # Confidence statistics
        if 'Predicted_Prob_Normal' in high_conf_df.columns:
            f.write("CONFIDENCE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Minimum confidence: {high_conf_df['Predicted_Prob_Normal'].min():.4f}\n")
            f.write(f"Maximum confidence: {high_conf_df['Predicted_Prob_Normal'].max():.4f}\n")
            f.write(f"Average confidence: {high_conf_df['Predicted_Prob_Normal'].mean():.4f}\n")
            f.write(f"Median confidence: {high_conf_df['Predicted_Prob_Normal'].median():.4f}\n")
            f.write(f"Standard deviation: {high_conf_df['Predicted_Prob_Normal'].std():.4f}\n")
            f.write("\n")
        
        # d-value statistics
        if high_conf_stats:
            f.write("D-VALUE STATISTICS BY SAMPLE TYPE:\n")
            f.write("-" * 40 + "\n")
            for label, stats in high_conf_stats.items():
                if isinstance(stats, dict):
                    f.write(f"\n{label}:\n")
                    for col, col_stats in stats.items():
                        if isinstance(col_stats, dict) and 'count' in col_stats and col_stats['count'] > 0:
                            f.write(f"  {col}: {col_stats['mean']:.4f} ± {col_stats['2sd']:.4f}\n")
                            f.write(f"    Range: {col_stats['range_lower']:.4f} to {col_stats['range_upper']:.4f}\n")
        
        # Data quality recommendations
        f.write("\n" + "=" * 60 + "\n")
        f.write("DATA QUALITY RECOMMENDATIONS:\n")
        f.write("=" * 60 + "\n")
        
        high_conf_ratio = len(high_conf_df) / max(1, total_normal_count)
        
        if high_conf_ratio >= 0.8:
            f.write("✅ Excellent: Over 80% of Normal predictions have high confidence.\n")
            f.write("   This indicates strong model performance and reliable predictions.\n")
        elif high_conf_ratio >= 0.6:
            f.write("⚠️ Good: 60-80% of Normal predictions have high confidence.\n")
            f.write("   Model performance is acceptable but could be improved.\n")
        elif high_conf_ratio >= 0.4:
            f.write("⚠️ Fair: 40-60% of Normal predictions have high confidence.\n")
            f.write("   Consider reviewing model performance and data quality.\n")
        else:
            f.write("❌ Poor: Less than 40% of Normal predictions have high confidence.\n")
            f.write("   Model confidence is low. Review training data and model parameters.\n")
    
    print(f"High confidence report saved to: {report_file}")

def run_external_inference(input_file, model_dir, output_dir, use_ensemble=True):
    """Run external model inference (Normal/Abnormal classification)"""
    print("\n" + "="*60)
    print("EXTERNAL MODEL INFERENCE - Normal/Abnormal Classification")
    print("="*60)
    
    print("Loading external model artifacts...")
    models, scaler, encoder, preprocessor = load_external_artifacts(model_dir)
    
    print(f"Reading input data from {input_file}...")
    df = pd.read_excel(input_file)
    
    # Note: now includes Label column
    feature_columns = ['d202(‰)', 'D199(‰)', 'D200(‰)', 'D201(‰)', 'Label']
    
    # Preprocess input data
    X, complete_indices, available_features = preprocess_external_input(df, feature_columns)

    # If no rows have complete d-values, exit
    if len(X) == 0:
        print("No rows with complete d-values found. Cannot proceed with prediction.")
        return None, None

    print(f"Preprocessed data shape: {X.shape}")
    print(f"Available features: {available_features}")

    # Apply preprocessor (priority) or separate scaler
    if preprocessor is not None:
        try:
            print("Applying preprocessor (includes scaling and one-hot encoding)...")
            X_preprocessed = preprocessor.transform(X)
            print(f"✅ Preprocessor applied. Transformed shape: {X_preprocessed.shape}")
        except Exception as e:
            raise RuntimeError(f"Preprocessor transform failed: {e}")
    elif scaler is not None:
        try:
            # Scale only numerical features
            numerical_features = ['d202(‰)', 'D199(‰)', 'D200(‰)', 'D201(‰)']
            X_numerical = X[numerical_features]
            X_numerical_scaled = scaler.transform(X_numerical)
            
            # Handle Label column (if categorical feature)
            if 'Label' in X.columns:
                # Simple label encoding (not OneHot)
                label_encoder = LabelEncoder()
                X_label_encoded = label_encoder.fit_transform(X['Label']).reshape(-1, 1)
                X_preprocessed = np.hstack([X_numerical_scaled, X_label_encoded])
            else:
                X_preprocessed = X_numerical_scaled
            
            print("Applied feature scaling (no one-hot encoding)")
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}")
    else:
        X_preprocessed = X.values
        print("⚠️ No scaler or preprocessor found, using raw features")

    print("Running external model prediction...")
    
    if use_ensemble:
        print("Using ensemble prediction with voting...")
        y_pred, y_prob, individual_results = run_external_ensemble_prediction(models, X_preprocessed)
        prediction_method = 'ensemble_voting'
    else:
        print("Using individual model predictions (no voting)...")
        y_pred, y_prob, individual_results = run_external_individual_prediction(models, X_preprocessed)
        prediction_method = 'individual_models'

    # Prepare output dataframe
    out = df.copy()
    
    # Initialize prediction columns with NaN
    out['Predicted_Label_encoded'] = np.nan
    out['Predicted_Label'] = np.nan
    out['Prediction_Method'] = np.nan
    
    # Add probability columns
    if y_prob is not None:
        if y_prob.shape[1] == 2:
            out['Predicted_Prob_Normal'] = np.nan
            out['Predicted_Prob_Abnormal'] = np.nan
        else:
            out['Predicted_Prob_Max'] = np.nan
    
    # Fill predictions only for rows with complete d-values
    out.loc[complete_indices, 'Predicted_Label_encoded'] = y_pred
    
    # Map encoded labels back to original labels
    if encoder is not None:
        try:
            # Print debugging information
            print(f"\n🔍 Label Decoding Debug Info:")
            print(f"Encoder type: {type(encoder)}")
            print(f"Encoder classes: {encoder.classes_}")
            print(f"Encoder class mapping: {dict(enumerate(encoder.classes_))}")
            
            # Check y_pred values
            unique_preds = np.unique(y_pred)
            print(f"Unique predictions (encoded): {unique_preds}")
            
            # Try to decode
            predicted_labels = encoder.inverse_transform(y_pred)
            print(f"Successfully decoded predictions")
            
            # Check decoded values
            unique_decoded = np.unique(predicted_labels)
            print(f"Unique decoded predictions: {unique_decoded}")
            
            out.loc[complete_indices, 'Predicted_Label'] = predicted_labels
            
        except Exception as e:
            print(f"❌ Warning: Could not inverse transform labels: {e}")
            print("Attempting manual mapping...")
            # Manual mapping: map according to encoder's classes_
            if hasattr(encoder, 'classes_') and len(encoder.classes_) == 2:
                print(f"Encoder classes found: {encoder.classes_}")
                
                # Find positions of Normal and Abnormal in classes_
                if 'Normal' in encoder.classes_ and 'Abnormal' in encoder.classes_:
                    normal_index = np.where(encoder.classes_ == 'Normal')[0][0]
                    abnormal_index = np.where(encoder.classes_ == 'Abnormal')[0][0]
                    print(f"Found: Normal at index {normal_index}, Abnormal at index {abnormal_index}")
                    
                    # According to your situation: 1->Normal, 0->Abnormal
                    label_mapping = {normal_index: 'Normal', abnormal_index: 'Abnormal'}
                    predicted_labels = np.array([label_mapping[p] for p in y_pred])
                else:
                    # If classes_ are not as expected, assume first is Normal, second is Abnormal
                    print("Warning: Could not find 'Normal'/'Abnormal' in encoder classes")
                    print(f"Using default mapping: 0->{encoder.classes_[0]}, 1->{encoder.classes_[1]}")
                    label_mapping = {0: encoder.classes_[0], 1: encoder.classes_[1]}
                    predicted_labels = np.array([label_mapping[p] for p in y_pred])
                
                out.loc[complete_indices, 'Predicted_Label'] = predicted_labels
            else:
                # According to your situation: 1->Normal, 0->Abnormal
                print("Using custom mapping: 1->Normal, 0->Abnormal")
                predicted_labels = np.where(y_pred == 1, 'Normal', 'Abnormal')
                out.loc[complete_indices, 'Predicted_Label'] = predicted_labels
    else:
        # According to your situation: 1->Normal, 0->Abnormal
        print("⚠️ No encoder found, using custom mapping: 1->Normal, 0->Abnormal")
        predicted_labels = np.where(y_pred == 1, 'Normal', 'Abnormal')
        out.loc[complete_indices, 'Predicted_Label'] = predicted_labels
    
    # Verify labels are correctly set
    print(f"\n✅ Label Verification:")
    unique_labels = out.loc[complete_indices, 'Predicted_Label'].unique()
    print(f"Unique labels in Predicted_Label: {unique_labels}")
    
    # Ensure labels are string type
    out['Predicted_Label'] = out['Predicted_Label'].astype(str)
    
    # Set prediction method
    out.loc[complete_indices, 'Prediction_Method'] = prediction_method

    # Fill probability columns
    if y_prob is not None:
        if y_prob.shape[1] == 2:
            # Determine which column corresponds to Normal probability
            # If encoder exists, check order of classes_
            if encoder is not None and hasattr(encoder, 'classes_'):
                if 'Normal' in encoder.classes_:
                    normal_idx = np.where(encoder.classes_ == 'Normal')[0][0]
                    abnormal_idx = np.where(encoder.classes_ == 'Abnormal')[0][0]
                    print(f"Probability mapping: column {normal_idx}->Normal, column {abnormal_idx}->Abnormal")
                    out.loc[complete_indices, 'Predicted_Prob_Normal'] = y_prob[:, normal_idx]
                    out.loc[complete_indices, 'Predicted_Prob_Abnormal'] = y_prob[:, abnormal_idx]
                else:
                    # Default: column 0 is Normal, column 1 is Abnormal
                    out.loc[complete_indices, 'Predicted_Prob_Normal'] = y_prob[:, 0]
                    out.loc[complete_indices, 'Predicted_Prob_Abnormal'] = y_prob[:, 1]
            else:
                # Default: column 0 is Normal, column 1 is Abnormal
                out.loc[complete_indices, 'Predicted_Prob_Normal'] = y_prob[:, 0]
                out.loc[complete_indices, 'Predicted_Prob_Abnormal'] = y_prob[:, 1]
        else:
            out.loc[complete_indices, 'Predicted_Prob_Max'] = y_prob.max(axis=1)

    # Add individual model predictions for analysis (always add them to the main file)
    for model_name, results in individual_results.items():
        # Add individual model predictions
        pred_col = f'Pred_{model_name}'
        out[pred_col] = np.nan
        out.loc[complete_indices, pred_col] = results['predictions']
        
        # Add individual model probabilities if available
        if results['probabilities'] is not None:
            if results['probabilities'].shape[1] == 2:
                prob_normal_col = f'Prob_Normal_{model_name}'
                prob_abnormal_col = f'Prob_Abnormal_{model_name}'
                out[prob_normal_col] = np.nan
                out[prob_abnormal_col] = np.nan
                
                # Determine which column corresponds to Normal
                if encoder is not None and hasattr(encoder, 'classes_'):
                    if 'Normal' in encoder.classes_:
                        normal_idx = np.where(encoder.classes_ == 'Normal')[0][0]
                        out.loc[complete_indices, prob_normal_col] = results['probabilities'][:, normal_idx]
                        out.loc[complete_indices, prob_abnormal_col] = results['probabilities'][:, 1-normal_idx]
                    else:
                        out.loc[complete_indices, prob_normal_col] = results['probabilities'][:, 0]
                        out.loc[complete_indices, prob_abnormal_col] = results['probabilities'][:, 1]
                else:
                    out.loc[complete_indices, prob_normal_col] = results['probabilities'][:, 0]
                    out.loc[complete_indices, prob_abnormal_col] = results['probabilities'][:, 1]
        
        # Also add decoded labels for individual models
        if encoder is not None:
            try:
                individual_pred_labels = encoder.inverse_transform(results['predictions'])
                pred_label_col = f'Pred_Label_{model_name}'
                out[pred_label_col] = np.nan
                out.loc[complete_indices, pred_label_col] = individual_pred_labels
            except Exception as e:
                print(f"Warning: Could not inverse transform labels for {model_name}: {e}")
                # Try manual decoding
                print(f"Attempting manual decoding for {model_name}...")
                if hasattr(encoder, 'classes_') and len(encoder.classes_) == 2:
                    label_mapping = {0: encoder.classes_[0], 1: encoder.classes_[1]}
                    individual_pred_labels = np.array([label_mapping[p] for p in results['predictions']])
                    out[pred_label_col] = np.nan
                    out.loc[complete_indices, pred_label_col] = individual_pred_labels
                else:
                    # Default mapping
                    individual_pred_labels = np.where(results['predictions'] == 0, 'Normal', 'Abnormal')
                    out[pred_label_col] = np.nan
                    out.loc[complete_indices, pred_label_col] = individual_pred_labels

    # Add a column to indicate which rows were used for prediction
    out['Used_For_Prediction'] = complete_indices
    
    # Check label format again before saving
    print(f"\n🔍 Final Check Before Saving:")
    if 'Predicted_Label' in out.columns:
        label_counts = out['Predicted_Label'].dropna().value_counts()
        print("Predicted Label Distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        # Correction: 1->Normal, 0->Abnormal
        if '0' in label_counts.index or '1' in label_counts.index:
            print("Found numeric labels, converting using 1->Normal, 0->Abnormal...")
            out['Predicted_Label'] = out['Predicted_Label'].replace({'0': 'Abnormal', '1': 'Normal', 0: 'Abnormal', 1: 'Normal'})
            
            # Print distribution again
            label_counts = out['Predicted_Label'].dropna().value_counts()
            print("After conversion:")
            for label, count in label_counts.items():
                print(f"  {label}: {count} samples")

    # ==================== High Confidence Filtering ====================
    print("\n" + "="*60)
    print("HIGH CONFIDENCE NORMAL PREDICTIONS (Confidence > 0.9)")
    print("="*60)
    
    # Filter high confidence Normal predictions
    if 'Predicted_Label' in out.columns and 'Predicted_Prob_Normal' in out.columns:
        high_confidence_normal = out[
            (out['Predicted_Label'] == 'Normal') & 
            (out['Predicted_Prob_Normal'] > 0.9)
        ]
        
        high_confidence_count = len(high_confidence_normal)
        total_normal = len(out[out['Predicted_Label'] == 'Normal'])
        
        print(f"High confidence Normal predictions (>0.9): {high_confidence_count} out of {total_normal} Normal predictions")
        print(f"Percentage: {high_confidence_count/max(1, total_normal)*100:.1f}%")
        
        # If there are high confidence Normal predictions, save separate file
        if high_confidence_count > 0:
            high_conf_file = os.path.join(output_dir, 'high_confidence_normal_predictions.xlsx')
            high_confidence_normal.to_excel(high_conf_file, index=False)
            print(f"High confidence Normal predictions saved to: {high_conf_file}")
            
            # Generate statistical report for high confidence data
            high_conf_stats = calculate_normal_statistics(
                high_confidence_normal, 
                'Predicted_Label', 
                output_dir, 
                "High_Confidence_Normal"
            )
            
            # Generate detailed report for high confidence data
            generate_high_confidence_report(
                high_confidence_normal, 
                high_conf_stats, 
                total_normal, 
                output_dir
            )
    else:
        print("No probability information available for confidence filtering")
    
    # ==================== Confidence Distribution Statistics ====================
    print("\n" + "="*60)
    print("PREDICTION CONFIDENCE DISTRIBUTION")
    print("="*60)
    
    if 'Predicted_Prob_Normal' in out.columns:
        # Define confidence intervals
        confidence_intervals = [
            (0.0, 0.5, "Very Low"),
            (0.5, 0.7, "Low"),
            (0.7, 0.85, "Medium"),
            (0.85, 0.95, "High"),
            (0.95, 1.01, "Very High")  # Use 1.01 to ensure inclusion of 1.0
        ]
        
        confidence_distribution = []
        
        # Statistics for each confidence interval
        for lower, upper, level in confidence_intervals:
            mask = (out['Predicted_Prob_Normal'] >= lower) & (out['Predicted_Prob_Normal'] < upper)
            count = mask.sum()
            confidence_distribution.append({
                'Level': level,
                'Range': f"{lower}-{upper}",
                'Count': count,
                'Percentage': f"{count/len(out)*100:.1f}%"
            })
        
        # Print confidence distribution
        for item in confidence_distribution:
            print(f"{item['Level']} confidence ({item['Range']}): {item['Count']} samples ({item['Percentage']})")
        
        # Save confidence distribution report
        confidence_report_file = os.path.join(output_dir, 'confidence_distribution_report.txt')
        with open(confidence_report_file, 'w', encoding='utf-8') as f:
            f.write("PREDICTION CONFIDENCE DISTRIBUTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total samples: {len(out)}\n\n")
            f.write("Confidence Level Distribution:\n")
            f.write("-" * 40 + "\n")
            for item in confidence_distribution:
                f.write(f"{item['Level']} ({item['Range']}): {item['Count']} samples ({item['Percentage']})\n")
        print(f"Confidence distribution report saved to: {confidence_report_file}")

    # Save main results
    out_file = os.path.join(output_dir, 'external_model_inference_results.xlsx')
    out.to_excel(out_file, index=False)
    print(f"\nExternal model inference results saved to {out_file}")
    
    # Print summary
    total_rows = len(df)
    predicted_rows = complete_indices.sum()
    skipped_rows = total_rows - predicted_rows
    
    print(f"\nExternal Model Prediction Summary:")
    print(f"Total rows: {total_rows}")
    print(f"Rows predicted: {predicted_rows}")
    print(f"Rows skipped (missing d-values): {skipped_rows}")
    
    if predicted_rows > 0:
        # Count predictions by class
        if encoder is not None:
            try:
                unique_preds, counts = np.unique(out.loc[complete_indices, 'Predicted_Label'], return_counts=True)
                print("\nOverall Prediction distribution:")
                for pred, count in zip(unique_preds, counts):
                    print(f"  {pred}: {count} ({count/predicted_rows*100:.1f}%)")
            except:
                pass
        
        # Count predictions by class for each model
        print("\nIndividual Model Prediction Distribution:")
        for model_name in individual_results.keys():
            if f'Pred_Label_{model_name}' in out.columns:
                model_preds = out.loc[complete_indices, f'Pred_Label_{model_name}'].dropna()
                unique_preds, counts = np.unique(model_preds, return_counts=True)
                print(f"\n{model_name}:")
                for pred, count in zip(unique_preds, counts):
                    print(f"  {pred}: {count} ({count/predicted_rows*100:.1f}%)")
    
    # Calculate and save statistics
    if use_ensemble:
        # Calculate original statistics
        stats_results = calculate_normal_statistics(out, 'Predicted_Label', output_dir, "Overall")
        
        # Calculate high confidence statistics
        high_conf_stats = calculate_filtered_statistics(out, 'Predicted_Label', 
                                                       'Predicted_Prob_Normal',
                                                       output_dir, "Overall")
    
    else:
        # For individual mode
        # Calculate overall statistics
        stats_results = calculate_normal_statistics(out, 'Predicted_Label', output_dir, "Overall")
        
        # Calculate high confidence statistics for each model
        for model_name in individual_results.keys():
            if f'Pred_Label_{model_name}' in out.columns and f'Prob_Normal_{model_name}' in out.columns:
                filtered_stats = calculate_filtered_statistics(
                    out, 
                    f'Pred_Label_{model_name}', 
                    f'Prob_Normal_{model_name}',
                    output_dir, 
                    model_name
                )
        
        # Create separate result files for each individual model
        create_individual_model_results(df, individual_results, encoder, complete_indices, output_dir)
    
    return out, stats_results

# ============================ INTERNAL MODEL FUNCTIONS ============================

class GeochemicalMLInference:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_models()
        
    def load_models(self):
        """Load trained models and preprocessing objects"""
        print("=== Loading Internal Trained Models ===")
        try:
            # Auto-detect model type based on file naming convention
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            
            # Look for basic or enhanced model files
            basic_model = [f for f in model_files if f.endswith('_basic.pkl')]
            enhanced_model = [f for f in model_files if f.endswith('_enhanced.pkl')]
            
            if enhanced_model:
                # Use enhanced model if available
                model_file = enhanced_model[0]
                model_type = "enhanced"
            elif basic_model:
                # Use basic model if enhanced not available
                model_file = basic_model[0]
                model_type = "basic"
            else:
                # Look for any model file
                model_candidates = [f for f in model_files if f.startswith('best_model') or f.startswith('model')]
                if model_candidates:
                    model_file = model_candidates[0]
                    model_type = "standard"
                else:
                    raise FileNotFoundError(f"No model files found in {self.model_dir}")
            
            print(f"Selected model: {model_file} ({model_type})")
            
            # Load the model
            self.model = joblib.load(os.path.join(self.model_dir, model_file))
            
            # Auto-detect and load other artifacts
            self.scaler = self._load_artifact('scaler')
            self.label_encoder = self._load_artifact('label_encoder')
            self.imputer = self._load_artifact('imputer')
            
            # Load feature list
            feature_file = self._find_feature_file()
            with open(feature_file, 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            
            print(f"✅ Internal models loaded successfully")
            print(f"✅ Model type: {type(self.model).__name__} ({model_type})")
            print(f"✅ Features: {len(self.features)} features")
            print(f"✅ Classes: {self.label_encoder.classes_}")
            
        except Exception as e:
            print(f"❌ Error loading internal models: {e}")
            raise
    
    def _load_artifact(self, artifact_name):
        """Load preprocessing artifacts with auto-detection"""
        # Try basic version first
        basic_file = os.path.join(self.model_dir, f'{artifact_name}_basic.pkl')
        enhanced_file = os.path.join(self.model_dir, f'{artifact_name}_enhanced.pkl')
        standard_file = os.path.join(self.model_dir, f'{artifact_name}.pkl')
        
        if os.path.exists(enhanced_file):
            print(f"  Loading {artifact_name} (enhanced version)")
            return joblib.load(enhanced_file)
        elif os.path.exists(basic_file):
            print(f"  Loading {artifact_name} (basic version)")
            return joblib.load(basic_file)
        elif os.path.exists(standard_file):
            print(f"  Loading {artifact_name} (standard version)")
            return joblib.load(standard_file)
        else:
            print(f"  ⚠️  Warning: {artifact_name} not found")
            return None
    
    def _find_feature_file(self):
        """Find feature list file"""
        files = os.listdir(self.model_dir)
        feature_files = [f for f in files if 'feature' in f.lower() and f.endswith('.txt')]
        
        if feature_files:
            # Prioritize enhanced version
            for f in feature_files:
                if 'enhanced' in f.lower():
                    return os.path.join(self.model_dir, f)
            # Otherwise use any feature file
            return os.path.join(self.model_dir, feature_files[0])
        else:
            raise FileNotFoundError(f"No feature list file found in {self.model_dir}")
    
    def filter_sample_data(self, df):
        """Filter out 'sample' data from prediction"""
        print("\n=== Filtering Data ===")
        
        # Check if we need to filter based on Label column or Cause of the anomaly
        if 'Label' in df.columns:
            # Filter out rows where Label is 'sample'
            original_count = len(df)
            df_filtered = df[df['Label'] != 'sample']
            filtered_count = len(df_filtered)
            print(f"Filtered out 'sample' data: {original_count - filtered_count} samples removed")
            print(f"Remaining samples for prediction: {filtered_count}")
        else:
            df_filtered = df
            print("No 'sample' data found to filter")
        
        return df_filtered
    
    def create_geochemical_features(self, df):
        """Create the same geochemical features used during training"""
        print("\n=== Creating Geochemical Features ===")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # 1. Relative Standard Deviation (RSD)
        if all(col in df_processed.columns for col in ['202Hg/198Hg', '201Hg/198Hg', '200Hg/198Hg', '199Hg/198Hg']):
            df_processed['RSD_202Hg'] = df_processed['StdErr(abs)202Hg/198Hg'] / df_processed['202Hg/198Hg'].replace(0, np.nan).abs()
            df_processed['RSD_201Hg'] = df_processed['StdErr(abs)201Hg/198Hg'] / df_processed['201Hg/198Hg'].replace(0, np.nan).abs()
            df_processed['RSD_200Hg'] = df_processed['StdErr(abs)200Hg/198Hg'] / df_processed['200Hg/198Hg'].replace(0, np.nan).abs()
            df_processed['RSD_199Hg'] = df_processed['StdErr(abs)199Hg/198Hg'] / df_processed['199Hg/198Hg'].replace(0, np.nan).abs()
            print("RSD features created")
        
        # 2. StdErr ratios
        df_processed['StdErr_ratio_199_202'] = df_processed['StdErr(abs)199Hg/198Hg'] / df_processed['StdErr(abs)202Hg/198Hg'].replace(0, np.nan)
        df_processed['StdErr_ratio_201_202'] = df_processed['StdErr(abs)201Hg/198Hg'] / df_processed['StdErr(abs)202Hg/198Hg'].replace(0, np.nan)
        print("StdErr ratio features created")
        
        # 3. Total noise level
        df_processed['Total_StdErr'] = (df_processed['StdErr(abs)202Hg/198Hg'] + 
                                      df_processed['StdErr(abs)201Hg/198Hg'] + 
                                      df_processed['StdErr(abs)200Hg/198Hg'] + 
                                      df_processed['StdErr(abs)199Hg/198Hg']) / 4
        print("Total StdErr feature created")
        
        # 4. Data quality score
        valid_rtHg = df_processed['R-THg(%)'].replace(0, np.nan)
        df_processed['Data_Quality_Score'] = 1 / (1 + df_processed['Total_StdErr'] * valid_rtHg)
        print("Data quality score feature created")
        
        # 5. Coefficient of variation features
        df_processed['CV_202Hg'] = df_processed['StdErr(abs)202Hg/198Hg'] / df_processed['Total_StdErr'].replace(0, np.nan)
        df_processed['CV_201Hg'] = df_processed['StdErr(abs)201Hg/198Hg'] / df_processed['Total_StdErr'].replace(0, np.nan)
        print("CV features created")
        
        return df_processed
    
    def encode_label_column(self, df):
        """Encode Label column to match training data encoding"""
        print("Encoding Label column...")
        
        # Check what Label values we have
        unique_labels = df['Label'].unique()
        print(f"Unique Label values: {unique_labels}")
        
        # Create a copy to avoid modifying original
        df_encoded = df.copy()
        
        # Since the model was trained with numeric Labels, we need to encode them
        # We'll use the same encoding as during training
        label_encoder = LabelEncoder()
        
        # Fit on all possible labels (including 'sample' if present)
        all_possible_labels = ['3133', '3177', '8610', 'sample']
        label_encoder.fit(all_possible_labels)
        
        # Transform the Label column
        df_encoded['Label_encoded'] = label_encoder.transform(df_encoded['Label'])
        
        print(f"Label encoding mapping:")
        for i, label in enumerate(label_encoder.classes_):
            print(f"  {label} -> {i}")
        
        return df_encoded, label_encoder
    
    def preprocess_data(self, df):
        """Preprocess data in the same way as during training

        - Create geochemical features
        - Encode Label
        - Build feature matrix according to self.features
        - Impute missing values using self.imputer; if imputer missing or not fitted,
          fit a new SimpleImputer on the provided X (fallback) and warn user.
        """
        print("\n=== Preprocessing Data ===")

        # Create features
        df_processed = self.create_geochemical_features(df)

        # Encode Label column to match training data (creates Label_encoded)
        df_encoded, self.label_encoder_local = self.encode_label_column(df_processed)

        # Select features expected by the model
        available_features = [f for f in self.features if f in df_encoded.columns or f == 'Label']
        missing_features = [f for f in self.features if f not in df_encoded.columns and f != 'Label']

        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")

        # Build feature matrix in expected order
        X = pd.DataFrame(index=df_encoded.index)
        for feature in self.features:
            if feature == 'Label':
                # Use encoded Label if available
                X['Label'] = df_encoded.get('Label_encoded', pd.Series([np.nan]*len(df_encoded), index=df_encoded.index))
            elif feature in df_encoded.columns:
                X[feature] = df_encoded[feature]
            else:
                X[feature] = np.nan
                print(f"⚠️  Feature '{feature}' not found, filling with NaN")

        print(f"✅ Final feature matrix shape: {X.shape}")
        print(f"✅ Features used: {len(self.features)}")

        # Imputation: ensure imputer exists and is fitted before transform
        print(f"Missing values before imputation: {X.isnull().sum().sum()}")

        # If no imputer provided, create one and fit on X (fallback)
        if getattr(self, 'imputer', None) is None:
            print("⚠️ Warning: No imputer found in model artifacts. Creating a SimpleImputer and fitting on input data (fallback).")
            self.imputer = SimpleImputer(strategy='median')
            # fit on X to compute medians for imputation
            self.imputer.fit(X)
        else:
            # If imputer exists, check whether it's fitted
            try:
                # check_is_fitted will raise NotFittedError if not fitted
                check_is_fitted(self.imputer)
            except (NotFittedError, AttributeError):
                print("⚠️ Warning: Loaded imputer is not fitted. Fitting imputer on input data (fallback).")
                try:
                    # fit on X to compute medians for imputation
                    self.imputer.fit(X)
                except Exception as e:
                    print(f"❌ Failed to fit imputer on input data: {e}")
                    raise

        # Now safe to transform
        try:
            X_imputed = self.imputer.transform(X)
        except Exception as e:
            print(f"❌ Imputer.transform failed: {e}")
            raise

        X_clean = pd.DataFrame(X_imputed, columns=self.features, index=X.index)
        print(f"Missing values after imputation: {X_clean.isnull().sum().sum()}")

        return X_clean, self.features
    
    def predict(self, df):
        """Make predictions on new data"""
        print("\n=== Making Internal Model Predictions ===")
        
        # First filter out 'sample' data
        df_filtered = self.filter_sample_data(df)
        
        if len(df_filtered) == 0:
            print("⚠️  No data remaining after filtering 'sample' data")
            return pd.DataFrame(), None
        
        # Preprocess data
        X, features_used = self.preprocess_data(df_filtered)
        
        # Prepare data for model (scale if needed)
        model_type = type(self.model).__name__
        if model_type in ['SVC', 'LogisticRegression']:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed) if hasattr(self.model, 'predict_proba') else None
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Index': df_filtered.index,
            'Predicted_Cause_of_Anomaly': predicted_labels
        })
        
        # Add confidence scores if available
        if probabilities is not None:
            max_probabilities = np.max(probabilities, axis=1)
            results['Prediction_Confidence'] = max_probabilities
            
            # Add probabilities for each class
            for i, class_name in enumerate(self.label_encoder.classes_):
                results[f'Probability_{class_name}'] = probabilities[:, i]
        
        # Add original features for analysis
        for feature in ['Label', 'R-THg(%)', 'Total_StdErr']:
            if feature in df_filtered.columns:
                results[feature] = df_filtered[feature].values
        
        print(f"✅ Internal model predictions completed for {len(df_filtered)} samples")
        print(f"✅ Features used: {len(features_used)}")
        
        return results, probabilities
    
    def analyze_predictions(self, results_df):
        """Analyze and summarize prediction results"""
        print("\n=== Internal Model Prediction Analysis ===")
        
        analysis_results = []
        
        # Basic statistics
        total_samples = len(results_df)
        analysis_results.append(f"Total samples predicted: {total_samples}")
        
        # Prediction distribution
        pred_distribution = results_df['Predicted_Cause_of_Anomaly'].value_counts()
        analysis_results.append("\nPrediction Distribution:")
        for cause, count in pred_distribution.items():
            percentage = (count / total_samples) * 100
            analysis_results.append(f"  {cause}: {count} samples ({percentage:.1f}%)")
        
        # Confidence analysis
        if 'Prediction_Confidence' in results_df.columns:
            avg_confidence = results_df['Prediction_Confidence'].mean()
            median_confidence = results_df['Prediction_Confidence'].median()
            low_confidence = (results_df['Prediction_Confidence'] < 0.7).sum()
            
            analysis_results.append(f"\nConfidence Analysis:")
            analysis_results.append(f"  Average Confidence: {avg_confidence:.3f}")
            analysis_results.append(f"  Median Confidence: {median_confidence:.3f}")
            analysis_results.append(f"  Low Confidence Predictions (<0.7): {low_confidence} ({low_confidence/total_samples*100:.1f}%)")
        
        # R-THg analysis by predicted cause
        if 'R-THg(%)' in results_df.columns:
            analysis_results.append(f"\nR-THg(%) Analysis by Predicted Cause:")
            for cause in results_df['Predicted_Cause_of_Anomaly'].unique():
                subset = results_df[results_df['Predicted_Cause_of_Anomaly'] == cause]
                abs_mean_rtHg = subset['R-THg(%)'].abs().mean()
                analysis_results.append(f"  {cause}: {abs_mean_rtHg:.3f}")
        
        return analysis_results
    
    def create_visualizations(self, results_df, output_dir):
        """Create Nature-style visualization charts for predictions"""
        print("\n=== Creating Internal Model Visualizations (Nature Style) ===")
        
        # Set Nature-style plotting parameters
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.linewidth': 1.0,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        # Define Nature color palette
        nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
                        '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
        
        # 1. Prediction Distribution - Nature-style Bar Chart
        plt.figure(figsize=(8, 6))
        pred_counts = results_df['Predicted_Cause_of_Anomaly'].value_counts()
        
        # Create bar chart
        bars = plt.bar(pred_counts.index, pred_counts.values, 
                      color=nature_colors[:len(pred_counts)],
                      edgecolor='black', linewidth=1.0, alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Predicted Cause', fontweight='bold')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.title('Distribution of Predicted Causes', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add light background
        plt.gca().set_facecolor('#F8F8F8')
        plt.gca().patch.set_alpha(0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'internal_prediction_distribution_nature.png'), dpi=600)
        plt.close()
        
        # 2. Confidence Distribution - Nature-style Histogram with KDE
        if 'Prediction_Confidence' in results_df.columns:
            plt.figure(figsize=(8, 6))
            
            # Create histogram
            n, bins, patches = plt.hist(results_df['Prediction_Confidence'], 
                                       bins=15, alpha=0.7, 
                                       color=nature_colors[0],
                                       edgecolor='black', linewidth=1.0,
                                       density=True)
            
            # Add KDE curve
            from scipy import stats
            kde = stats.gaussian_kde(results_df['Prediction_Confidence'])
            x_range = np.linspace(results_df['Prediction_Confidence'].min(), 
                                 results_df['Prediction_Confidence'].max(), 1000)
            plt.plot(x_range, kde(x_range), 'k-', linewidth=2, alpha=0.8)
            
            # Add vertical line for mean
            mean_conf = results_df['Prediction_Confidence'].mean()
            plt.axvline(mean_conf, color='red', linestyle='--', linewidth=1.5, 
                       alpha=0.7, label=f'Mean: {mean_conf:.3f}')
            
            plt.xlabel('Prediction Confidence', fontweight='bold')
            plt.ylabel('Density', fontweight='bold')
            plt.title('Distribution of Prediction Confidence Scores', 
                     fontsize=14, fontweight='bold', pad=15)
            plt.legend()
            plt.grid(alpha=0.3, linestyle='--')
            
            # Add light background
            plt.gca().set_facecolor('#F8F8F8')
            plt.gca().patch.set_alpha(0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'internal_confidence_distribution_nature.png'), dpi=600)
            plt.close()
        
        # 3. R-THg vs Prediction - Nature-style Box Plot
        if 'R-THg(%)' in results_df.columns and 'Predicted_Cause_of_Anomaly' in results_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            for cause in results_df['Predicted_Cause_of_Anomaly'].unique():
                subset = results_df[results_df['Predicted_Cause_of_Anomaly'] == cause]
                if len(subset) > 1:  # Need at least 2 points for box plot
                    box_data.append(subset['R-THg(%)'].abs().values)
                    box_labels.append(cause)
            
            if box_data:
                # Create box plot
                box = plt.boxplot(box_data, labels=box_labels, patch_artist=True,
                                 medianprops=dict(color='black', linewidth=2),
                                 boxprops=dict(linewidth=1.5),
                                 whiskerprops=dict(linewidth=1.5),
                                 capprops=dict(linewidth=1.5))
                
                # Color boxes
                for patch, color in zip(box['boxes'], nature_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add individual data points
                for i, data in enumerate(box_data):
                    x = np.random.normal(i+1, 0.04, size=len(data))
                    plt.scatter(x, data, alpha=0.6, color='gray', s=20, edgecolor='black', linewidth=0.5)
                
                plt.xlabel('Predicted Cause', fontweight='bold')
                plt.ylabel('|R-THg(%)|', fontweight='bold')
                plt.title('Distribution of |R-THg(%)| by Predicted Cause', 
                         fontsize=14, fontweight='bold', pad=15)
                plt.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add light background
                plt.gca().set_facecolor('#F8F8F8')
                plt.gca().patch.set_alpha(0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'internal_rtHg_by_prediction_nature.png'), dpi=600)
                plt.close()
        
        # 4. Confusion Matrix Style Visualization - Nature-style
        if 'Prediction_Confidence' in results_df.columns:
            plt.figure(figsize=(8, 6))
            
            # Create scatter plot of confidence vs R-THg
            scatter = plt.scatter(results_df['Prediction_Confidence'], 
                                 results_df['R-THg(%)'].abs() if 'R-THg(%)' in results_df.columns else range(len(results_df)),
                                 c=results_df['Prediction_Confidence'],
                                 cmap='viridis',
                                 s=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            plt.colorbar(scatter, label='Confidence Score')
            plt.xlabel('Prediction Confidence', fontweight='bold')
            plt.ylabel('|R-THg(%)|' if 'R-THg(%)' in results_df.columns else 'Sample Index', 
                      fontweight='bold')
            plt.title('Confidence vs |R-THg(%)| by Sample', 
                     fontsize=14, fontweight='bold', pad=15)
            plt.grid(alpha=0.3, linestyle='--')
            
            # Add light background
            plt.gca().set_facecolor('#F8F8F8')
            plt.gca().patch.set_alpha(0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'internal_confidence_vs_rtHg_nature.png'), dpi=600)
            plt.close()
        
        # 5. Stacked Bar Chart for Confidence Levels by Cause
        if 'Prediction_Confidence' in results_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Define confidence levels
            confidence_levels = {
                'High (≥0.8)': 0.8,
                'Medium (0.6-0.8)': 0.6,
                'Low (<0.6)': 0.0
            }
            
            # Prepare data for stacked bar chart
            causes = results_df['Predicted_Cause_of_Anomaly'].unique()
            bottom = np.zeros(len(causes))
            colors = ['#4E79A7', '#F28E2B', '#E15759']  # Nature colors for 3 levels
            
            for i, (level_name, threshold) in enumerate(confidence_levels.items()):
                level_counts = []
                for cause in causes:
                    subset = results_df[results_df['Predicted_Cause_of_Anomaly'] == cause]
                    if level_name == 'High (≥0.8)':
                        count = (subset['Prediction_Confidence'] >= threshold).sum()
                    elif level_name == 'Medium (0.6-0.8)':
                        count = ((subset['Prediction_Confidence'] >= threshold) & 
                                (subset['Prediction_Confidence'] < 0.8)).sum()
                    else:  # Low
                        count = (subset['Prediction_Confidence'] < 0.6).sum()
                    level_counts.append(count)
                
                bars = plt.bar(causes, level_counts, bottom=bottom, 
                              label=level_name, color=colors[i],
                              edgecolor='black', linewidth=1.0, alpha=0.8)
                bottom += np.array(level_counts)
            
            plt.xlabel('Predicted Cause', fontweight='bold')
            plt.ylabel('Number of Samples', fontweight='bold')
            plt.title('Confidence Level Distribution by Predicted Cause', 
                     fontsize=14, fontweight='bold', pad=15)
            plt.legend(title='Confidence Level', frameon=True, framealpha=0.8)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for i, cause in enumerate(causes):
                total = bottom[i]
                plt.text(i, total + 0.5, f'{int(total)}', 
                        ha='center', va='bottom', fontsize=9)
            
            # Add light background
            plt.gca().set_facecolor('#F8F8F8')
            plt.gca().patch.set_alpha(0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'internal_confidence_by_cause_nature.png'), dpi=600)
            plt.close()
        
        print("✅ Nature-style internal model visualizations created")
    
    def save_results(self, original_df, results_df, analysis_results, output_dir):
        """Save all results to files"""
        print("\n=== Saving Internal Model Results ===")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save detailed predictions
        results_df.to_excel(os.path.join(output_dir, 'internal_detailed_predictions.xlsx'), index=False)
        
        # 2. Save combined results (original data + predictions)
        # First, create a copy of the original data
        combined_df = original_df.copy()
        
        # Add prediction columns to the combined dataframe
        combined_df['ML_Predicted_Cause'] = ''
        combined_df['ML_Prediction_Confidence'] = np.nan
        
        # Fill in predictions for the filtered rows
        if not results_df.empty:
            for idx, row in results_df.iterrows():
                original_idx = row['Index']
                combined_df.loc[original_idx, 'ML_Predicted_Cause'] = row['Predicted_Cause_of_Anomaly']
                if 'Prediction_Confidence' in row:
                    combined_df.loc[original_idx, 'ML_Prediction_Confidence'] = row['Prediction_Confidence']
        
        # Mark 'sample' data
        if 'Label' in combined_df.columns:
            combined_df['ML_Data_Type'] = combined_df['Label'].apply(
                lambda x: 'sample' if x == 'sample' else 'predicted'
            )
        
        combined_df.to_excel(os.path.join(output_dir, 'internal_combined_results.xlsx'), index=False)
        
        # 3. Save analysis report
        with open(os.path.join(output_dir, 'internal_inference_analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write("INTERNAL GEOCHEMICAL MACHINE LEARNING INFERENCE REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model used: {type(self.model).__name__}\n")
            f.write(f"Total samples processed: {len(original_df)}\n")
            f.write(f"Samples predicted: {len(results_df)}\n")
            f.write(f"Samples filtered out (sample data): {len(original_df) - len(results_df)}\n")
            f.write("=" * 60 + "\n\n")
            
            for line in analysis_results:
                f.write(line + "\n")
        
        # 4. Create visualizations
        if not results_df.empty:
            self.create_visualizations(results_df, output_dir)
        
        print(f"✅ Internal model results saved to: {output_dir}")
        print(f"📊 Detailed predictions: internal_detailed_predictions.xlsx")
        print(f"📊 Combined results: internal_combined_results.xlsx")
        print(f"📊 Analysis report: internal_inference_analysis_report.txt")
        if not results_df.empty:
            print(f"📈 Nature-style visualization charts saved")

def run_internal_inference(input_file, model_dir, output_dir):
    """Run internal model inference (cause analysis)"""
    print("\n" + "="*60)
    print("INTERNAL MODEL INFERENCE - Cause Analysis")
    print("="*60)
    
    try:
        # Initialize inference engine with user-specified model directory
        print("=== Internal Geochemical ML Inference ===")
        inference_engine = GeochemicalMLInference(model_dir)
        
        # Load data
        print(f"\n=== Loading Internal Model Data ===")
        df = pd.read_excel(input_file)
        print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Make predictions
        results_df, probabilities = inference_engine.predict(df)
        
        if not results_df.empty:
            # Analyze predictions
            analysis_results = inference_engine.analyze_predictions(results_df)
            
            # Print analysis to console
            print("\n" + "="*50)
            print("INTERNAL MODEL PREDICTION SUMMARY")
            print("="*50)
            for line in analysis_results:
                print(line)
            
            # Save results
            inference_engine.save_results(df, results_df, analysis_results, output_dir)
            
            print("\n🎯 Internal Model Inference Complete!")
            print(f"📍 Results saved to: {output_dir}")
        else:
            print("⚠️  No internal model predictions made - all data was filtered out as 'sample' data")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Internal model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_prediction_mode():
    """Get user choice for prediction mode"""
    print("\n--- Prediction Mode Selection ---")
    print("Please choose the prediction mode for external model:")
    print("1. Ensemble voting (default) - Use majority voting from all models")
    print("2. Individual models - Show results for each model separately")
    
    while True:
        choice = input("Enter your choice (1 or 2, default is 1): ").strip()
        if not choice:
            return True  # Default to ensemble voting
        if choice in ['1', '2']:
            return choice == '1'  # True for ensemble, False for individual
        print("Invalid choice. Please enter 1 or 2.")

def main():
    """Main function"""
    apply_lang_from_cli()
    # Get user input paths
    external_data_path, internal_data_path, external_model_dir, internal_model_dir, results_path = get_user_paths()
    
    # Check model directories
    print("\n--- Checking External Model Directory ---")
    check_model_directory(external_model_dir)
    
    print("\n--- Checking Internal Model Directory ---")
    check_model_directory(internal_model_dir)
    
    # Get prediction mode
    use_ensemble = get_prediction_mode()
    
    # Create output directories
    external_output_dir, internal_output_dir = setup_directories(results_path)
    
    # Run external model inference
    try:
        external_results, external_statistics = run_external_inference(
            external_data_path, 
            external_model_dir, 
            external_output_dir, 
            use_ensemble=use_ensemble
        )
    except Exception as e:
        print(f"❌ External model inference failed: {e}")
        import traceback
        traceback.print_exc()
        external_results = None
    
    # Run internal model inference
    try:
        internal_results = run_internal_inference(internal_data_path, internal_model_dir, internal_output_dir)
    except Exception as e:
        print(f"❌ Internal model inference failed: {e}")
        internal_results = None
    
    print("\n" + "=" * 60)
    print("COMBINED MACHINE LEARNING INFERENCE COMPLETED!")
    print("=" * 60)
    print(f"📍 All results saved to: {results_path}")
    print(f"📊 External model results: {external_output_dir}")
    print(f"📊 Internal model results: {internal_output_dir}")
    print(f"🔧 Prediction mode: {'Ensemble Voting' if use_ensemble else 'Individual Models'}")

if __name__ == "__main__":
    main()