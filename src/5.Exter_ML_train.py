import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os
import joblib
from collections import Counter
from sklearn.inspection import permutation_importance
import json
from scipy import stats

LANG = os.environ.get("HG_MC_AUTO_LANG", "zh").lower()
TRANSLATIONS = {
    "zh": {
        "title": "机器学习分类程序",
        "input_data": "请输入输入数据文件路径（all_empirical_model_classified_data.xlsx）（Excel 格式）: ",
        "input_output": "请输入输出结果目录路径（推荐放在 Model_Exter 文件夹中）: ",
        "path_empty": "路径不能为空，请重新输入。",
        "not_exist": "文件不存在: {path}，请重新输入。",
        "excel_required": "请提供 Excel 文件（.xlsx 或 .xls 格式）",
    },
    "en": {
        "title": "Machine Learning Classification Program",
        "input_data": "Please enter the input data file path (all_empirical_model_classified_data.xlsx) (Excel format): ",
        "input_output": "Please enter the output results directory path (Recommended to be in the Model_Exter folder): ",
        "path_empty": "Path cannot be empty, please re-enter.",
        "not_exist": "File does not exist: {path}, please re-enter.",
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
        results_dir = input(text("input_output")).strip()
        if not results_dir:
            print(text("path_empty"))
            continue
        break
    
    os.makedirs(results_dir, exist_ok=True)
    return data_path, results_dir

# Set plotting fonts and style for Nature style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Nature style settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Seaborn style with Nature-like color palette
sns.set_style("white")
sns.set_palette("colorblind")

# Read data - MODIFIED: Include Label as feature
def load_and_preprocess_data(data_path):
    """Load and preprocess data - FIXED: Include Label as feature"""
    try:
        df = pd.read_excel(data_path)
        print("Data loaded successfully!")
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None
    
    # Remove rows where all target columns are missing
    target_cols = ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]
    df_clean = df.dropna(subset=target_cols, how='all')
    print(f"Shape after dropping rows with all target columns missing: {df_clean.shape}")
    
    # Define feature columns - MODIFIED: Include "Label" as a feature
    feature_columns = [
        "d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"
    ]
    
    # Check required feature columns
    missing_features = [col for col in feature_columns if col not in df_clean.columns]
    if missing_features:
        print(f"Warning: the following feature columns are missing: {missing_features}")
        feature_columns = [col for col in feature_columns if col in df_clean.columns]
    
    # Check Label column exists (now as a feature)
    if "Label" not in df_clean.columns:
        print("Warning: 'Label' column not found, cannot analyze sample types")
        df_clean["Label"] = "Unknown"
    
    # Add Label to feature columns
    feature_columns.append("Label")
    
    # Ensure target column exists
    if "Check Status" not in df_clean.columns:
        print("Error: target column 'Check Status' not found")
        return None
    
    # Select features and target variable
    X = df_clean[feature_columns].copy()
    y = df_clean["Check Status"].copy()
    
    # Store Label separately for analysis (optional)
    sample_labels = X["Label"].copy()
    
    # Data preprocessing
    print("Handling missing values...")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if col in ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]:
                # For numerical columns, fill with median
                X[col].fillna(X[col].median(), inplace=True)
            else:
                # For categorical columns like Label
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else "Unknown", inplace=True)
    
    # Prepare target variable
    y = y.dropna()
    valid_indices = y.index
    X = X.loc[valid_indices]
    sample_labels = sample_labels.loc[valid_indices]
    
    # Keep only samples labeled as Normal or Abnormal
    valid_labels = ["Normal", "Abnormal"]
    mask = y.isin(valid_labels)
    X = X[mask]
    y = y[mask]
    sample_labels = sample_labels[mask]
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    print(f"Final data shape: {X.shape}")
    print(f"Target distribution (encoded): {Counter(y_encoded)}")
    print(f"Class counts: Normal={sum(y_encoded==0)}, Abnormal={sum(y_encoded==1)}")
    print(f"Abnormal/Normal ratio: {sum(y_encoded==1)/sum(y_encoded==0):.2f}")
    
    # Print sample type distribution
    print("\nSample type distribution by class:")
    for label in ['Normal', 'Abnormal']:
        label_mask = (y == label)
        print(f"{label}:")
        print(sample_labels[label_mask].value_counts())
    
    return X, y_encoded, feature_columns, target_encoder, sample_labels

# NEW: Create preprocessing pipeline that handles categorical and numerical features separately
def create_preprocessing_pipeline():
    """Create preprocessing pipeline for mixed data types"""
    # Define numerical and categorical columns
    numerical_cols = ["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]
    categorical_cols = ["Label"]
    
    # Create transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

# Handle class imbalance methods
def get_balancing_methods():
    """Return different class balancing methods"""
    balancing_methods = {
        'Original': None,
        'SMOTE': SMOTE(random_state=42, k_neighbors=3),
        'ADASYN': ADASYN(random_state=42, n_neighbors=3),
        'SMOTEENN': SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=3)),
        'UnderSampling': RandomUnderSampler(random_state=42)
    }
    return balancing_methods

# Define machine learning models (with stronger anti-overfitting parameters)
def get_improved_models():
    """Define improved machine learning models with stronger anti-overfitting parameters"""
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced',
            C=0.1, penalty='l2', solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=50, max_depth=8,
            min_samples_split=15, min_samples_leaf=10, 
            max_features='sqrt', bootstrap=True, class_weight='balanced'
        ),
        'SVM': SVC(
            random_state=42, probability=True, class_weight='balanced',
            C=0.1, kernel='rbf', gamma='scale'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=6, min_samples_split=20,
            min_samples_leaf=10, class_weight='balanced',
            max_features=0.8
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=50, max_depth=4,
            min_samples_split=20, min_samples_leaf=10,
            learning_rate=0.05, subsample=0.8
        ),
        'Naive Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier(
            random_state=42, n_estimators=50, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=1,
            enable_categorical=False  # XGBoost doesn't directly handle categorical
        ),
        'Bagging RF': BaggingClassifier(
            RandomForestClassifier(
                n_estimators=30, max_depth=6, 
                min_samples_split=15, min_samples_leaf=7,
                random_state=42
            ),
            n_estimators=10, max_samples=0.8, random_state=42
        )
    }
    return models

# Model name abbreviations including sampling method
def abbreviate_model_name(full_name):
    """Abbreviate model names including sampling method for plotting"""
    # Split by underscore to separate model and sampling method
    parts = full_name.split('_')
    model_part = parts[0]
    sampling_part = '_'.join(parts[1:]) if len(parts) > 1 else 'Orig'
    
    # Model name mapping
    name_mapping = {
        'Logistic Regression': 'LR',
        'Random Forest': 'RF',
        'Support Vector Machine': 'SVM',
        'SVM': 'SVM',
        'K-Nearest Neighbors': 'KNN',
        'KNN': 'KNN',
        'Decision Tree': 'DT',
        'Gradient Boosting': 'GB',
        'Naive Bayes': 'NB',
        'XGBoost': 'XGB',
        'Bagging RF': 'BagRF'
    }
    
    # Sampling method abbreviation
    sampling_mapping = {
        'SMOTE': 'S',
        'ADASYN': 'A',
        'SMOTEENN': 'SE',
        'UnderSampling': 'U',
        'Original': 'Orig',
        'Orig': 'Orig'
    }
    
    # Get abbreviations
    model_abbr = name_mapping.get(model_part, model_part[:6])
    sampling_abbr = sampling_mapping.get(sampling_part, sampling_part[:3])
    
    # Combine with underscore
    if sampling_abbr == 'Orig':
        return f"{model_abbr}"
    else:
        return f"{model_abbr}_{sampling_abbr}"

# Cross-validation evaluation
def cross_validate_model(model, X, y, cv=5):
    """Evaluate model stability using cross-validation"""
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), 
                               scoring='f1_macro')
    return cv_scores.mean(), cv_scores.std()

# Evaluate model
def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name, balancing_method):
    """Evaluate single model performance"""
    # Training set predictions
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_train_pred)
    
    # Validation set predictions
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_val_pred)
    
    # Test set predictions
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_test_pred)
    
    # Calculate cross-validation scores
    cv_mean, cv_std = cross_validate_model(model, X_train, y_train)
    
    # Calculate various metrics
    results = {}
    
    for set_name, y_true, y_pred, y_prob in [
        ('Train', y_train, y_train_pred, y_train_prob),
        ('Validation', y_val, y_val_pred, y_val_prob),
        ('Test', y_test, y_test_pred, y_test_prob)
    ]:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate macro F1
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        results[set_name] = {
            'Accuracy': accuracy,
            'Precision_0': report['0']['precision'],
            'Recall_0': report['0']['recall'],
            'F1_0': report['0']['f1-score'],
            'Precision_1': report['1']['precision'],
            'Recall_1': report['1']['recall'],
            'F1_1': report['1']['f1-score'],
            'F1_Macro': f1_macro,
            'F1_Weighted': f1_weighted,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    results['Balancing_Method'] = balancing_method
    results['CV_Mean'] = cv_mean
    results['CV_Std'] = cv_std
    results['Overfitting_Gap'] = results['Train']['Accuracy'] - results['Test']['Accuracy']
    
    return results

def get_feature_importance(model, X, y, feature_names, model_name):
    """Get feature importance for any model type"""
    try:
        # Method 1: Built-in feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        # Method 2: Coefficients for linear models
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 2:
                # Multi-class classification
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        # Method 3: Permutation importance (fallback for all models)
        else:
            try:
                # This works for all scikit-learn models
                perm_importance = permutation_importance(model, X, y, 
                                                       n_repeats=10, 
                                                       random_state=42)
                importances = perm_importance.importances_mean
                return pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            except:
                # Last resort: equal importance
                print(f"Warning: Using equal importance for {model_name}")
                return pd.DataFrame({
                    'feature': feature_names,
                    'importance': [1.0/len(feature_names)] * len(feature_names)
                })
    except Exception as e:
        print(f"Error getting feature importance for {model_name}: {e}")
        return pd.DataFrame({
            'feature': feature_names,
            'importance': [1.0/len(feature_names)] * len(feature_names)
        })

# NEW: Calculate feature statistics for specific features
def calculate_feature_stats(data_df, features_of_interest, group_mask=None):
    """Calculate Mean ± 2SD for specific features"""
    stats_dict = {}
    
    for feature in features_of_interest:
        if feature in data_df.columns:
            if group_mask is not None:
                data = data_df.loc[group_mask, feature]
            else:
                data = data_df[feature]
            
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                n_samples = len(data)
                stats_dict[feature] = {
                    'mean': mean_val,
                    'std': std_val,
                    'mean_plus_2sd': mean_val + 2 * std_val,
                    'mean_minus_2sd': mean_val - 2 * std_val,
                    'n': n_samples
                }
            else:
                stats_dict[feature] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'mean_plus_2sd': np.nan,
                    'mean_minus_2sd': np.nan,
                    'n': 0
                }
    
    return stats_dict

# MODIFIED: Save top3 model predictions with confidence scores
def save_top3_predictions(top3_models_info, X_test, y_test, sample_labels, target_encoder, results_dir, original_features_df=None):
    """Save predictions from top 3 models with confidence scores"""
    predictions_data = []
    
    # Ensure X_test has index for alignment
    if hasattr(X_test, 'index'):
        test_indices = X_test.index
    else:
        test_indices = range(len(X_test))
    
    # Create base DataFrame
    base_df = pd.DataFrame({
        'Index': test_indices,
        'True_Label': target_encoder.inverse_transform(y_test),
        'True_Label_Code': y_test
    })
    
    # Add sample labels if available
    if sample_labels is not None:
        if hasattr(sample_labels, 'values'):
            base_df['Sample_Type'] = sample_labels.values
        else:
            base_df['Sample_Type'] = sample_labels
    
    # Add all original features, not just specific mappings
    if original_features_df is not None:
        # Add all original features to base_df
        for col in original_features_df.columns:
            base_df[f'Original_{col}'] = original_features_df[col].values
        
        print(f"Added original features to predictions: {list(original_features_df.columns)}")
    
    for i, (model_name, model_instance) in enumerate(top3_models_info, 1):
        # Make predictions
        y_pred = model_instance.predict(X_test)
        
        # Get confidence scores (probability)
        if hasattr(model_instance, "predict_proba"):
            y_prob = model_instance.predict_proba(X_test)
            confidence = np.max(y_prob, axis=1)  # Confidence is max probability
            prob_class_0 = y_prob[:, 0]  # Probability for class 0 (Normal)
            prob_class_1 = y_prob[:, 1]  # Probability for class 1 (Abnormal)
        else:
            confidence = np.ones(len(y_pred))
            prob_class_0 = np.where(y_pred == 0, 1, 0)
            prob_class_1 = np.where(y_pred == 1, 1, 0)
        
        # Add predictions to base DataFrame
        base_df[f'Model_{i}_Name'] = abbreviate_model_name(model_name)
        base_df[f'Model_{i}_Prediction'] = target_encoder.inverse_transform(y_pred)
        base_df[f'Model_{i}_Prediction_Code'] = y_pred
        base_df[f'Model_{i}_Confidence'] = confidence
        base_df[f'Model_{i}_Prob_Normal'] = prob_class_0
        base_df[f'Model_{i}_Prob_Abnormal'] = prob_class_1
        
        # Build column list dynamically based on what columns exist
        columns_list = ['Index', 'True_Label', 'True_Label_Code']
        if 'Sample_Type' in base_df.columns:
            columns_list.append('Sample_Type')
        
        columns_list.extend([
            f'Model_{i}_Name', f'Model_{i}_Prediction', f'Model_{i}_Prediction_Code',
            f'Model_{i}_Confidence', f'Model_{i}_Prob_Normal', f'Model_{i}_Prob_Abnormal'
        ])
        
        # Add original feature columns (if any)
        if original_features_df is not None:
            for col in original_features_df.columns:
                col_name = f'Original_{col}'
                if col_name in base_df.columns:
                    columns_list.append(col_name)
        
        model_pred_df = base_df[columns_list].copy()
        
        predictions_data.append((model_name, model_pred_df))
        
        # Save individual model predictions
        clean_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        pred_path = os.path.join(results_dir, f'top3_model_{i}_{clean_name}_predictions.xlsx')
        model_pred_df.to_excel(pred_path, index=False)
        print(f"Saved predictions for model {i} ({abbreviate_model_name(model_name)}) to: {pred_path}")
    
    # Save combined predictions with all models
    combined_path = os.path.join(results_dir, 'top3_models_combined_predictions.xlsx')
    base_df.to_excel(combined_path, index=False)
    print(f"Saved combined predictions to: {combined_path}")
    
    # Calculate agreement between models
    if len(predictions_data) >= 2:
        pred_columns = [f'Model_{i}_Prediction_Code' for i in range(1, len(predictions_data) + 1)]
        base_df['Agreement_Count'] = base_df[pred_columns].apply(lambda x: len(set(x)), axis=1)
        
        # Calculate majority vote
        if len(predictions_data) == 3:
            base_df['Majority_Vote'] = base_df[pred_columns].mode(axis=1)[0]
            base_df['Majority_Prediction'] = target_encoder.inverse_transform(base_df['Majority_Vote'].astype(int))
    
    # Save updated combined predictions
    base_df.to_excel(combined_path, index=False)
    
    return predictions_data, base_df

# Evaluate model by sample type
def evaluate_by_sample_type(model, X_test, y_test, sample_labels, target_encoder):
    """Evaluate model performance by sample type"""
    if sample_labels is None:
        return {}
    
    results_by_type = {}
    unique_labels = sample_labels.unique()
    
    # Reset the index to ensure alignment
    X_test_reset = X_test.reset_index(drop=True) if hasattr(X_test, 'reset_index') else X_test
    y_test_reset = pd.Series(y_test).reset_index(drop=True) if hasattr(y_test, 'index') else y_test
    sample_labels_reset = sample_labels.reset_index(drop=True) if hasattr(sample_labels, 'reset_index') else sample_labels
    
    for label in unique_labels:
        # Use the reset index
        mask = (sample_labels_reset == label)
        
        if mask.sum() == 0:  # If there are no samples of this type
            continue
        
        X_subset = X_test_reset[mask]
        y_subset = y_test_reset[mask]
        
        if len(X_subset) == 0:
            continue
        
        y_pred = model.predict(X_subset)
        
        # Calculate metrics
        accuracy = accuracy_score(y_subset, y_pred)
        f1_macro = f1_score(y_subset, y_pred, average='macro')
        
        # Safely calculate the recall for each category
        recall_normal = 0
        recall_abnormal = 0
        
        if sum(y_subset==0) > 0:
            recall_normal = recall_score(y_subset[y_subset==0], y_pred[y_subset==0], average='macro', zero_division=0)
        if sum(y_subset==1) > 0:
            recall_abnormal = recall_score(y_subset[y_subset==1], y_pred[y_subset==1], average='macro', zero_division=0)
        
        results_by_type[label] = {
            'n_samples': len(X_subset),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'recall_normal': recall_normal,
            'recall_abnormal': recall_abnormal
        }
    
    return results_by_type

# NEW: Function to plot prediction confidence in Nature style
def plot_prediction_confidence(predictions_df, results_dir):
    """Plot prediction confidence distribution in Nature style"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    # Colors for Nature style
    nature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, ax in enumerate(axes):
        model_idx = i + 1
        conf_col = f'Model_{model_idx}_Confidence'
        pred_col = f'Model_{model_idx}_Prediction'
        
        if conf_col in predictions_df.columns and pred_col in predictions_df.columns:
            # Get confidence values by prediction class
            normal_mask = predictions_df[pred_col] == 'Normal'
            abnormal_mask = predictions_df[pred_col] == 'Abnormal'
            
            # Plot distribution
            if normal_mask.any():
                ax.hist(predictions_df.loc[normal_mask, conf_col], bins=20, alpha=0.7, 
                       color=nature_colors[0], label='Normal', density=True)
            if abnormal_mask.any():
                ax.hist(predictions_df.loc[abnormal_mask, conf_col], bins=20, alpha=0.7, 
                       color=nature_colors[1], label='Abnormal', density=True)
            
            # Add vertical line at 0.5 threshold
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Prediction Confidence', fontsize=15, fontweight='bold')
            ax.set_ylabel('Density', fontsize=15, fontweight='bold')
            ax.set_title(f'Model {model_idx}: {predictions_df[f"Model_{model_idx}_Name"].iloc[0]}', 
                        fontsize=15, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12, width=2)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Move the legend to the middle-right position
            ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.1, 0.8))
            
            ax.grid(True, alpha=0.3)
            
            # Calculate and display statistics
            total_samples = len(predictions_df)
            high_conf = (predictions_df[conf_col] >= 0.8).sum()
            low_conf = (predictions_df[conf_col] <= 0.5).sum()
            
            stats_text = f"Total: {total_samples}\nHigh conf (≥0.8): {high_conf} ({high_conf/total_samples*100:.1f}%)\nLow conf (≤0.5): {low_conf} ({low_conf/total_samples*100:.1f}%)"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Model {model_idx}', fontsize=12)
            ax.axis('off')
    
    fig.suptitle('Prediction Confidence Distribution by Model', fontsize=16, fontweight='bold')
    fig_path = os.path.join(results_dir, 'prediction_confidence_distribution.png')
    fig.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confidence distribution plot to: {fig_path}")
    
    # Create another figure for confidence vs accuracy
    if 'True_Label' in predictions_df.columns:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        
        for i, ax in enumerate(axes2):
            model_idx = i + 1
            conf_col = f'Model_{model_idx}_Confidence'
            pred_col = f'Model_{model_idx}_Prediction'
            
            if conf_col in predictions_df.columns:
                # Calculate accuracy per confidence bin
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                accuracies = []
                conf_means = []
                
                for j in range(len(bins)-1):
                    mask = (predictions_df[conf_col] >= bins[j]) & (predictions_df[conf_col] < bins[j+1])
                    if mask.sum() > 0:
                        correct = (predictions_df.loc[mask, pred_col] == predictions_df.loc[mask, 'True_Label']).sum()
                        accuracies.append(correct / mask.sum())
                        conf_means.append(predictions_df.loc[mask, conf_col].mean())
                
                if accuracies:
                    ax.scatter(conf_means, accuracies, s=50, alpha=0.7, color=nature_colors[i])
                    ax.plot(conf_means, accuracies, '--', alpha=0.5, color=nature_colors[i])
                    ax.set_xlabel('Mean Confidence', fontsize=11)
                    ax.set_ylabel('Accuracy', fontsize=11)
                    ax.set_title(f'Model {model_idx}: Confidence vs Accuracy', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    
                    # Add trend line
                    if len(conf_means) > 1:
                        z = np.polyfit(conf_means, accuracies, 1)
                        p = np.poly1d(z)
                        ax.plot(conf_means, p(conf_means), "r--", alpha=0.5, linewidth=2)
            else:
                ax.axis('off')
        
        fig2.suptitle('Prediction Confidence vs Accuracy', fontsize=14, fontweight='bold')
        fig2_path = os.path.join(results_dir, 'confidence_vs_accuracy.png')
        fig2.savefig(fig2_path, dpi=600, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved confidence vs accuracy plot to: {fig2_path}")

# MODIFIED: Plot results with improved visualization and fixed Figure 4
def plot_improved_results(all_results, X_test, y_test, top3_models_info, target_encoder, sample_labels, results_dir):
    """Plot more concise and organized result charts with improved labeling"""
    import matplotlib.font_manager as fm
    
    # Improved color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#3F7CAC', '#5D737E', '#6A8EAE', '#FF6B6B', '#4ECDC4']
    
    # Get feature names (after preprocessing, names might be different)
    if hasattr(X_test, 'columns'):
        feature_names = X_test.columns.tolist()
    else:
        feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    
    # First sort by Test F1_Macro, take top 8 for comparison
    sorted_models = sorted(all_results.items(),
                          key=lambda x: x[1]['Test']['F1_Macro'],
                          reverse=True)
    top_models = sorted_models[:8]
    model_names = [m[0] for m in top_models]

    # Prepare metric arrays
    f1_scores = [all_results[n]['Test']['F1_Macro'] for n in model_names]
    cv_stds = [all_results[n]['CV_Std'] for n in model_names]
    recall_normal = [all_results[n]['Test']['Recall_0'] for n in model_names]
    overfit_gaps = [all_results[n]['Overfitting_Gap'] for n in model_names]

    # -------- Figure 1: F1 vs Normal Class Recall (left and right plots) --------
    fig1, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig1.suptitle('Model overall comparison (Top 8)', fontsize=18, fontweight='bold')
    
    # Add subplot labels
    ax_left.text(-0.1, 1.05, '(a)', transform=ax_left.transAxes, fontsize=16, fontweight='bold', va='top')
    ax_right.text(-0.1, 1.05, '(b)', transform=ax_right.transAxes, fontsize=16, fontweight='bold', va='top')

    x = np.arange(len(model_names))
    bar_width = 0.6

    # Left: F1 (with CV std error)
    bars = ax_left.bar(x, f1_scores, bar_width, yerr=cv_stds, 
                      color=colors[0], capsize=8, alpha=0.9, edgecolor='black', linewidth=1)
    ax_left.set_xticks(x)
    abbr_names = [abbreviate_model_name(n) for n in model_names]
    ax_left.set_xticklabels(abbr_names, rotation=30, ha='right', fontsize=14, fontweight='bold')
    ax_left.set_ylim(0.5, 1)
    ax_left.set_ylabel('Test Macro F1', fontsize=16, fontweight='bold')
    ax_left.set_title('Model Performance (F1 Score)', fontsize=16, fontweight='bold')
    ax_left.grid(axis='y', alpha=0.3)
    ax_left.tick_params(axis='y', labelsize=14)
    for label in ax_left.get_yticklabels():
        label.set_fontweight('bold')
    for i, b in enumerate(bars):
        ax_left.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.08, f'{f1_scores[i]:.3f}',
                     ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Right: Normal class recall
    bars2 = ax_right.bar(x, recall_normal, bar_width, 
                        color=colors[2], alpha=0.9, edgecolor='black', linewidth=1)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(abbr_names, rotation=30, ha='right', fontsize=14, fontweight='bold')
    ax_right.set_ylim(0.5, 1)
    ax_right.set_ylabel('Normal class Recall', fontsize=16, fontweight='bold')
    ax_right.set_title('Normal Class Recall', fontsize=16, fontweight='bold')
    ax_right.axhline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_right.grid(axis='y', alpha=0.3)
    ax_right.tick_params(axis='y', labelsize=14)
    for label in ax_right.get_yticklabels():
        label.set_fontweight('bold')
    for i, b in enumerate(bars2):
        ax_right.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.08, f'{recall_normal[i]:.3f}',
                      ha='center', va='bottom', fontsize=14, fontweight='bold')

    fig1_path = os.path.join(results_dir, 'summary_top8_f1_recall.png')
    fig1.savefig(fig1_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # -------- Figure 2: Top 3 models confusion matrices and feature importance --------
    fig2, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
    fig2.suptitle('Top 3 Models: Confusion Matrices and Feature Importance', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Add subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    label_idx = 0
    for i in range(3):
        for j in range(2):
            axes[i, j].text(-0.1, 1.05, subplot_labels[label_idx], transform=axes[i, j].transAxes, 
                           fontsize=16, fontweight='bold', va='top')
            label_idx += 1
    
    for idx, (model_name, model_instance) in enumerate(top3_models_info):
        results = all_results[model_name]
        
        # Confusion Matrix (left column)
        ax_cm = axes[idx, 0]
        y_test_pred = results['Test']['y_pred']
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Create heatmap with better styling
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=target_encoder.classes_, 
                    yticklabels=target_encoder.classes_,
                    cbar=False, annot_kws={'size': 14, 'weight': 'bold'},
                    square=True, linewidths=1, linecolor='gray')
        ax_cm.tick_params(axis='x', labelsize=14)
        ax_cm.tick_params(axis='y', labelsize=14)
        # Bold tick labels
        for label in ax_cm.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax_cm.get_yticklabels():
            label.set_fontweight('bold')
        ax_cm.set_xlabel('Predicted', fontsize=14, fontweight='bold')
        ax_cm.set_ylabel('Actual', fontsize=14, fontweight='bold')
        
        # Add model name and metrics as title
        abbr_name = abbreviate_model_name(model_name)
        ax_cm.set_title(f'{abbr_name} - Confusion Matrix\nTest F1={results["Test"]["F1_Macro"]:.3f}', 
                       fontsize=14, fontweight='bold')
        
        # Feature Importance (right column)
        ax_fi = axes[idx, 1]
        
        # Get feature importance
        feat_df = get_feature_importance(model_instance, X_test, y_test, feature_names, model_name)
        
        if feat_df is not None and not feat_df.empty:
            # Take top 8 features
            top_features = feat_df.head(8).copy()
            top_features = top_features.sort_values('importance', ascending=True)  # For horizontal bar
            
            # Create horizontal bar plot
            bars = ax_fi.barh(range(len(top_features)), top_features['importance'], 
                            color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1)
            
            # Customize plot
            ax_fi.set_yticks(range(len(top_features)))
            ax_fi.set_yticklabels(top_features['feature'], fontsize=14, fontweight='bold')
            ax_fi.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
            ax_fi.set_title(f'{abbr_name} - Feature Importance', 
                          fontsize=14, fontweight='bold')
            ax_fi.grid(axis='x', alpha=0.3)
            # Adjust the x-axis scale of the feature importance chart - enlarge and bold
            ax_fi.tick_params(axis='x', labelsize=13)  # Enlarge X-axis tick labels
            for label in ax_fi.get_xticklabels():
                label.set_fontweight('bold')  # Bold the x-axis tick labels
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, top_features['importance'])):
                ax_fi.text(imp + 0.01 * max(top_features['importance']), 
                          i, f'{imp:.3f}', va='center', fontsize=13, fontweight='bold')
        else:
            ax_fi.axis('off')
            ax_fi.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                      ha='center', va='center', fontsize=12, fontweight='bold')
    
    fig2_path = os.path.join(results_dir, 'top3_confusion_matrices_and_feature_importance.png')
    fig2.savefig(fig2_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    # -------- Figure 3: ROC comparison (Top8) + Stability scatter --------
    top8 = sorted_models[:8]
    fig3, (ax_roc, ax_stab) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig3.suptitle('ROC Comparison (Top8) and Stability Analysis', fontsize=18, fontweight='bold', y=1.05)
    
    # Add subplot labels
    ax_roc.text(-0.1, 1.05, '(a)', transform=ax_roc.transAxes, fontsize=16, fontweight='bold', va='top')
    ax_stab.text(-0.1, 1.05, '(b)', transform=ax_stab.transAxes, fontsize=16, fontweight='bold', va='top')

    # ROC curve (Top 8)
    roc_info = []
    for i, (mname, res) in enumerate(top8):
        y_prob = res['Test'].get('y_prob', None)
        y_true = res['Test']['y_true']
        if y_prob is not None and len(np.unique(y_prob)) > 1:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                roc_info.append({
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc,
                    'name': mname,
                    'color': colors[i % len(colors)]
                })
            except Exception as e:
                print(f"Warning: Could not compute ROC for {mname}: {e}")
                continue
    
    # Plot ROC curves with better styling
    for info in roc_info:
        short_name = abbreviate_model_name(info['name'])
        ax_roc.plot(info['fpr'], info['tpr'], 
                   label=f'{short_name} (AUC={info["auc"]:.3f})',
                   color=info['color'], linewidth=2, alpha=0.8)
    
    ax_roc.plot([0, 0.1], [0.9, 1], 'k--', alpha=0.6, linewidth=2)
    ax_roc.set_xlim([0.0, 0.1])
    ax_roc.set_ylim([0.9, 1.0])
    ax_roc.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    # Enlarge and bold the tick labels
    ax_roc.tick_params(axis='both', labelsize=14)  # Enlarge scale labels
    for label in ax_roc.get_xticklabels():
        label.set_fontweight('bold')  # Bold the x-axis tick labels
    for label in ax_roc.get_yticklabels():
        label.set_fontweight('bold')  # Bold the y-axis tick labels
    ax_roc.set_title('ROC Curves (Top 8 Models)', fontsize=16, fontweight='bold')
    roc_legend = ax_roc.legend(fontsize=15, loc='lower right', framealpha=0.9, frameon=True, edgecolor='black')
    # ax_roc.legend.set_title('Models', prop={'size': 12, 'weight': 'bold'})
    for text in roc_legend.get_texts():
        text.set_fontweight('bold')
    ax_roc.grid(alpha=0.3)

    # Stability scatter plot: x=overfitting gap, y=Test F1 (Top 8)
    top8_model_names = [n for n, _ in top8]
    top8_of_gaps = [all_results[n]['Overfitting_Gap'] for n in top8_model_names]
    top8_test_f1s = [all_results[n]['Test']['F1_Macro'] for n in top8_model_names]
    top8_cv_stds = [all_results[n]['CV_Std'] for n in top8_model_names]
    
    # Create scatter plot with color gradient and LARGER size based on CV std
    cv_stds_scaled = [std * 800 for std in top8_cv_stds]  # Scale for visibility
    base_size = 150  # Increased base size for better visibility
    
    sc = ax_stab.scatter(top8_of_gaps, top8_test_f1s, 
                        c='red',
                        s=[base_size + size for size in cv_stds_scaled],
                        alpha=0.8, 
                        edgecolors='black', 
                        linewidth=1,  # Thicker border
                        zorder=5)  # Ensure points are on top
    
    # Replace the numbers representing each point with the model names, enlarge the font, and place it to the right of the point
    for i, (x_val, y_val, model_name) in enumerate(zip(top8_of_gaps, top8_test_f1s, top8_model_names)):
        short_model_name = abbreviate_model_name(model_name)
        font_size = 14
        font_weight = 'bold'
        text_color = 'black'
        ax_stab.text(x_val + 0.004, y_val, short_model_name,
                    ha='left', va='center',
                    fontsize=font_size, 
                    fontweight=font_weight,
                    color=text_color,
                    zorder=10)
    
    # Add reference lines
    ax_stab.axvline(0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax_stab.axvline(0.05, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='5% overfit')
    ax_stab.axvline(0.1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='10% overfit')
    
    # Shade regions
    ax_stab.axvspan(-0.1, 0.05, alpha=0.1, color='green', label='Good stability')
    ax_stab.axvspan(0.05, 0.1, alpha=0.1, color='orange', label='Moderate overfit')
    ax_stab.axvspan(0.1, 0.2, alpha=0.1, color='red', label='High overfit')
    
    ax_stab.set_xlabel('Overfitting Gap (Train - Test Accuracy)', fontsize=16, fontweight='bold')
    ax_stab.set_ylabel('Test Macro F1 Score', fontsize=16, fontweight='bold')
    # Enlarge and bold the tick labels
    ax_stab.tick_params(axis='both', labelsize=14)  # Enlarge scale labels
    for label in ax_stab.get_xticklabels():
        label.set_fontweight('bold')  # Bold the x-axis tick labels
    for label in ax_stab.get_yticklabels():
        label.set_fontweight('bold')  # Bold the y-axis tick labels
    ax_stab.set_title('Model Stability Analysis', fontsize=15, fontweight='bold')
    ax_stab.grid(alpha=0.3, zorder=0)
    # Enlarge and bold the legend of the stability chart (small box)
    stab_legend = ax_stab.legend(fontsize=15, loc='upper right', framealpha=0.9, 
                            frameon=True, edgecolor='black')
    # Set the legend title (if needed)
    # stab_legend.set_title('Legend', prop={'size': 12, 'weight': 'bold'})
    # Bold the text in the legend
    for text in stab_legend.get_texts():
        text.set_fontweight('bold')
    
    # Adjust x-axis limits
    x_min, x_max = min(top8_of_gaps), max(top8_of_gaps)
    ax_stab.set_xlim([x_min - 0.05, x_max + 0.15]) 
    
    fig3_path = os.path.join(results_dir, 'roc_and_stability_top8.png')
    fig3.savefig(fig3_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig3)

    # -------- FIXED Figure 4: Performance by Sample Type --------
    # Check if sample_labels is available and has data
    if sample_labels is not None and len(sample_labels) > 0:
        # Ensure we have the test set sample labels
        if hasattr(sample_labels, '__len__') and len(sample_labels) == len(y_test):
            fig4, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            fig4.suptitle('Model Performance by Sample Type', fontsize=18, fontweight='bold', y=1.05)
            
            # Add subplot labels
            for i, ax in enumerate(axes):
                ax.text(-0.1, 1.05, f'({chr(97+i)})', transform=ax.transAxes, 
                       fontsize=16, fontweight='bold', va='top')
            
            # Evaluate top 3 models by sample type
            for idx, (model_name, model_instance) in enumerate(top3_models_info[:3]):
                ax = axes[idx]
                
                # Get performance by sample type
                results_by_type = evaluate_by_sample_type(
                    model_instance, X_test, y_test, sample_labels, target_encoder
                )
                
                if results_by_type and len(results_by_type) > 0:
                    # Prepare data for plotting
                    sample_types = list(results_by_type.keys())
                    accuracies = [results_by_type[t]['accuracy'] for t in sample_types]
                    f1_scores_type = [results_by_type[t]['f1_macro'] for t in sample_types]
                    
                    x = np.arange(len(sample_types))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
                                  color=colors[0], alpha=0.8)
                    bars2 = ax.bar(x + width/2, f1_scores_type, width, label='F1 Macro', 
                                  color=colors[2], alpha=0.8)
                    
                    ax.set_xlabel('Sample Type', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
                    ax.set_title(f'{abbreviate_model_name(model_name)}\nby Sample Type', 
                               fontsize=13, fontweight='bold')
                    ax.set_xticks(x)
                    
                    # Truncate long sample type names if necessary
                    truncated_labels = [label[:15] + '...' if len(str(label)) > 15 else str(label) 
                                       for label in sample_types]
                    ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
                    ax.set_ylim(0, 1.05)
                    ax.legend(loc='lower right')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add value labels
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No sample type data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{abbreviate_model_name(model_name)}', fontsize=13)
                    ax.axis('off')
            
            fig4_path = os.path.join(results_dir, 'performance_by_sample_type.png')
            fig4.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig4)
            print(f"- {os.path.basename(fig4_path)}")
        else:
            print("Warning: sample_labels length doesn't match y_test length, skipping Figure 4")
    else:
        print("No sample labels available, skipping Figure 4")

    print("Improved plots generated:")
    print(f"- {os.path.basename(fig1_path)}")
    print(f"- {os.path.basename(fig2_path)}")
    print(f"- {os.path.basename(fig3_path)}")

# MODIFIED: Generate detailed report with feature statistics
def generate_improved_report(all_results, top3_models_info, target_encoder, X_test, y_test, sample_labels, 
                            results_dir, predictions_df=None, original_features_df=None):
    """Generate a detailed performance report in English with feature statistics."""
    report_file = os.path.join(results_dir, 'machine_learning_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Machine Learning Classification Report\n")
        f.write("=" * 70 + "\n\n")
        
        # Data summary
        f.write("DATA SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total samples: {len(y_test)}\n")
        f.write(f"Features used: {X_test.shape[1]}\n")
        if hasattr(X_test, 'columns'):
            f.write(f"Features: {list(X_test.columns)}\n\n")
        
        if sample_labels is not None:
            f.write("Sample type distribution in test set:\n")
            sample_counts = pd.Series(sample_labels).value_counts()
            for label, count in sample_counts.items():
                f.write(f"  {label}: {count} samples\n")
            f.write("\n")

        # NEW: Feature statistics for Normal samples in test set
        f.write("FEATURE STATISTICS FOR NORMAL SAMPLES IN TEST SET:\n")
        f.write("-" * 50 + "\n")
        
        # Check if we have predictions DataFrame
        if predictions_df is not None and 'True_Label' in predictions_df.columns:
            # Get Normal samples from true labels
            normal_mask = predictions_df['True_Label'] == 'Normal'
            n_normal = normal_mask.sum()
            f.write(f"Number of Normal samples in test set: {n_normal}\n\n")
            
            # Feature column name (change to the original feature name)
            original_feature_columns = ['d202(‰)', 'D199(‰)', 'D200(‰)', 'D201(‰)']
            
            # Check if there is a Sample_Type column for grouping
            if 'Sample_Type' in predictions_df.columns:
                # Get all sample types
                sample_types = predictions_df['Sample_Type'].unique()
                
                for sample_type in sample_types:
                    if pd.isna(sample_type):
                        continue
                    
                    # Obtain Normal samples of this sample type
                    type_normal_mask = normal_mask & (predictions_df['Sample_Type'] == sample_type)
                    n_type_normal = type_normal_mask.sum()
                    
                    if n_type_normal > 0:
                        f.write(f"Sample Type: {sample_type} (Normal samples: {n_type_normal})\n")
                        
                        # Count each feature
                        for feature in original_feature_columns:
                            original_col = f'Original_{feature}'
                            if original_col in predictions_df.columns:
                                data = predictions_df.loc[type_normal_mask, original_col].dropna()
                                if len(data) > 0:
                                    mean_val = data.mean()
                                    std_val = data.std()
                                    f.write(f"  {feature}: {mean_val:.4f} ± {2*std_val:.4f} (n={len(data)})\n")
                                else:
                                    f.write(f"  {feature}: No data available\n")
                        f.write("\n")
            else:
                # If there is no Sample_Type, count all Normal samples
                f.write("All Normal samples (no sample type grouping):\n")
                for feature in original_feature_columns:
                    original_col = f'Original_{feature}'
                    if original_col in predictions_df.columns:
                        data = predictions_df.loc[normal_mask, original_col].dropna()
                        if len(data) > 0:
                            mean_val = data.mean()
                            std_val = data.std()
                            f.write(f"{feature} for all Normal samples (n={len(data)}):\n")
                            f.write(f"  Mean ± 2SD: {mean_val:.4f} ± {2*std_val:.4f}\n")
                            f.write(f"  Range: [{data.min():.4f}, {data.max():.4f}]\n\n")
                        else:
                            f.write(f"{feature}: No data available for Normal samples\n\n")
            
            # NEW: Statistics for each top3 model's predicted Normal samples
            f.write("\nFEATURE STATISTICS FOR MODEL-PREDICTED NORMAL SAMPLES:\n")
            f.write("-" * 50 + "\n")
            
            for i, (model_name, _) in enumerate(top3_models_info, 1):
                pred_col = f'Model_{i}_Prediction'
                if pred_col in predictions_df.columns:
                    # Get samples predicted as Normal by this model
                    pred_normal_mask = predictions_df[pred_col] == 'Normal'
                    n_pred_normal = pred_normal_mask.sum()
                    
                    f.write(f"\nModel {i} ({abbreviate_model_name(model_name)}) predicted Normal samples: {n_pred_normal}\n")
                    
                    # Check if there is a Sample_Type column
                    if 'Sample_Type' in predictions_df.columns and 'Sample_Type' in predictions_df.columns:
                        # Grouped statistics by sample type
                        for sample_type in predictions_df['Sample_Type'].unique():
                            if pd.isna(sample_type):
                                continue
                            
                            type_mask = pred_normal_mask & (predictions_df['Sample_Type'] == sample_type)
                            n_type_pred = type_mask.sum()
                            
                            if n_type_pred > 0:
                                f.write(f"  Sample Type {sample_type} (n={n_type_pred}):\n")
                                
                                # Count each feature
                                for feature in original_feature_columns:
                                    original_col = f'Original_{feature}'
                                    if original_col in predictions_df.columns:
                                        data = predictions_df.loc[type_mask, original_col].dropna()
                                        if len(data) > 0:
                                            mean_val = data.mean()
                                            std_val = data.std()
                                            f.write(f"    {feature}: {mean_val:.4f} ± {2*std_val:.4f}\n")
                        f.write("\n")
                    else:
                        # If there is no Sample_Type, count all samples predicted as Normal.
                        f.write("  All predicted Normal samples:\n")
                        for feature in original_feature_columns:
                            original_col = f'Original_{feature}'
                            if original_col in predictions_df.columns:
                                data = predictions_df.loc[pred_normal_mask, original_col].dropna()
                                if len(data) > 0:
                                    mean_val = data.mean()
                                    std_val = data.std()
                                    f.write(f"    {feature}: {mean_val:.4f} ± {2*std_val:.4f} (n={len(data)})\n")
                    
                    # Calculate accuracy for Normal predictions
                    if n_pred_normal > 0:
                        correct_normal = ((predictions_df.loc[pred_normal_mask, 'True_Label'] == 'Normal')).sum()
                        normal_accuracy = correct_normal / n_pred_normal
                        f.write(f"\n  Accuracy for Normal predictions: {normal_accuracy:.4f} ({correct_normal}/{n_pred_normal})\n")
        else:
            f.write("No prediction data available for feature statistics.\n")
        
        f.write("\n" + "=" * 70 + "\n\n")

        # Overall Summary
        f.write("OVERALL SUMMARY:\n")
        f.write("-" * 50 + "\n")
        
        # Show top 3 models
        f.write("Top 3 models:\n")
        for i, (model_name, model_instance) in enumerate(top3_models_info, 1):
            results = all_results[model_name]
            abbr_name = abbreviate_model_name(model_name)
            f.write(f"{i}. {model_name} ({abbr_name})\n")
            f.write(f"   - Test Macro F1: {results['Test']['F1_Macro']:.4f}\n")
            f.write(f"   - Normal Recall: {results['Test']['Recall_0']:.4f}\n")
            f.write(f"   - Abnormal Recall: {results['Test']['Recall_1']:.4f}\n")
            f.write(f"   - CV Std: {results['CV_Std']:.4f}\n")
            f.write(f"   - Balancing method: {results['Balancing_Method']}\n")
        
        # Best model details
        best_model_name = top3_models_info[0][0]
        best_model_instance = top3_models_info[0][1]
        best_results = all_results[best_model_name]
        best_f1_macro = best_results['Test']['F1_Macro']
        best_accuracy = best_results['Test']['Accuracy']
        best_recall_0 = best_results['Test']['Recall_0']
        balancing_method = best_results['Balancing_Method']
        cv_std = best_results['CV_Std']

        f.write(f"\nBest model: {best_model_name}\n")
        f.write(f"Balancing method: {balancing_method}\n")
        f.write(f"Test accuracy: {best_accuracy:.4f}\n")
        f.write(f"Test macro F1: {best_f1_macro:.4f}\n")
        f.write(f"Normal class recall: {best_recall_0:.4f}\n")
        f.write(f"Cross-validation std: {cv_std:.4f}\n")
        f.write(f"Overfitting gap: {best_results['Overfitting_Gap']:.4f}\n\n")

        # Performance tier
        f.write("Performance tier:\n")
        if best_f1_macro > 0.8 and best_recall_0 > 0.7 and cv_std < 0.05:
            f.write("✅ Excellent - strong performance and stability; ready for deployment\n")
        elif best_f1_macro > 0.75 and best_recall_0 > 0.65:
            f.write("⚠️ Good - reasonable performance; consider further tuning\n")
        elif best_f1_macro > 0.65:
            f.write("⚠️ Fair - performance needs improvement\n")
        else:
            f.write("❌ Poor - significant improvements required\n")
        
        # Sample type analysis (if available)
        if sample_labels is not None:
            f.write("\nPERFORMANCE BY SAMPLE TYPE (Best Model):\n")
            f.write("-" * 50 + "\n")
            
            results_by_type = evaluate_by_sample_type(
                best_model_instance, X_test, y_test, sample_labels, target_encoder
            )
            
            for label, metrics in results_by_type.items():
                f.write(f"\nSample type: {label}\n")
                f.write(f"  Number of samples: {metrics['n_samples']}\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1 Macro: {metrics['f1_macro']:.4f}\n")
                f.write(f"  Recall (Normal): {metrics['recall_normal']:.4f}\n")
                f.write(f"  Recall (Abnormal): {metrics['recall_abnormal']:.4f}\n")
            
            # Identify problematic sample types
            problematic_types = [
                label for label, metrics in results_by_type.items()
                if metrics['accuracy'] < 0.7 or metrics['f1_macro'] < 0.6
            ]
            
            if problematic_types:
                f.write("\n⚠️ Problematic sample types (performance < 70%):\n")
                for label in problematic_types:
                    f.write(f"  - {label}\n")
                f.write("\nSuggestions:\n")
                f.write("  1. Check if these sample types have sufficient training data\n")
                f.write("  2. Consider if these sample types need separate models\n")
                f.write("  3. Review feature distributions for these sample types\n")

        # Detailed per-model results
        f.write("\nDETAILED MODEL COMPARISON (sorted by Test macro F1):\n")
        f.write("-" * 50 + "\n")

        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['Test']['F1_Macro'], reverse=True)
        for model_name, results in sorted_models:
            abbr_name = abbreviate_model_name(model_name)
            f.write(f"\n{model_name} ({abbr_name}):\n")
            f.write(f"  Balancing method: {results['Balancing_Method']}\n")
            f.write(f"  Test - Accuracy: {results['Test']['Accuracy']:.4f}, Macro F1: {results['Test']['F1_Macro']:.4f}\n")
            f.write(f"  Normal - Recall: {results['Test']['Recall_0']:.4f}, F1: {results['Test']['F1_0']:.4f}\n")
            f.write(f"  Abnormal - Recall: {results['Test']['Recall_1']:.4f}, F1: {results['Test']['F1_1']:.4f}\n")
            f.write(f"  Cross-validation: {results['CV_Mean']:.4f} ± {results['CV_Std']:.4f}\n")
            f.write(f"  Overfitting gap: {results['Overfitting_Gap']:.4f}\n")

        # Suggestions
        f.write("\nSUGGESTIONS FOR IMPROVEMENT:\n")
        f.write("-" * 50 + "\n")
        if best_recall_0 < 0.6:
            f.write("1. Low recall for the Normal class. Suggestions:\n")
            f.write("   - Adjust classification threshold\n")
            f.write("   - Use cost-sensitive learning\n")
            f.write("   - Increase Normal-class samples if possible\n")

        if best_results['Overfitting_Gap'] > 0.1:
            f.write("2. Possible overfitting. Suggestions:\n")
            f.write("   - Increase regularization\n")
            f.write("   - Reduce feature dimensionality\n")
            f.write("   - Use simpler models\n")

        if cv_std > 0.08:
            f.write("3. Stability concerns. Suggestions:\n")
            f.write("   - Increase dataset size\n")
            f.write("   - Use ensemble methods\n")
            f.write("   - Tune model hyperparameters\n")
        
        # Specific suggestions for sample type issues
        if sample_labels is not None:
            f.write("\n4. For sample type-specific performance issues:\n")
            f.write("   - Ensure balanced representation of all sample types in training\n")
            f.write("   - Consider stratified sampling during train-test split\n")
            f.write("   - If certain sample types have different patterns, consider separate models\n")
            f.write("   - Add interaction terms between features and sample types if needed\n")
        
        # Feature distribution suggestions 
        f.write("\n5. Feature distribution observations:\n")
        if predictions_df is not None:
            original_feature_columns = ['d202(‰)', 'D199(‰)', 'D200(‰)', 'D201(‰)']
            
            if 'Sample_Type' in predictions_df.columns:
                # Analysis by Sample Type
                for sample_type in predictions_df['Sample_Type'].unique():
                    if pd.isna(sample_type):
                        continue
                    
                    type_mask = predictions_df['Sample_Type'] == sample_type
                    type_normal_mask = type_mask & (predictions_df['True_Label'] == 'Normal')
                    
                    if type_normal_mask.sum() > 0:
                        f.write(f"  Sample Type: {sample_type}\n")
                        for feature in original_feature_columns:
                            original_col = f'Original_{feature}'
                            if original_col in predictions_df.columns:
                                data = predictions_df.loc[type_normal_mask, original_col].dropna()
                                if len(data) > 0 and data.mean() != 0:
                                    cv = data.std() / data.mean()
                                    f.write(f"    - {feature}: CV = {cv:.3f} (n={len(data)})\n")
                                    if cv > 0.3:
                                        f.write(f"      Warning: High variability in {feature} for {sample_type} samples\n")
            else:
                # Overall Analysis
                normal_mask = predictions_df['True_Label'] == 'Normal'
                for feature in original_feature_columns:
                    original_col = f'Original_{feature}'
                    if original_col in predictions_df.columns:
                        data = predictions_df.loc[normal_mask, original_col].dropna()
                        if len(data) > 0 and data.mean() != 0:
                            cv = data.std() / data.mean()
                            f.write(f"   - {feature}: CV = {cv:.3f} (n={len(data)})\n")
                            if cv > 0.3:
                                f.write(f"     Warning: High variability in {feature} for Normal samples\n")

    print(f"Detailed report saved to: {report_file}")

# Save improved models and results
def save_improved_models_and_results(all_results, top3_models_info, preprocessor, target_encoder, results_dir):
    """Save models and results to files"""
    
    # Save preprocessor
    preprocessor_path = os.path.join(results_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save target encoder
    encoder_path = os.path.join(results_dir, 'target_encoder.pkl')
    joblib.dump(target_encoder, encoder_path)
    
    # Save top 3 models
    top3_model_paths = []
    for i, (model_name, model_instance) in enumerate(top3_models_info, 1):
        # Clean model name for filename
        clean_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        model_path = os.path.join(results_dir, f'top3_model_{i}_{clean_name}.pkl')
        joblib.dump(model_instance, model_path)
        top3_model_paths.append(model_path)
        print(f"Saved top {i} model to: {model_path}")
    
    # Save results to Excel
    results_data = []
    for model_name, results in all_results.items():
        results_data.append({
            'Model': model_name,
            'Model_Abbreviation': abbreviate_model_name(model_name),
            'Balancing_Method': results['Balancing_Method'],
            'Train_Accuracy': results['Train']['Accuracy'],
            'Validation_Accuracy': results['Validation']['Accuracy'],
            'Test_Accuracy': results['Test']['Accuracy'],
            'Test_F1_Macro': results['Test']['F1_Macro'],
            'Test_F1_Weighted': results['Test']['F1_Weighted'],
            'Test_Precision_Normal': results['Test']['Precision_0'],
            'Test_Recall_Normal': results['Test']['Recall_0'],
            'Test_F1_Normal': results['Test']['F1_0'],
            'Test_Precision_Abnormal': results['Test']['Precision_1'],
            'Test_Recall_Abnormal': results['Test']['Recall_1'],
            'Test_F1_Abnormal': results['Test']['F1_1'],
            'CV_Mean': results['CV_Mean'],
            'CV_Std': results['CV_Std'],
            'Overfitting_Gap': results['Overfitting_Gap']
        })
    
    results_df = pd.DataFrame(results_data)
    results_excel_path = os.path.join(results_dir, 'model_results.xlsx')
    results_df.to_excel(results_excel_path, index=False)
    
    print(f"Model results saved to: {results_excel_path}")

# Main function - MODIFIED to handle all new requirements
def main():
    """Main execution function"""
    apply_lang_from_cli()
    print("Starting machine learning classification prediction...")
    
    # 1. Get user input paths
    data_path, results_dir = get_user_paths()
    
    # 2. Load and preprocess data
    X, y, feature_columns, target_encoder, sample_labels = load_and_preprocess_data(data_path)
    if X is None:
        return
    
    # Store original features for later statistics
    original_features_df = X[["d202(‰)", "D199(‰)", "D200(‰)", "D201(‰)"]].copy()
    
    # 3. Split dataset (stratified by both target and sample type if possible)
    print("\nSplitting dataset...")
    try:
        # Create a combined stratification variable
        if sample_labels is not None:
            # Combine target and sample type for stratification
            stratify_col = pd.Series(y).astype(str) + "_" + sample_labels.astype(str)
            stratify_values = stratify_col.values
        else:
            stratify_values = y
        
        X_temp, X_test, y_temp, y_test, sample_labels_temp, sample_labels_test = train_test_split(
            X, y, sample_labels, test_size=0.2, random_state=42, stratify=stratify_values
        )
        
        # Second split
        if sample_labels_temp is not None:
            stratify_temp = pd.Series(y_temp).astype(str) + "_" + sample_labels_temp.astype(str)
            stratify_values_temp = stratify_temp.values
        else:
            stratify_values_temp = y_temp
            
        X_train, X_val, y_train, y_val, sample_labels_train, sample_labels_val = train_test_split(
            X_temp, y_temp, sample_labels_temp, test_size=0.25, random_state=42, stratify=stratify_values_temp
        )
    except:
        # Fallback to simple stratification
        print("Using simple stratification...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        sample_labels_test = None
    
    print(f"Train shape: {X_train.shape}, class distribution: {Counter(y_train)}")
    print(f"Validation shape: {X_val.shape}, class distribution: {Counter(y_val)}")
    print(f"Test shape: {X_test.shape}, class distribution: {Counter(y_test)}")
    
    # 4. Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Fit preprocessor on training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    print("\nGetting feature names from ColumnTransformer...")
    try:
        # Get all feature names
        feature_names = preprocessor.get_feature_names_out()
        print(f"Successfully got {len(feature_names)} feature names from ColumnTransformer")
    except Exception as e:
        print(f"Error getting feature names from ColumnTransformer: {e}")
        print("Using fallback method...")
        
        # Fallback solution
        feature_names = []
        
        # Numerical features (4)
        feature_names.extend(["d202", "D199", "D200", "D201"])
        
        # Get categorical features
        try:
            cat_transformer = preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'categories_'):
                categories = cat_transformer.categories_[0]
                for cat in categories:
                    feature_names.append(f"Label_{cat}")
                print(f"Created {len(categories)} categorical feature names")
            else:
                unique_labels = X_train['Label'].nunique()
                for i in range(unique_labels):
                    feature_names.append(f"Label_{i}")
                print(f"Estimated {unique_labels} categorical feature names")
        except Exception as e2:
            print(f"Error in fallback: {e2}")
            total_features = X_train_preprocessed.shape[1]
            feature_names = [f'feature_{i}' for i in range(total_features)]
            print(f"Created {total_features} generic feature names")
    
    # Convert to DataFrame for feature names
    X_train_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_preprocessed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)
    
    # Add shape check
    print(f"\nDataFrame shapes check:")
    print(f"X_train_preprocessed shape: {X_train_preprocessed.shape}")
    print(f"X_train_df shape: {X_train_df.shape}")
    print(f"Number of feature names: {len(feature_names)}")
    
    # Validate shape matches
    if X_train_preprocessed.shape[1] != len(feature_names):
        print(f"ERROR: Shape mismatch! Data has {X_train_preprocessed.shape[1]} columns, "
              f"but {len(feature_names)} feature names provided")
        feature_names = [f'feature_{i}' for i in range(X_train_preprocessed.shape[1])]
        X_train_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
        X_val_df = pd.DataFrame(X_val_preprocessed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)
        print(f"Recreated DataFrames with {len(feature_names)} feature names")
    
    # 5. Get balancing methods and models
    balancing_methods = get_balancing_methods()
    models = get_improved_models()
    all_results = {}
    
    print("\nBegin training and evaluating models (with different balancing methods)...")
    
    # Experiment with each balancing method
    for balance_name, balancer in balancing_methods.items():
        print(f"\n=== Using balancing method: {balance_name} ===")
        
        # Handle training data balancing
        if balancer is not None:
            X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_preprocessed, y_train)
            print(f"Balanced train shape: {X_train_balanced.shape}, class distribution: {Counter(y_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate model
                full_model_name = f"{model_name}_{balance_name}"
                results = evaluate_model(model, X_train_balanced, X_val_preprocessed, X_test_preprocessed, 
                                       y_train_balanced, y_val, y_test, full_model_name, balance_name)
                all_results[full_model_name] = results
                
                test_f1_macro = results['Test']['F1_Macro']
                test_recall_0 = results['Test']['Recall_0']
                overfitting_gap = results['Overfitting_Gap']
                print(f"  {model_name} - F1: {test_f1_macro:.4f}, Recall(normal): {test_recall_0:.4f}, Overfitting: {overfitting_gap:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
    
    # 6. Select top 3 models from different algorithms and balancing methods
    print("\nSelecting top 3 models from different algorithms...")
    
    # Define allowed balancing methods for top models
    allowed_balancing_methods = ['SMOTE', 'ADASYN', 'SMOTEENN', 'UnderSampling']
    
    # Filter results to only include allowed balancing methods
    filtered_results = {k: v for k, v in all_results.items() 
                       if any(method in k for method in allowed_balancing_methods)}
    
    # Group by algorithm type (first part of model name)
    algorithm_groups = {}
    for model_name in filtered_results.keys():
        algorithm = model_name.split('_')[0]
        if algorithm not in algorithm_groups:
            algorithm_groups[algorithm] = []
        algorithm_groups[algorithm].append(model_name)
    
    # Select top model from each algorithm group
    top_models = []
    used_algorithms = set()
    
    # Sort all models by Test F1 Macro score
    all_sorted_models = sorted(filtered_results.items(), 
                              key=lambda x: x[1]['Test']['F1_Macro'], 
                              reverse=True)
    
    # Select top 3 models ensuring different algorithms
    for model_name, results in all_sorted_models:
        algorithm = model_name.split('_')[0]
        if algorithm not in used_algorithms:
            top_models.append((model_name, results))
            used_algorithms.add(algorithm)
            if len(top_models) >= 3:
                break
    
    # If we don't have 3 different algorithms, fill with next best models
    if len(top_models) < 3:
        for model_name, results in all_sorted_models:
            if model_name not in [m[0] for m in top_models]:
                top_models.append((model_name, results))
                if len(top_models) >= 3:
                    break
    
    print("\nTop 3 selected models (different algorithms):")
    for i, (model_name, results) in enumerate(top_models, 1):
        abbr_name = abbreviate_model_name(model_name)
        print(f"  {i}. {model_name} ({abbr_name})")
        print(f"     Test Macro F1: {results['Test']['F1_Macro']:.4f}")
        print(f"     Normal Recall: {results['Test']['Recall_0']:.4f}")
        print(f"     CV Std: {results['CV_Std']:.4f}")
        print(f"     Balancing method: {results['Balancing_Method']}")
    
    # 7. Retrain and save top 3 models
    top3_models_info = []  # Store (model_name, model_instance)
    for i, (full_model_name, results) in enumerate(top_models, 1):
        base_model_name = full_model_name.split('_')[0]
        balance_method = results['Balancing_Method']
        
        # Prepare balanced training data
        if balance_method != 'Original':
            balancer = balancing_methods.get(balance_method)
            if balancer is not None:
                X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_preprocessed, y_train)
            else:
                X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
        else:
            X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
        
        # Get the model instance and retrain
        model_to_train = models[base_model_name]
        print(f"Retraining top model #{i}: {full_model_name} ...")
        try:
            model_to_train.fit(X_train_balanced, y_train_balanced)
            
            # Store model info for plotting
            top3_models_info.append((full_model_name, model_to_train))
            
        except Exception as e:
            print(f"Error retraining {full_model_name}: {e}")
            continue
    
    # 8. Save top 3 model predictions WITH CONFIDENCE SCORES
    print("\nSaving top 3 model predictions with confidence scores...")
    
    # Get original features for test set
    if original_features_df is not None:
        # Align with test indices
        test_indices = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
        original_features_test = original_features_df.loc[test_indices].copy()
    else:
        original_features_test = None
    
    predictions_data, predictions_df = save_top3_predictions(
        top3_models_info, X_test_df, y_test, sample_labels_test, 
        target_encoder, results_dir, original_features_test
    )
    
    # 9. Plot prediction confidence
    print("\nPlotting prediction confidence...")
    plot_prediction_confidence(predictions_df, results_dir)
    
    # 10. Plot results
    print("Generating plots...")
    plot_improved_results(all_results, X_test_df, y_test, top3_models_info, 
                         target_encoder, sample_labels_test, results_dir)
    
    # 11. Generate report WITH FEATURE STATISTICS
    print("Generating detailed report with feature statistics...")
    generate_improved_report(all_results, top3_models_info, target_encoder, 
                            X_test_df, y_test, sample_labels_test, results_dir,
                            predictions_df, original_features_test)
    
    # 12. Save models and results
    print("Saving models and results...")
    save_improved_models_and_results(all_results, top3_models_info, preprocessor, target_encoder, results_dir)

    print(f"\nAll results saved to: {results_dir}")
    print("Machine learning classification pipeline completed!")

if __name__ == "__main__":
    main()