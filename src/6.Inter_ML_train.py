import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import joblib
from datetime import datetime
import matplotlib as mpl

LANG = os.environ.get("HG_MC_AUTO_LANG", "zh").lower()

TRANSLATIONS = {
    "zh": {
        "banner_title": "Hg 同位素异常分类",
        "banner_subtitle": "分析系统",
        "input_data": "请输入输入数据文件路径（all_abnormal_data_processed.xlsx）（Excel 格式）: ",
        "input_output": "请输入输出结果目录路径（建议放在 Model_Inter 文件夹中）: ",
        "path_empty": "路径不能为空，请重新输入。",
        "not_exist": "文件不存在: {path}，请重新输入。",
        "excel_required": "请提供 Excel 文件（.xlsx 或 .xls 格式）",
        "feature_title": "特征集选择",
        "feature_prompt": "请选择特征集:",
        "feature_basic": "1. 仅使用基础特征",
        "feature_enhanced": "2. 使用增强地球化学特征",
        "choice_prompt": "请输入选择 (1 或 2): ",
        "invalid_choice": "无效选择。请输入 1 或 2。",
        "selected": "已选择: {feature} 特征",
        "data_path": "数据路径: {path}",
        "output_dir": "输出目录: {dir}",
        "feature_option_basic": "基础",
        "feature_option_enhanced": "增强",
    },
    "en": {
        "banner_title": "Hg Isotope Anomaly Classification",
        "banner_subtitle": "Analysis System",
        "input_data": "Please enter the input data file path (all_abnormal_data_processed.xlsx) (Excel format): ",
        "input_output": "Please enter the output results directory path (Recommended to be in the Model_Inter folder): ",
        "path_empty": "Path cannot be empty, please re-enter.",
        "not_exist": "File does not exist: {path}, please re-enter.",
        "excel_required": "Please provide an Excel file (.xlsx or .xls format)",
        "feature_title": "Feature Set Selection",
        "feature_prompt": "Please select feature set:",
        "feature_basic": "1. Basic features only",
        "feature_enhanced": "2. Enhanced geochemical features",
        "choice_prompt": "Enter your choice (1 or 2): ",
        "invalid_choice": "Invalid choice. Please enter 1 or 2.",
        "selected": "Selected: {feature} features",
        "data_path": "Data path: {path}",
        "output_dir": "Output directory: {dir}",
        "feature_option_basic": "basic",
        "feature_option_enhanced": "enhanced",
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

# Set Nature-style plotting parameters
def set_nature_style():
    """Set plotting parameters for figures"""
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        
        # Figure settings
        'figure.dpi': 300,
        'figure.figsize': (3.5, 2.6),  # single column width
        'figure.autolayout': True,
        
        # Line and marker settings
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.linewidth': 0.5,
        
        # Tick settings
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.25,
        'ytick.minor.width': 0.25,
        
        # Grid settings
        'grid.linewidth': 0.25,
        'grid.alpha': 0.3,
        
        # Savefig settings
        'savefig.dpi': 600,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })
    
    # Set color palette
    sns.set_palette("husl")

# Initialize Nature style
set_nature_style()

def get_user_paths():
    """Get input and output paths from user"""
    print("=" * 60)
    print(text("banner_title"))
    print(text("banner_subtitle"))
    print("=" * 60)
    
    # Get input file path
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
    
    # Get output directory path
    while True:
        results_dir = input(text("input_output")).strip()
        if not results_dir:
            print(text("path_empty"))
            continue
        break
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    return data_path, results_dir

class GeochemicalMLAnalyzer:
    def __init__(self, feature_option='basic', output_dir=None):
        """
        Initialize analyzer with feature option
        
        Parameters:
        feature_option: 'basic' or 'enhanced'
        output_dir: directory to save results
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.results = {}
        self.analysis_report = []
        self.feature_option = feature_option
        self.output_dir = output_dir
        
        # color palette
        self.colors = {
            'primary': ['#2E86AB', '#A23B72', '#F18F01', '#4ECDC4', '#FF6B6B', '#95E08E'],
            'categorical': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        }
        
    def add_to_report(self, section, content):
        """Add content to analysis report"""
        self.analysis_report.append(f"\n{'='*50}")
        self.analysis_report.append(f"{section}")
        self.analysis_report.append(f"{'='*50}")
        if isinstance(content, list):
            self.analysis_report.extend(content)
        else:
            self.analysis_report.append(str(content))
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess data"""
        print("=== Data Loading and Preprocessing ===")
        df = pd.read_excel(file_path)
        
        print(f"Original data shape: {df.shape}")
        
        # Keep only data with anomaly annotations
        df_anomaly = df[df['Cause of the anomaly'].notna()]
        print(f"Anomaly data shape: {df_anomaly.shape}")
        
        self.add_to_report("DATA OVERVIEW", [
            f"Original dataset shape: {df.shape}",
            f"Anomaly dataset shape: {df_anomaly.shape}",
            f"Features available: {len(df.columns)}",
            f"Target variable: 'Cause of the anomaly'",
            f"Feature option selected: {self.feature_option}"
        ])
        
        return df_anomaly
    
    def select_features(self, df):
        """Select features based on user choice"""
        print(f"\n=== Feature Selection ({self.feature_option.upper()}) ===")
        
        # Define basic features
        basic_features = ["StdErr(abs)202Hg/198Hg", "StdErr(abs)201Hg/198Hg", 
                         "StdErr(abs)200Hg/198Hg", "StdErr(abs)199Hg/198Hg", "R-THg(%)"]
        
        # Check if basic features exist
        missing_features = [f for f in basic_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Following features not found: {missing_features}")
            basic_features = [f for f in basic_features if f in df.columns]
        
        if self.feature_option == 'basic':
            print("Using BASIC features only")
            selected_features = basic_features.copy()
            
        elif self.feature_option == 'enhanced':
            print("Using ENHANCED geochemical features")
            selected_features = basic_features.copy()
            
            # Geochemical enhanced features
            # 1. Relative Standard Deviation (RSD)
            if all(col in df.columns for col in ['202Hg/198Hg', '201Hg/198Hg', '200Hg/198Hg', '199Hg/198Hg']):
                df['RSD_202Hg'] = df['StdErr(abs)202Hg/198Hg'] / df['202Hg/198Hg'].replace(0, np.nan).abs()
                df['RSD_201Hg'] = df['StdErr(abs)201Hg/198Hg'] / df['201Hg/198Hg'].replace(0, np.nan).abs()
                df['RSD_200Hg'] = df['StdErr(abs)200Hg/198Hg'] / df['200Hg/198Hg'].replace(0, np.nan).abs()
                df['RSD_199Hg'] = df['StdErr(abs)199Hg/198Hg'] / df['199Hg/198Hg'].replace(0, np.nan).abs()
                selected_features.extend(['RSD_202Hg', 'RSD_201Hg', 'RSD_200Hg', 'RSD_199Hg'])
                print("RSD features added")
            
            # 2. StdErr ratios
            df['StdErr_ratio_199_202'] = df['StdErr(abs)199Hg/198Hg'] / df['StdErr(abs)202Hg/198Hg'].replace(0, np.nan)
            df['StdErr_ratio_201_202'] = df['StdErr(abs)201Hg/198Hg'] / df['StdErr(abs)202Hg/198Hg'].replace(0, np.nan)
            selected_features.extend(['StdErr_ratio_199_202', 'StdErr_ratio_201_202'])
            print("StdErr ratio features added")
            
            # 3. Total noise level
            df['Total_StdErr'] = (df['StdErr(abs)202Hg/198Hg'] + 
                                 df['StdErr(abs)201Hg/198Hg'] + 
                                 df['StdErr(abs)200Hg/198Hg'] + 
                                 df['StdErr(abs)199Hg/198Hg']) / 4
            selected_features.append('Total_StdErr')
            print("Total StdErr feature added")
            
            # 4. Data quality score
            valid_rtHg = df['R-THg(%)'].replace(0, np.nan)
            df['Data_Quality_Score'] = 1 / (1 + df['Total_StdErr'] * valid_rtHg)
            selected_features.append('Data_Quality_Score')
            print("Data quality score feature added")
            
            # 5. Coefficient of variation features
            df['CV_202Hg'] = df['StdErr(abs)202Hg/198Hg'] / df['Total_StdErr'].replace(0, np.nan)
            df['CV_201Hg'] = df['StdErr(abs)201Hg/198Hg'] / df['Total_StdErr'].replace(0, np.nan)
            selected_features.extend(['CV_202Hg', 'CV_201Hg'])
            print("CV features added")
        else:
            raise ValueError(f"Unknown feature option: {self.feature_option}. Use 'basic' or 'enhanced'.")
        
        print(f"\nSelected features ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            print(f"  {i:2d}. {feature}")
        
        # Check missing values
        print("\nFeature missing values statistics:")
        missing_info = []
        for feature in selected_features:
            if feature in df.columns:
                missing_count = df[feature].isnull().sum()
                if missing_count > 0:
                    print(f"  {feature}: {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")
                    missing_info.append(f"  {feature}: {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")
        
        # Remove features with too many missing values (>50%)
        valid_features = []
        for feature in selected_features:
            if feature in df.columns:
                missing_count = df[feature].isnull().sum()
                if missing_count / len(df) < 0.5:
                    valid_features.append(feature)
                else:
                    print(f"Removing feature {feature} (too many missing values: {missing_count/len(df)*100:.2f}%)")
        
        print(f"\nFinal feature set ({len(valid_features)} features): {valid_features}")
        
        self.add_to_report("FEATURE SELECTION", [
            f"Feature option: {self.feature_option}",
            f"Selected features: {valid_features}",
            f"Total features: {len(valid_features)}",
            "Missing values information:"
        ] + missing_info + [f"Final feature set: {valid_features}"])
        
        return df, valid_features
    
    def handle_missing_values(self, X_train, X_test, features):
        """Handle missing values in training and test sets"""
        print("\n=== Handling Missing Values ===")
        
        # Reset indices to avoid index issues
        X_train_reset = X_train.reset_index(drop=True)
        X_test_reset = X_test.reset_index(drop=True)
        
        train_nan = X_train_reset.isnull().sum().sum()
        test_nan = X_test_reset.isnull().sum().sum()
        print(f"Training set total missing values: {train_nan}")
        print(f"Test set total missing values: {test_nan}")
        
        if train_nan > 0 or test_nan > 0:
            # Fit imputer on training data and transform both sets
            X_train_imputed = self.imputer.fit_transform(X_train_reset)
            X_test_imputed = self.imputer.transform(X_test_reset)
            
            # Convert back to DataFrame with correct column names
            X_train_clean = pd.DataFrame(X_train_imputed, columns=features)
            X_test_clean = pd.DataFrame(X_test_imputed, columns=features)
            
            print("Missing values filled with median")
        else:
            X_train_clean = X_train_reset.copy()
            X_test_clean = X_test_reset.copy()
            print("No missing values to handle")
        
        return X_train_clean, X_test_clean
    
    def analyze_class_distribution(self, df, target_col):
        """Analyze class distribution"""
        print("\n=== Class Distribution Analysis ===")
        class_dist = df[target_col].value_counts()
        print("Class distribution:")
        class_info = []
        for i, (cls, count) in enumerate(class_dist.items()):
            print(f"  {i+1}. {cls}: {count} samples ({count/len(df)*100:.1f}%)")
            class_info.append(f"  {i+1}. {cls}: {count} samples ({count/len(df)*100:.1f}%)")
        
        # Calculate imbalance ratio
        imbalance_ratio = class_dist.max() / class_dist.min()
        print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 10:
            print("⚠️  Severe class imbalance, using cost-sensitive learning")
            imbalance_note = "Severe class imbalance detected - using cost-sensitive learning"
        elif imbalance_ratio > 5:
            print("⚠️  Moderate class imbalance, using cost-sensitive learning")
            imbalance_note = "Moderate class imbalance detected - using cost-sensitive learning"
        else:
            print("✅ Relatively balanced class distribution")
            imbalance_note = "Relatively balanced class distribution"
        
        self.add_to_report("CLASS DISTRIBUTION", class_info + [
            f"Imbalance ratio: {imbalance_ratio:.1f}:1",
            imbalance_note
        ])
        
        return class_dist
    
    def strategic_data_splitting(self, X, y, test_size=0.2):
        """Strategic data splitting with geochemical considerations"""
        print("\n=== Strategic Data Splitting ===")
        
        # Use stratified splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y, shuffle=True
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Check class distribution after splitting
        train_dist = pd.Series(y_train).value_counts()
        test_dist = pd.Series(y_test).value_counts()
        
        print("\nClass distribution after splitting:")
        split_info = []
        for cls in np.unique(y):
            cls_name = self.label_encoder.inverse_transform([cls])[0]
            print(f"  {cls_name}: Train {train_dist[cls]}, Test {test_dist[cls]}")
            split_info.append(f"  {cls_name}: Train {train_dist[cls]}, Test {test_dist[cls]}")
        
        self.add_to_report("DATA SPLITTING", [
            f"Training set size: {X_train.shape[0]} samples",
            f"Test set size: {X_test.shape[0]} samples",
            "Class distribution after splitting:"
        ] + split_info)
        
        return X_train, X_test, y_train, y_test
    
    def calculate_absolute_rtHg_stats(self, df):
        """Calculate absolute R-THg(%) statistics"""
        print("\n=== Absolute R-THg(%) Statistics ===")
        
        abs_rtHg_stats = []
        anomaly_types = df['Cause of the anomaly'].unique()
        
        for anomaly_type in anomaly_types:
            subset = df[df['Cause of the anomaly'] == anomaly_type]
            abs_mean_rtHg = subset['R-THg(%)'].abs().mean()
            print(f"{anomaly_type}: {abs_mean_rtHg:.3f}")
            abs_rtHg_stats.append(f"{anomaly_type}: {abs_mean_rtHg:.3f}")
        
        self.add_to_report("ABSOLUTE R-THG STATISTICS", [
            "Average absolute R-THg(%) by anomaly type:"
        ] + abs_rtHg_stats)
        
        return abs_rtHg_stats
    
    def train_cost_sensitive_models(self, X_train, X_test, y_train, y_test, features):
        """Train cost-sensitive learning models with comprehensive evaluation"""
        print("\n=== Cost-Sensitive Model Training ===")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        weight_dict = dict(enumerate(class_weights))
        
        print("Class weights:")
        weight_info = []
        for cls, weight in weight_dict.items():
            cls_name = self.label_encoder.inverse_transform([cls])[0]
            print(f"  {cls_name}: {weight:.2f}")
            weight_info.append(f"  {cls_name}: {weight:.2f}")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                class_weight=weight_dict,
                random_state=42,
                max_depth=10
            ),
            'SVM': SVC(
                class_weight=weight_dict,
                random_state=42,
                probability=True,
                kernel='rbf'
            ),
            'Logistic Regression': LogisticRegression(
                class_weight=weight_dict,
                random_state=42,
                max_iter=1000,
                C=0.1
            )
        }
        
        # Handle missing values
        X_train_clean, X_test_clean = self.handle_missing_values(X_train, X_test, features)
        
        # Convert to numpy arrays for consistency
        X_train_array = X_train_clean.values
        X_test_array = X_test_clean.values
        
        # Train and evaluate models
        model_performance = []
        for name, model in models.items():
            print(f"\n--- {name} ---")
            
            try:
                # Feature scaling for SVM and Logistic Regression
                if name in ['SVM', 'Logistic Regression']:
                    X_train_used = self.scaler.fit_transform(X_train_array)
                    X_test_used = self.scaler.transform(X_test_array)
                else:
                    # For Random Forest, use the data as is (no scaling needed)
                    X_train_used = X_train_array
                    X_test_used = X_test_array
                
                # Training
                model.fit(X_train_used, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train_used)
                y_pred_test = model.predict(X_test_used)
                y_pred_proba = model.predict_proba(X_test_used) if hasattr(model, 'predict_proba') else None
                
                # Cross-validation predictions
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_predictions = []
                cv_true = []
                
                for train_idx, val_idx in cv.split(X_train_used, y_train):
                    X_train_cv, X_val_cv = X_train_used[train_idx], X_train_used[val_idx]
                    y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                    
                    # Create a new instance of the model for each fold
                    model_cv = type(model)(**model.get_params())
                    model_cv.fit(X_train_cv, y_train_cv)
                    y_pred_val = model_cv.predict(X_val_cv)
                    
                    cv_predictions.extend(y_pred_val)
                    cv_true.extend(y_val_cv)
                
                # Calculate metrics for all sets
                def calculate_metrics(y_true, y_pred, set_name):
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    balanced_acc = balanced_accuracy_score(y_true, y_pred)
                    
                    return {
                        'set': set_name,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'balanced_accuracy': balanced_acc
                    }
                
                train_metrics = calculate_metrics(y_train, y_pred_train, 'Training')
                test_metrics = calculate_metrics(y_test, y_pred_test, 'Test')
                val_metrics = calculate_metrics(cv_true, cv_predictions, 'Validation')
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train_used, y_train, cv=5, scoring='balanced_accuracy')
                
                print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
                print(f"CV Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
                # Get feature importance if available
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                
                # Save results
                self.results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'val_metrics': val_metrics,
                    'cv_scores': cv_scores,
                    'predictions': y_pred_test,
                    'probabilities': y_pred_proba,
                    'feature_importance': feature_importance
                }
                
                # Add to performance summary
                model_performance.append({
                    'Model': name,
                    'Train_Accuracy': train_metrics['accuracy'],
                    'Test_Accuracy': test_metrics['accuracy'],
                    'Validation_Accuracy': val_metrics['accuracy'],
                    'Train_Precision': train_metrics['precision'],
                    'Test_Precision': test_metrics['precision'],
                    'Validation_Precision': val_metrics['precision'],
                    'Train_Recall': train_metrics['recall'],
                    'Test_Recall': test_metrics['recall'],
                    'Validation_Recall': val_metrics['recall'],
                    'Train_F1': train_metrics['f1_score'],
                    'Test_F1': test_metrics['f1_score'],
                    'Validation_F1': val_metrics['f1_score'],
                    'Balanced_Accuracy': test_metrics['balanced_accuracy'],
                    'CV_Mean': cv_scores.mean(),
                    'CV_Std': cv_scores.std()
                })
                
            except Exception as e:
                print(f"Model {name} training failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.add_to_report("CLASS WEIGHTS", weight_info)
        
        # Create formatted performance report
        perf_report = []
        for perf in model_performance:
            perf_report.append(f"\n{perf['Model']}:")
            perf_report.append(f"  Train Accuracy: {perf['Train_Accuracy']:.4f}")
            perf_report.append(f"  Test Accuracy: {perf['Test_Accuracy']:.4f}")
            perf_report.append(f"  Validation Accuracy: {perf['Validation_Accuracy']:.4f}")
            perf_report.append(f"  Balanced Accuracy: {perf['Balanced_Accuracy']:.4f}")
        
        self.add_to_report("MODEL PERFORMANCE SUMMARY", perf_report)
        
        return models, model_performance

    def comprehensive_evaluation(self, y_test, features):
        """Comprehensive evaluation of best model"""
        print("\n=== Comprehensive Model Evaluation ===")
        
        if not self.results:
            print("No successful model training results")
            return None, None
        
        # Select best model based on balanced accuracy
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_metrics']['balanced_accuracy'])
        best_result = self.results[best_model_name]
        
        print(f"Best Model: {best_model_name}")
        print(f"Balanced Accuracy: {best_result['test_metrics']['balanced_accuracy']:.4f}")
        
        # Detailed classification report
        y_pred = best_result['predictions']
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, best_model_name)
        
        # Feature importance analysis
        if best_result['feature_importance'] is not None:
            self.plot_feature_importance(best_result['model'], features, best_model_name)
        
        # Prediction confidence analysis
        self.analyze_prediction_confidence(best_result['probabilities'], y_test, best_model_name)
        
        # Model performance comparison
        self.plot_model_performance_comparison()
        
        # Add best model info to report
        self.add_to_report("BEST MODEL SELECTION", [
            f"Best Model: {best_model_name}",
            f"Test Balanced Accuracy: {best_result['test_metrics']['balanced_accuracy']:.4f}",
            f"Test Accuracy: {best_result['test_metrics']['accuracy']:.4f}",
            f"Test Precision: {best_result['test_metrics']['precision']:.4f}",
            f"Test Recall: {best_result['test_metrics']['recall']:.4f}",
            f"Test F1-Score: {best_result['test_metrics']['f1_score']:.4f}"
        ])
        
        return best_model_name, best_result
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        """Plot confusion matrix"""
        # Set figure size for Nature (single column)
        fig_width = 3.5  # inches
        fig_height = 2.8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap with Nature-style colors
        im = ax.imshow(cm_normalized, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Show both count and percentage
                count_text = f"{cm[i, j]}"
                percent_text = f"{cm_normalized[i, j]*100:.1f}%"
                
                # Choose text color based on background
                text_color = 'black' if cm_normalized[i, j] < 0.6 else 'white'
                
                ax.text(j, i, f"{count_text}\n({percent_text})",
                       ha='center', va='center',
                       color=text_color, fontsize=7)
        
        # Set labels
        class_names = self.label_encoder.classes_
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        
        # Set axis labels
        ax.set_xlabel('Predicted Label', fontsize=9, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Frequency', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f'confusion_matrix_{self.feature_option}.png')
        fig.savefig(fig_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"✓ Confusion matrix saved: {fig_path}")
        plt.close(fig)
    
    def plot_feature_importance(self, model, features, model_name):
        """Plot feature importance"""
        fig_width = 3.5
        fig_height = 2.8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Take top 10 features for clarity
        importance_df = importance_df.tail(10)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['importance'], 
                      color=self.colors['primary'][0], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Customize bars with different colors for feature types
        for i, feature in enumerate(importance_df['feature']):
            if 'StdErr' in feature:
                bars[i].set_color(self.colors['primary'][1])
            elif 'RSD' in feature:
                bars[i].set_color(self.colors['primary'][2])
            elif 'R-THg' in feature:
                bars[i].set_color(self.colors['primary'][3])
            elif 'Quality' in feature:
                bars[i].set_color(self.colors['primary'][4])
            elif 'CV' in feature:
                bars[i].set_color(self.colors['primary'][5])
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'], fontsize=7)
        ax.set_xlabel('Importance Score', fontsize=9, fontweight='bold')
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', va='center', fontsize=6)
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set title
        feature_set_title = "Enhanced Geochemical" if self.feature_option == 'enhanced' else "Basic"
        ax.set_title(f'{model_name}\nTop Feature Importance', 
                    fontsize=9, fontweight='bold', pad=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f'feature_importance_{self.feature_option}.png')
        fig.savefig(fig_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"✓ Feature importance plot saved: {fig_path}")
        plt.close(fig)
        
        # Feature importance interpretation
        print("\n=== Feature Importance Interpretation ===")
        top_features = importance_df.tail(5)
        importance_info = []
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            if 'StdErr' in feature:
                interpretation = "Instrument measurement precision indicator"
            elif 'RSD' in feature:
                interpretation = "Relative error, data stability indicator"
            elif 'R-THg' in feature:
                interpretation = "Data recovery rate, quality control core"
            elif 'Total_StdErr' in feature:
                interpretation = "Overall noise level comprehensive indicator"
            elif 'Data_Quality_Score' in feature:
                interpretation = "Composite precision-accuracy quality score"
            elif 'CV' in feature:
                interpretation = "Error contribution coefficient"
            else:
                interpretation = "Basic geochemical feature"
            
            print(f"  {feature}: {importance:.3f} - {interpretation}")
            importance_info.append(f"  {feature}: {importance:.3f} - {interpretation}")
        
        self.add_to_report("FEATURE IMPORTANCE", [
            "Top 5 most important features:"
        ] + importance_info)
    
    def plot_model_performance_comparison(self):
        """Plot model performance comparison"""
        fig_width = 3.5
        fig_height = 2.6
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        model_names = []
        balanced_accs = []
        test_accs = []
        
        for name, result in self.results.items():
            model_names.append(name)
            balanced_accs.append(result['test_metrics']['balanced_accuracy'])
            test_accs.append(result['test_metrics']['accuracy'])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, balanced_accs, width, 
                      label='Balanced Accuracy',
                      color=self.colors['primary'][0],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        bars2 = ax.bar(x + width/2, test_accs, width,
                      label='Test Accuracy',
                      color=self.colors['primary'][2],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                   f'{balanced_accs[i]:.3f}', ha='center', va='bottom', fontsize=6)
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                   f'{test_accs[i]:.3f}', ha='center', va='bottom', fontsize=6)
        
        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=9, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add legend
        ax.legend(fontsize=7, frameon=True, fancybox=False,
                edgecolor='black', framealpha=0.8,
                loc='upper right', prop={'size': 6})
        
        # Set title
        ax.set_title('Model Performance Comparison', 
                    fontsize=9, fontweight='bold', pad=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f'model_performance_{self.feature_option}.png')
        fig.savefig(fig_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"✓ Model performance comparison saved: {fig_path}")
        plt.close(fig)
    
    def analyze_prediction_confidence(self, probabilities, y_test, model_name):
        """Analyze prediction confidence plot"""
        fig_width = 3.5
        fig_height = 2.6
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        max_probs = np.max(probabilities, axis=1)
        
        # Create histogram plot
        n, bins, patches = ax.hist(max_probs, bins=15, alpha=0.8, 
                                  color=self.colors['primary'][0],
                                  edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Prediction Confidence', fontsize=9, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics text
        stats_text = (f'Mean: {np.mean(max_probs):.3f}\n'
                     f'Median: {np.median(max_probs):.3f}\n'
                     f'Low (<0.7): {np.sum(max_probs < 0.7)}')
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='top',
               horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', 
                        alpha=0.8, edgecolor='black', linewidth=0.5))
        
        # Set title
        ax.set_title(f'{model_name}\nPrediction Confidence', 
                    fontsize=9, fontweight='bold', pad=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f'prediction_confidence_{self.feature_option}.png')
        fig.savefig(fig_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"✓ Prediction confidence plot saved: {fig_path}")
        plt.close(fig)
        
        print(f"\nPrediction Confidence Statistics:")
        print(f"Average Confidence: {np.mean(max_probs):.3f}")
        print(f"Median Confidence: {np.median(max_probs):.3f}")
        print(f"Low Confidence Samples (<0.7): {np.sum(max_probs < 0.7)}")
        
        self.add_to_report("PREDICTION CONFIDENCE", [
            f"Average Confidence: {np.mean(max_probs):.3f}",
            f"Median Confidence: {np.median(max_probs):.3f}",
            f"Low Confidence Samples (<0.7): {np.sum(max_probs < 0.7)}"
        ])
    
    def save_comprehensive_results(self, X_test, y_test, df, features, best_model_name, model_performance):
        """Save comprehensive results"""
        print("\n=== Saving Results ===")
        
        if best_model_name is None:
            print("No best model to save")
            return
        
        # Save prediction results
        best_result = self.results[best_model_name]
        results_df = pd.DataFrame({
            'Actual_Label': self.label_encoder.inverse_transform(y_test),
            'Predicted_Label': self.label_encoder.inverse_transform(best_result['predictions']),
            'Confidence': np.max(best_result['probabilities'], axis=1)
        })
        
        # Save performance summary
        performance_df = pd.DataFrame(model_performance)
        
        # Save feature importance
        if best_result['feature_importance'] is not None:
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': best_result['feature_importance']
            }).sort_values('Importance', ascending=False)
        else:
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': [0] * len(features)
            })
        
        # Save to files with feature option in filename
        results_df.to_excel(os.path.join(self.output_dir, f'prediction_results_{self.feature_option}.xlsx'), index=False)
        performance_df.to_excel(os.path.join(self.output_dir, f'model_performance_{self.feature_option}.xlsx'), index=False)
        feature_importance_df.to_excel(os.path.join(self.output_dir, f'feature_importance_{self.feature_option}.xlsx'), index=False)
        
        # Save models and preprocessing objects
        joblib.dump(best_result['model'], os.path.join(self.output_dir, f'best_model_{self.feature_option}.pkl'))
        joblib.dump(self.scaler, os.path.join(self.output_dir, f'scaler_{self.feature_option}.pkl'))
        joblib.dump(self.label_encoder, os.path.join(self.output_dir, f'label_encoder_{self.feature_option}.pkl'))
        joblib.dump(self.imputer, os.path.join(self.output_dir, f'imputer_{self.feature_option}.pkl'))
        
        # Save feature list
        with open(os.path.join(self.output_dir, f'feature_list_{self.feature_option}.txt'), 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")
        
        # Save analysis report
        with open(os.path.join(self.output_dir, f'analysis_report_{self.feature_option}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"GEOCHEMICAL MACHINE LEARNING ANALYSIS REPORT ({self.feature_option.upper()} FEATURES)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature option: {self.feature_option}\n")
            f.write("=" * 50 + "\n")
            
            for line in self.analysis_report:
                f.write(line + "\n")
        
        print(f"✅ All results saved to: {self.output_dir}")
        print(f"📊 Analysis report saved as: analysis_report_{self.feature_option}.txt")

def main():
    """Main function with user input for paths and feature selection"""
    apply_lang_from_cli()
    
    print("=" * 60)
    print(f"          {text('banner_title')}")
    print(f"                  {text('banner_subtitle')}")
    print("=" * 60)
    
    # Get user paths
    data_path, output_dir = get_user_paths()
    
    print("\n" + "=" * 40)
    print(text("feature_title"))
    print("=" * 40)
    print(f"\n{text('feature_prompt')}")
    print(text("feature_basic"))
    print(text("feature_enhanced"))
    print("-" * 40)
    
    while True:
        choice = input(f"\n{text('choice_prompt')}").strip()
        if choice == '1':
            feature_option = 'basic'
            break
        elif choice == '2':
            feature_option = 'enhanced'
            break
        else:
            print(text("invalid_choice"))
    
    print(f"\n{'='*40}")
    print(text("selected", feature=text("feature_option_basic") if feature_option == 'basic' else text("feature_option_enhanced").upper()))
    print(text("data_path", path=data_path))
    print(text("output_dir", dir=output_dir))
    print(f"{'='*40}\n")
    
    # Create analyzer with selected feature option and output directory
    analyzer = GeochemicalMLAnalyzer(feature_option=feature_option, output_dir=output_dir)
    
    try:
        # 1. Load and preprocess data
        df_anomaly = analyzer.load_and_preprocess_data(data_path)
        
        # 2. Select features based on user choice
        df_processed, features = analyzer.select_features(df_anomaly)
        
        # 3. Encode target variable
        y = analyzer.label_encoder.fit_transform(df_processed['Cause of the anomaly'])
        X = df_processed[features]
        
        print(f"\nFinal feature set ({len(features)} features):")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
        
        # 4. Analyze class distribution
        analyzer.analyze_class_distribution(df_processed, 'Cause of the anomaly')
        
        # 5. Calculate absolute R-THg statistics
        analyzer.calculate_absolute_rtHg_stats(df_processed)
        
        # 6. Data splitting
        X_train, X_test, y_train, y_test = analyzer.strategic_data_splitting(X, y)
        
        # 7. Train cost-sensitive models
        models, model_performance = analyzer.train_cost_sensitive_models(X_train, X_test, y_train, y_test, features)
        
        # 8. Comprehensive evaluation
        best_model_name, best_result = analyzer.comprehensive_evaluation(y_test, features)
        
        # 9. Save results
        analyzer.save_comprehensive_results(X_test, y_test, df_processed, features, best_model_name, model_performance)
        
        print(f"\n{'='*50}")
        print(f"🎯 Analysis Complete with {feature_option.upper()} features!")
        print(f"{'='*50}")
        print("Key Features of This Analysis:")
        print("  ✅ Publication-quality figures")
        print("  ✅ User-defined input and output paths")
        print("  ✅ Cost-sensitive learning for class imbalance")
        print("  ✅ Balanced accuracy as primary evaluation metric")
        print("  ✅ Comprehensive validation on train/test/validation sets")
        print("  ✅ Proper missing value handling")
        print("  ✅ Absolute R-THg(%) statistics for anomaly characterization")
        print("  ✅ Detailed analysis report generated")
        
        # Display file naming info
        print(f"\n📁 Output files in: {output_dir}")
        print(f"   Figures saved with '{feature_option}' suffix:")
        print(f"   • confusion_matrix_{feature_option}.png")
        print(f"   • feature_importance_{feature_option}.png")
        print(f"   • model_performance_{feature_option}.png")
        print(f"   • prediction_confidence_{feature_option}.png")
        
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()