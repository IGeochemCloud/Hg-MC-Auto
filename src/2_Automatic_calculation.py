# This script is the second step of the export software, implementing four tasks:
# Task 1: .csv_to_one_.xlsx - Integrate all .csv files exported in the first step into one Excel file (extracting Mean and StdErr (abs) from each .csv)
# Task 2: time_.log_to_.xlsx - Convert ".log" format to ".xlsx" format and sort the converted table by time
# Task 3: Merge two tables, add "Label" column, mark files containing "3133" as 3133, "3177" as 3177, "8610" as 8610, and other test samples as "sample", generating "Before_Fractionation_Calculation.xlsx"
# Task 4: Fractionation calculation based on "Before_Fractionation_Calculation.xlsx", generating "Before_Fractionation_Calculation_Completed.xlsx"

import argparse
import os
import re
import pandas as pd
import numpy as np

LANG = os.environ.get("HG_MC_AUTO_LANG", "zh").lower()

TRANSLATIONS = {
    "zh": {
        "input_log": "请输入 .log 文件路径: ",
        "input_output": "请输入输出文件路径（需与先前导出的 .csv 同位素文件位于同一文件夹中）: ",
        "extract_params": "正在提取仪器参数...",
        "extract_isotopes": "正在提取同位素测试值...",
        "merge_labels": "正在合并数据并添加标签...",
        "fractionation": "正在进行分馏计算...",
        "calc_d_values": "正在计算 D199(‰), D200(‰), D201(‰)...",
    },
    "en": {
        "input_log": "Please enter the .log file path: ",
        "input_output": "Please enter the output file path (It needs to be in the same folder as the .csv isotope file you exported earlier.): ",
        "extract_params": "Extracting instrument parameters...",
        "extract_isotopes": "Extracting isotope test values...",
        "merge_labels": "Merging data and adding labels...",
        "fractionation": "Performing fractionation calculation...",
        "calc_d_values": "Calculating D199(‰), D200(‰), D201(‰)...",
    },
}


def apply_lang_from_cli():
    global LANG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lang", choices=["zh", "en"], default=os.environ.get("HG_MC_AUTO_LANG", "zh"))
    args, _ = parser.parse_known_args()
    LANG = args.lang.lower()
    os.environ["HG_MC_AUTO_LANG"] = LANG


def text(key):
    return TRANSLATIONS.get(LANG, TRANSLATIONS["zh"]).get(key, key)


def main():
    apply_lang_from_cli()
    # Get user input paths
    DAT_DIR = input(text("input_log"))
    OUTPUT_DIR = input(text("input_output"))
    
    # Code 1: Extract instrument parameters
    print(text("extract_params"))
    fields = [
        "Extraction[V]", "Focus[V]", "Source Quad1[V]", "Rot-Quad1[V]", "Foc-Quad1[V]",
        "Rot-Quad2[V]", "Source Offset[V]", "Matsuda Plate[V]", "Cool Gas[l/min]",
        "Aux Gas[l/min]", "Sample Gas[l/min]", "Add Gas[l/min]", "Org Gas[l/min]",
        "Operation Power[W]", "X-Pos[mm]", "Y-Pos[mm]", "Z-Pos[mm]", "Ampl.-Temp[癈]",
        "Fore Vacuum[mbar]", "High Vacuum[mbar]", "IonGetter-Press[mbar]"
    ]

    # Match time (first line of file)
    time_pattern = re.compile(r'^([A-Za-z]{3},\d{1,2}-[A-Za-z]{3}-\d{4} \d{1,2}:\d{2}:\d{2}[AP]M)')

    # Match parameters
    pattern = re.compile(r'^(.*?\[.*?\]):\s*([-\d.eE]+)$')

    data_list = []

    for filename in os.listdir(DAT_DIR):
        if filename.lower().endswith(('.txt', '.log')):
            file_path = os.path.join(DAT_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                    lines = f.readlines()
            record = {'File': filename.replace('.log', '.csv')}
            # Extract first line time
            if lines:
                m = time_pattern.match(lines[0].strip())
                if m:
                    record['Time'] = m.group(1)
                else:
                    record['Time'] = ''
            else:
                record['Time'] = ''
            for line in lines:
                match = pattern.match(line.strip())
                if match:
                    key, value = match.groups()
                    if key in fields:
                        record[key] = value
            data_list.append(record)

    # Organize into DataFrame
    df_params = pd.DataFrame(data_list)

    # Arrange in field order
    df_params = df_params.reindex(columns=['File', 'Time'] + fields)

    # Sort by time (empty ones at the end)
    df_params['Time'] = pd.to_datetime(df_params['Time'], errors='coerce')
    df_params = df_params.sort_values('Time')

    # Export instrument parameters to Excel
    params_output_path = os.path.join(OUTPUT_DIR, 'Instrument_Parameters_Extraction_Result.xlsx')
    df_params.to_excel(params_output_path, index=False)
    print(f'Instrument parameter data saved to {params_output_path}')
    
    # Code 2: Extract isotope test values
    print("Extracting isotope test values...")
    result_list = []

    for filename in os.listdir(OUTPUT_DIR):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                df = pd.read_csv(file_path, sep=',', header=0, encoding='utf-8')  # header=0 means first row is header
            except Exception:
                try:
                    df = pd.read_csv(file_path, sep='\t', header=0, encoding='utf-8')
                except Exception:
                    df = pd.read_csv(file_path, sep='\t', header=0, encoding='gbk')
            
            # Get headers
            headers = list(df.columns)
            
            mean_values = []
            stderr_values = []
            
            # Find Mean and StdErr (abs) rows
            for i in range(len(df)):
                # Check first column value
                first_col_value = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
                
                if first_col_value.strip() == '***':
                    # Check second column value
                    second_col_value = str(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else ""
                    
                    if second_col_value.strip() == 'Mean':
                        # Extract all values from Mean row (starting from third column)
                        mean_values = [df.iloc[i, j] if pd.notna(df.iloc[i, j]) else '' for j in range(2, len(df.columns))]
                    
                    elif second_col_value.strip() == 'StdErr (abs)':
                        # Extract all values from StdErr (abs) row (starting from third column)
                        stderr_values = [df.iloc[i, j] if pd.notna(df.iloc[i, j]) else '' for j in range(2, len(df.columns))]
            
            print(f"{filename} isotope data extraction completed")
            
            # Directly expand to multiple columns
            result_list.append([filename] + mean_values + stderr_values)

    # Construct headers - use original CSV file headers (skip first two columns)
    if result_list:  # Ensure there is data
        data_headers = headers[2:]  # Skip Cycle and Time columns
        
        # Create complete column names
        mean_columns = [f'{header}' for header in data_headers]
        stderr_columns = [f'StdErr(abs){header}' for header in data_headers]
        columns = ['File'] + mean_columns + stderr_columns

        df_isotopes = pd.DataFrame(result_list, columns=columns)

        # Delete columns that are empty except for headers
        columns_to_keep = []
        for col in df_isotopes.columns:
            # Check if the column has at least one non-empty value (besides the header itself)
            if col == 'File' or df_isotopes[col].notna().any() and (df_isotopes[col] != '').any():
                columns_to_keep.append(col)

        # Keep only columns with data
        df_isotopes = df_isotopes[columns_to_keep]

        # Export isotope test values to Excel
        isotopes_output_path = os.path.join(OUTPUT_DIR, 'Isotope_Test_Values_Extraction_Result.xlsx')
        df_isotopes.to_excel(isotopes_output_path, index=False)
        print(f'Isotope test value data saved to {isotopes_output_path}')
        
        # Code 3: Merge two tables and add label column
        print("Merging data and adding labels...")
        # Merge using "File" as primary key, keeping all file data
        merged = pd.merge(df_params, df_isotopes, on='File', how='left')  # Use df_params order as primary (already sorted)
        
        # Add label column
        def assign_label(filename):
            """Assign labels based on filename"""
            filename_str = str(filename)
            if '3133' in filename_str:
                return '3133'
            elif '3177' in filename_str:
                return '3177'
            elif '8610' in filename_str:
                return '8610'
            else:
                return 'sample'
        
        # Insert label column before first column
        merged.insert(0, 'Label', merged['File'].apply(assign_label))
        
        # Save merge result
        final_output_path = os.path.join(OUTPUT_DIR, 'Before_Fractionation_Calculation.xlsx')
        merged.to_excel(final_output_path, index=False)
        print(f'Merge completed, final result saved to: {final_output_path}')
        print(f"Label statistics:")
        print(merged['Label'].value_counts())
        
        # Code 4: Fractionation calculation
        print("Performing fractionation calculation...")
        
        # Ensure column names have no extra spaces
        merged.columns = merged.columns.str.strip()
        merged['Label'] = merged['Label'].astype(str).str.strip()

        # Column mapping for processing and target columns
        col_map = {
            '200Hg': 'R-THg(%)',
            '199Hg/198Hg (4)': 'd199(‰)',
            '200Hg/198Hg (3)': 'd200(‰)',
            '201Hg/198Hg (2)': 'd201(‰)',
            '202Hg/198Hg (1)': 'd202(‰)'
        }

        # Ensure all required columns are numeric
        for src_col in col_map.keys():
            if src_col in merged.columns:
                merged[src_col] = pd.to_numeric(merged[src_col], errors='coerce')
            else:
                print(f"Warning: Column '{src_col}' does not exist, skipping fractionation calculation for this column")

        # Find all indices of rows with label 3133
        idx_3133 = merged.index[merged['Label'] == '3133'].tolist()

        def find_nearest_3133(idx):
            """Find the nearest 3133 label indices before and after current index"""
            prev_idx = [i for i in idx_3133 if i < idx]
            next_idx = [i for i in idx_3133 if i > idx]
            return prev_idx[-1] if prev_idx else None, next_idx[0] if next_idx else None

        def calc_fractionation(row, col):
            """Calculate fractionation value"""
            idx = row.name
            cur_val = row[col]
            prev_idx, next_idx = find_nearest_3133(idx)
            if prev_idx is None or next_idx is None or pd.isna(cur_val):
                return np.nan
            
            prev_val = merged.loc[prev_idx, col]
            next_val = merged.loc[next_idx, col]
            if pd.isna(prev_val) or pd.isna(next_val) or (prev_val + next_val) == 0:
                return np.nan
            
            # Calculate fractionation value
            fractionation_value = ((2 * cur_val) / (prev_val + next_val) - 1) * 1000
            
            # If it's THg(%), first divide by 1000, then convert to percentage
            if col == '200Hg':
                fractionation_value = (fractionation_value / 1000) * 100
            
            return fractionation_value

        # Calculate all target columns
        for src_col, tgt_col in col_map.items():
            if src_col in merged.columns:
                merged[tgt_col] = merged.apply(lambda row: calc_fractionation(row, src_col), axis=1)
                print(f"Completed {src_col} -> {tgt_col} fractionation calculation")
            else:
                print(f"Skipping {src_col} -> {tgt_col} fractionation calculation (source column does not exist)")

        # Code 5: Calculate D199, D200, D201
        print("Calculating D199(‰), D200(‰), D201(‰)...")
        
        # Ensure d202, d199, d200, d201 columns exist and are numeric
        required_cols = ['d202(‰)', 'd199(‰)', 'd200(‰)', 'd201(‰)']
        for col in required_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors='coerce')
            else:
                print(f"Warning: Column '{col}' does not exist, cannot calculate related D values")
        
        # Calculate D199, D200, D201
        if all(col in merged.columns for col in required_cols):
            merged['D199(‰)'] = merged['d199(‰)'] - merged['d202(‰)'] * 0.252
            merged['D200(‰)'] = merged['d200(‰)'] - merged['d202(‰)'] * 0.5024
            merged['D201(‰)'] = merged['d201(‰)'] - merged['d202(‰)'] * 0.752
            print("D199(‰), D200(‰), D201(‰) calculation completed")
        else:
            print("Cannot calculate D199(‰), D200(‰), D201(‰) due to missing required columns")

        # Rename column headers for final output
        merged.rename(columns={
            "Ampl.-Temp[癈]": "Ampl.-Temp[°C]",
            "202Hg/198Hg (1)": "202Hg/198Hg",
            "201Hg/198Hg (2)": "201Hg/198Hg",
            "200Hg/198Hg (3)": "200Hg/198Hg",
            "199Hg/198Hg (4)": "199Hg/198Hg",
            "205Tl/203Tl (5)": "205Tl/203Tl",
            "StdErr(abs)202Hg/198Hg (1)": "StdErr(abs)202Hg/198Hg",
            "StdErr(abs)201Hg/198Hg (2)": "StdErr(abs)201Hg/198Hg",
            "StdErr(abs)200Hg/198Hg (3)": "StdErr(abs)200Hg/198Hg",
            "StdErr(abs)199Hg/198Hg (4)": "StdErr(abs)199Hg/198Hg",
            "StdErr(abs)205Tl/203Tl (5)": "StdErr(abs)205Tl/203Tl"
        }, inplace=True)

        # Save final result
        fractionation_output_path = os.path.join(OUTPUT_DIR, 'Before_Fractionation_Calculation_Completed.xlsx')
        merged.to_excel(fractionation_output_path, index=False)
        print(f'Fractionation calculation completed, final result saved to: {fractionation_output_path}')
        
    else:
        print("No isotope test data found, skipping merge and fractionation calculation steps")

if __name__ == "__main__":
    main()