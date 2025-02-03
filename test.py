import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.model_selection import train_test_split

def cleandata(input_file, output_file):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Drop columns where all values are 0 or NaN
        df = df.loc[:, ~((df.eq(0) | df.isna()).all())]

        # Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert all object-type columns to lowercase and strip whitespace
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].astype(str).str.strip().str.lower()

        # Mapping of input files to features to retain
        file_feature_map = {
            "Host Events/EVSE-B-HPC.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "State", "Attack", "Scenario", "Label", "interface"],
            "Host Events/EVSE-B-Kernel-Events.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "syscalls:sys_enter_close", "syscalls:sys_exit_close", "State", "Attack", "Attack-Group", "Label", "interface"],
            "Power Consumption/EVSE-B-PowerCombined.csv": ["time", "shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State", "Attack", "Attack-Group", "Label", "interface"],
        }

        # Select specific columns based on file
        selected_columns = file_feature_map.get(input_file, df.columns)
        df = df[selected_columns]

        # Separate features and target variable ('State')
        target_column = 'Attack'
        X = df.drop(columns=[target_column])
        df = df.dropna(subset=[target_column])
        y = df[target_column]

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        print(f"Class distribution for {input_file} before resampling: \n{y.value_counts()}")

        # Convert numerical columns to float, handling errors
        for col in numerical_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Replace NaN values with the column mean
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

        # Encode categorical columns using LabelEncoder (avoid one-hot encoding)
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))  # Convert to string to avoid type issues
            label_encoders[col] = le  # Store for future use

        # Identify categorical feature indices for SMOTENC
        categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]

        if y.value_counts().min() <= 1:
               print(f"Skipping SMOTE for {input_file} due to insufficient samples in a class.")
               df.to_csv(output_file, index=False)
               return

        # Apply SMOTENC only if categorical features exist, otherwise use SMOTE
        if categorical_indices:
           k_neighbors = max(1, min(3, y.value_counts().min() - 1))
           smote = SMOTENC(categorical_features=categorical_indices, random_state=42, k_neighbors=max(1, min(3, y.value_counts().min() - 1)))
        else:
            smote = SMOTE(random_state=42)

        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Save the modified DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled
        df_resampled.to_csv(output_file, index=False)

        print(f"Class distribution for {input_file} after resampling: \n{y_resampled.value_counts()}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Files
input_files = ["Host Events/EVSE-B-HPC.csv", "Host Events/EVSE-B-Kernel-Events.csv", "Power Consumption/EVSE-B-PowerCombined.csv"]
output_files = ["Host Events/EVSE-B-HPC-cleaned.csv", "Host Events/EVSE-B-Kernel-Events-cleaned.csv", "Power Consumption/EVSE-B-PowerCombined-cleaned.csv"]

if len(input_files) != len(output_files):
    raise ValueError("The number of input files and output files must be the same.")

# Process each file
for input_file, output_file in zip(input_files, output_files):
    cleandata(input_file, output_file)
