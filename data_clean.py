import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def cleandata(input_file, output_file):
     try:
          # Load the CSV file into a DataFrame
          df = pd.read_csv(input_file)

          # Drop columns where all values are 0
          df = df.loc[:, ~((df.eq(0) | df.isna()).all())]

          #Dropping any duplicate columns if existing
          df = df.loc[:, ~df.columns.duplicated()]

          # Convert all object-type columns to lowercase and strip whitespace
          for column in df.select_dtypes(include=['object']).columns:
               df[column] = df[column].str.strip().str.lower()

          # Mapping of input files to features (columns) to retain
          file_feature_map = { # need to adjust later
               "Host Events/EVSE-B-HPC.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "State", "Attack", "Scenario", "Label", "interface"],  # need to adjust later
               "Host Events/EVSE-B-Kernel-Events.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "syscalls:sys_enter_close", "syscalls:sys_exit_close",  "State", "Attack", "Attack-Group", "Label", "interface"],    
               "Power Consumption/EVSE-B-PowerCombined.csv": ["time", "shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State", "Attack", "Attack-Group", "Label", "interface"], 
          }
     
          # Get the features to include based on the input file
          if input_file in file_feature_map:
               selected_columns = file_feature_map[input_file]
          else:
               selected_columns = df.columns  # If no mapping exists, retain all columns
          
          # Select the specified columns
          df = df[selected_columns]

          #finding missing data
          #numerical features
          numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
          #numerical_cols = [col for col in numerical_cols if df[col].std() != 0]

          #categorical features
          categorical_cols = df.select_dtypes(include=['object']).columns

          # filling missing values for numerical columns using the mean
          #for col in numerical_cols:
          #     df[col] = df[col].fillna(df[col].mean())  # Reassign the filled column

          for col in numerical_cols:
               if (df[col] == 0).sum() > 0:  # Check if there are any 0 values
                    df[col] = df[col].replace(0, df[col].mean())  # Replace 0 with the mean of the column

          # filling missing values for categorical columns using mode
          #for col in categorical_cols:
          #     df[col] = df[col].fillna(df[col].mode()[0])  # Reassign the filled column

          # normalizing features
          # minmax scaler
          #scaler = MinMaxScaler()
          #df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

          # standard scaler
          scaler = StandardScaler()
          #df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

          #encoding categorical columns using one-hot encoding or if the number of unique values is greater than 100, use label encoding
          #df = pd.get_dummies(df, columns=categorical_cols)
          for col in categorical_cols:
               if df[col].nunique() > 100:  # Example threshold
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
               else:
                    df = pd.get_dummies(df, columns=[col])


          # Save the modified DataFrame to a new CSV file
          df.to_csv(output_file, index=False)

     except Exception as e:
        print(f"Error processing {input_file}: {e}")

# files
input_files = ["Host Events/EVSE-B-HPC.csv", "Host Events/EVSE-B-Kernel-Events.csv", "Power Consumption/EVSE-B-PowerCombined.csv"]  # Replace with your CSV file paths
output_files = ["Host Events/EVSE-B-HPC-cleaned.csv", "Host Events/EVSE-B-Kernel-Events-cleaned.csv", "Power Consumption/EVSE-B-PowerCombined-cleaned.csv"]  # Replace with your desired output file path

if len(input_files) != len(output_files):
    raise ValueError("The number of input files and output files must be the same.")

# Process each file
for input_file, output_file in zip(input_files, output_files):
     cleandata(input_file, output_file)
