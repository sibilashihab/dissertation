import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, output_size)

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.4)  # Increase dropout slightly

        #self.softmax = nn.Softmax(dim=1)  # For multi-class classification (final output layer)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)  # No activation; handled in CrossEntropyLoss
        return x

# Function to prepare data for the DNN
def prepare_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Clean data by dropping columns with all NaN/Zero values
    df = df.loc[:, ~((df.eq(0) | df.isna()).all())]

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert all object-type columns to lowercase and strip whitespace
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype(str).str.strip().str.lower()

    # Column selection mapping for specific input files
    file_feature_map = {
        "Host Events/EVSE-B-HPC.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "raw_syscalls_sys_enter", "raw_syscalls_sys_exit", "syscalls_sys_enter_close",  "syscalls_sys_enter_read", "syscalls_sys_enter_write", "syscalls_sys_exit_write", "syscalls_sys_exit_read", "syscalls_sys_exit_close",  "State", "Attack", "Scenario", "Label", "interface"],
        "Host Events/EVSE-B-Kernel-Events.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "syscalls:sys_enter_close", "syscalls:sys_exit_close", "State", "Attack", "Attack-Group", "Label", "interface"],
        "Power Consumption/EVSE-B-PowerCombined.csv": ["time", "shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State", "Attack", "Attack-Group", "Label", "interface"],
    }

    # Filter columns based on the specific file
    selected_columns = file_feature_map.get(input_file, df.columns)
    df = df[selected_columns]

    target_column = 'Attack'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Clean and preprocess numerical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Convert numerical columns to float, handling errors
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])  # Apply normalization

    # Encode categorical columns in X
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    df.to_csv(output_file, index=False)  # Save the cleaned/resampled file

    # Encode the target variable y
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)  # Ensure y is treated as strings

    input_size = X.shape[1]  # Access the number of columns (features) from the numpy array    

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)  # Use long dtype for integer labels

    return X_tensor, y_tensor, input_size, le_target

# Function to train the DNN
def train_dnn(model, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs

    best_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            # Convert logits to predicted class labels
            _, predicted = torch.max(val_outputs, 1)  
            val_accuracy = (predicted == y_val).float().mean().item()

            # Convert tensors to numpy arrays for metric calculation
            y_true = y_val.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            # Compute Precision, Recall, and F1-score
            val_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            scheduler.step()

        # Print metrics for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {loss.item():.4f}, "
              f"Val Loss: {val_loss.item():.4f}, "
              f"Accuracy: {val_accuracy:.4f}, "
              f"Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, "
              f"F1-score: {val_f1:.4f}")

        # Early Stopping
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

def test_dnn(model, X_test, y_test):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)  # Get raw logits
        _, predicted = torch.max(outputs, 1)  # Convert logits to class predictions

    # Convert tensors to numpy for evaluation
    y_true = y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    # Compute metrics
    accuracy = (predicted == y_test).float().mean().item()
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Print the results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")

    return y_pred  # Return predictions for further analysis if needed


# File paths for input and output
input_files = ["Host Events/EVSE-B-HPC.csv"]
output_files = ["Host Events/EVSE-B-HPC-cleaned.csv"]

for input_file, output_file in zip(input_files, output_files):
     X_tensor, y_tensor, input_size, le_target = prepare_data(input_file, output_file)

     # Split data into training and validation sets
     X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

     # Determine output size based on the number of unique classes
     output_size = len(le_target.classes_)  # For multiclass classification, this is the number of attack types

     # Initialize the DNN model
     hidden_size = 1024  # Adjust based on dataset size and complexity
     model = DNN(input_size, hidden_size, output_size)

     # Train the DNN
     train_dnn(model, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.001)

     # Load and preprocess test data
     test_file = "Host Events/EVSE-B-HPC-cleaned.csv"  # Adjust with your test file
     X_test_tensor, y_test_tensor, _, _ = prepare_data(test_file, "test-cleaned.csv")

     # Test the trained model
     test_dnn(model, X_test_tensor, y_test_tensor)
