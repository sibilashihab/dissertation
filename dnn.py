import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.model_selection import train_test_split

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Additional hidden layer
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification
        self.softmax = nn.Softmax(dim=1)  # For multi-class classification

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out  # Return raw logits for loss computation

# Function to prepare data for the DNN
def prepare_data(input_file):
    df = pd.read_csv(input_file)
    target_column = 'Attack'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical columns in X
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode the target variable y
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y.astype(str))  # Ensure y is treated as strings

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long if len(le_target.classes_) > 2 else torch.float32)

    return X_tensor, y_tensor, len(X.columns), le_target

# Function to train the DNN
def train_dnn(model, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss() if len(torch.unique(y_train)) > 2 else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, predicted = torch.max(val_outputs, 1) if len(torch.unique(y_train)) > 2 else (torch.sigmoid(val_outputs) > 0.5).float()
            val_accuracy = (predicted == y_val).float().mean()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

# Load and preprocess data
input_files = ["Host Events/EVSE-B-HPC-cleaned.csv", "Host Events/EVSE-B-Kernel-Events-cleaned.csv", "Power Consumption/EVSE-B-PowerCombined-cleaned.csv"]
output_files = ["Host Events/EVSE-B-HPC-cleaned.csv", "Host Events/EVSE-B-Kernel-Events-cleaned.csv", "Power Consumption/EVSE-B-PowerCombined-cleaned.csv"]

for input_file, output_file in zip(input_files, output_files):
    X_tensor, y_tensor, input_size, le_target = prepare_data(input_file)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Determine output size based on the number of unique classes
    output_size = len(le_target.classes_) if len(le_target.classes_) > 2 else 1

    # Initialize the DNN model
    hidden_size = 128  # Adjust based on dataset size and complexity
    model = DNN(input_size, hidden_size, output_size)

    # Train the DNN
    train_dnn(model, X_train, y_train, X_val, y_val, num_epochs=50, learning_rate=0.001)