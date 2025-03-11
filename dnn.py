import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        #self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        #self.fc4 = nn.Linear(hidden_size // 4, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # Increase dropout slightly

        #self.softmax = nn.Softmax(dim=1)  # For multi-class classification (final output layer)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        #x = self.fc4(x)  # No activation; handled in CrossEntropyLoss
        return x

# Function to prepare data for the DNN
def prepare_data(input_file, output_file, fit_scaler=None, fit_encoder=None):
    # Check if the augmented file already exists
    if os.path.exists(output_file):
        print(f"Loading pre-augmented data from {output_file}")
        df = pd.read_csv(output_file)
        
        # Separate features and target
        X = df.drop(columns=['Attack'])
        y = df['Attack']
        
        # Encode categorical columns in X
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit_encoder is None:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            else:
                le = fit_encoder[col]
                X[col] = le.transform(X[col].astype(str))
        
        # Encode the target variable y
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)

        # Initialize and fit the scaler on the loaded data
        if fit_scaler is None:
            scaler = StandardScaler()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        else:
            scaler = fit_scaler
            X[numerical_cols] = scaler.transform(X[numerical_cols])
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        
        return X_tensor, y_tensor, X.shape[1], le_target, scaler, label_encoders

    # If the augmented file does not exist, perform augmentation
    print(f"Augmented file not found. Performing augmentation and saving to {output_file}")
    
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
        "Host Events/EVSE-B-HPC.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "State", "Attack", "Scenario", "Label", "interface"],
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

    print(f"Class distribution for {input_file} before resampling: \n{y.value_counts()}")

    # Convert numerical columns to float, handling errors
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace 0 values with the mean of the column
    for col in numerical_cols:
        X[col] = X[col].astype(float)
        for i in range(len(X)):
            if X.at[i, col] == 0:
                col_mean = X[col].replace(0, np.nan).mean()
                X.at[i, col] = col_mean

    # Apply scaling
    if fit_scaler is None:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        scaler = fit_scaler
        X[numerical_cols] = scaler.transform(X[numerical_cols])

    # Encode categorical columns in X
    label_encoders = {}

    for col in categorical_cols:
        if fit_encoder is None:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        else:
            le = fit_encoder[col]
            X[col] = le.transform(X[col].astype(str))
    
    # Identify categorical feature indices for SMOTENC
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]

    # Perform resampling using SMOTENC if necessary
    if y.value_counts().min() <= 1:
        print(f"Skipping SMOTE for {input_file} due to insufficient samples in a class.")
        df.to_csv(output_file, index=False)
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)  # Encode y
        X_tensor = torch.tensor(X.values.copy(), dtype=torch.float32)  # Make the array writable
        y_tensor = torch.tensor(y_encoded.copy(), dtype=torch.long)    # Make the array writable
        return X_tensor, y_tensor, X.shape[1], le_target, scaler, label_encoders  # Return the tensors and encoded labels
        return

    # Use SMOTENC or SMOTE depending on whether categorical features are present
    if categorical_indices:
        k_neighbors = max(1, min(3, y.value_counts().min() - 1))
        smote = SMOTENC(categorical_features=categorical_indices, random_state=42, k_neighbors=k_neighbors)
    else:
        smote = SMOTE(random_state=42)

    # Resample and save the modified data
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine original and augmented data
    X_combined = pd.concat([X, X_resampled], axis=0)
    y_combined = pd.concat([y, y_resampled], axis=0)

    # Shuffle the combined dataset
    combined_df = pd.concat([X_combined, y_combined], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to file
    combined_df.to_csv(output_file, index=False)

    print(f"Class distribution after augmentation:\n{y_combined.value_counts()}")

    # Encode the target variable y
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_combined)  # Ensure y is treated as strings

    input_size = X_combined.shape[1]  # Access the number of columns (features) from the numpy array    

    # Visualize original and augmented data using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_combined = np.vstack([X, X_resampled])
    y_combined = np.hstack([y, y_resampled])
    X_embedded = tsne.fit_transform(X_combined)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_combined, alpha=0.5)
    plt.title("t-SNE Visualization of Original and Augmented Data")
    plt.show()

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_combined.values.copy(), dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded.copy(), dtype=torch.long)  # Use long dtype for integer labels

    return X_tensor, y_tensor, input_size, le_target, scaler, label_encoders

def remove_zero_variance_features(X_train, X_test):
    # Calculate variance along the columns (features)
    variances = np.var(X_train, axis=0, ddof=0)  # ddof=0 for population variance
    
    # Identify indices of features with zero variance
    zero_variance_indices = np.where(variances == 0)[0]
    
    # Remove zero-variance features from both training and test sets
    X_train = np.delete(X_train, zero_variance_indices, axis=1)
    X_test = np.delete(X_test, zero_variance_indices, axis=1)
    
    return X_train, X_test, zero_variance_indices

# Function to remove outliers using Z-score
def remove_outliers(X, y, threshold=3):
    z_scores = np.abs(zscore(X))
    outlier_indices = np.where(z_scores > threshold)
    X_cleaned = np.delete(X, outlier_indices[0], axis=0)
    y_cleaned = np.delete(y, outlier_indices[0], axis=0)
    return X_cleaned, y_cleaned

# Function to debug the test set
def debug_test_set(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols):

    # Ensure X_train and X_test are NumPy arrays
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.cpu().numpy()

     # Debug: Print input shapes
    print(f"X_train shape before removal: {X_train.shape}")
    print(f"X_test shape before removal: {X_test.shape}")
    print(f"numerical_cols: {numerical_cols}")

    # Remove zero-variance features
    X_train, X_test, zero_variance_indices = remove_zero_variance_features(X_train, X_test)
    print(f"Removed zero-variance features at indices: {zero_variance_indices}")

    # Debug: Print shapes after removal
    print(f"X_train shape after removal: {X_train.shape}")
    print(f"X_test shape after removal: {X_test.shape}")

    # 1. Compare feature distributions
    # Update numerical_cols to reflect the new column indices
    # Create a list of remaining column indices after removing zero-variance features
    remaining_indices = [i for i in range(X_train.shape[1])]
    print(f"Remaining column indices: {remaining_indices}")

    # Filter numerical_cols to only include indices that are in remaining_indices
    numerical_cols = [col_idx for col_idx in numerical_cols if col_idx in remaining_indices]
    print(f"Updated numerical_cols: {numerical_cols}")

    for col_idx in numerical_cols:
        plt.figure(figsize=(10, 4))
        # Update the KDE plot code
        sns.kdeplot(X_train[:, col_idx], label="Train")
        sns.kdeplot(X_test[:, col_idx], label="Test")
        plt.legend(loc="upper right")  # Use a specific location
        plt.title(f"Distribution of {col_idx}")
        plt.legend()
        plt.show()
    
    # 2. Compare class distributions
    plt.figure(figsize=(10, 4))
    sns.countplot(y_train, label="Train")
    sns.countplot(y_test, label="Test")
    plt.title("Class Distribution")
    plt.legend()
    plt.show()

    # 3. Check for missing values
    print("Missing values in test set:", np.isnan(X_test).sum())

     # 4. Check for invalid labels
    valid_labels = np.unique(y_train)
    invalid_labels = [label for label in np.unique(y_test) if label not in valid_labels]
    print("Invalid labels in test set:", invalid_labels)

    # 5. Detect outliers using Z-score
    z_scores = np.abs(zscore(X_test))
    outliers = np.where(z_scores > 3)
    print("Outliers in test set:", outliers)


    # 6. Visualize test set using PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], label="Train", alpha=0.5)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], label="Test", alpha=0.5)
    plt.legend()
    plt.title("PCA Visualization of Training and Test Sets")
    plt.show()

    # 7. Compare with a baseline model
    baseline_model = LogisticRegression()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    print("Baseline model accuracy:", accuracy_score(y_test, y_pred_baseline))

# Function to train the DNN
def train_dnn(model, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) #L2 regularization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs
    scaler = GradScaler()  # For mixed precision training

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    best_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):  # Mixed precision
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))

            # Convert logits to predicted class labels
            _, predicted = torch.max(val_outputs, 1)  
            val_accuracy = (predicted == y_val.to(device)).float().mean().item()

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

        scheduler.step()

def cross_validate_dnn(model, X, y, num_epochs=50, learning_rate=0.005, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    evalMetrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n==== Fold {fold+1}/{k} ====")
        
        # Split data into training and validation for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize model for each fold
        model = DNN(input_size, hidden_size, output_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        # Train model
        train_dnn(model, X_train, y_train, X_val, y_val, num_epochs=num_epochs, learning_rate=learning_rate)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs, 1)
        
        y_true = y_val.cpu().numpy()
        y_pred = predicted.cpu().numpy()

        # Compute metrics
        acc = (predicted == y_val).float().mean().item()
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Store metrics
        evalMetrics["accuracy"].append(acc)
        evalMetrics["precision"].append(prec)
        evalMetrics["recall"].append(rec)
        evalMetrics["f1_score"].append(f1)

        print(f"Fold {fold+1} Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    
    # Print average metrics across all folds
    print("\n=== Cross-Validation Results ===")
    for metric in evalMetrics:
        print(f"{metric.capitalize()}: {np.mean(evalMetrics[metric]):.4f} ± {np.std(evalMetrics[metric]):.4f}")

def test_dnn(model, X_test, y_test):
    device = next(model.parameters()).device  # Get the device of the model
    X_test = X_test.to(device)  # Move X_test to the same device as the model
    y_test = y_test.to(device)  # Move y_test to the same device as the model

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
    X_tensor, y_tensor, input_size, le_target, scaler, label_encoders = prepare_data(input_file, output_file)

    # Analyze class distribution
    print("Class Distribution Analysis:")
    class_counts = pd.Series(y_tensor.numpy()).value_counts()
    class_proportions = class_counts / len(y_tensor)
    print("Class Proportions:\n", class_proportions)

    majority_class_count = class_counts.max()
    minority_class_count = class_counts.min()
    imbalance_ratio = majority_class_count / minority_class_count
    print(f"Imbalance Ratio: {imbalance_ratio}")

    # Remove outliers from the entire dataset before splitting
    X_cleaned, y_cleaned = remove_outliers(X_tensor, y_tensor)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

    # Determine output size based on the number of unique classes
    output_size = len(le_target.classes_)  # For multiclass classification, this is the number of attack types

    # Initialize the DNN model
    hidden_size = 512  # Adjust based on dataset size and complexity
    model = DNN(input_size, hidden_size, output_size)

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is NOT available. Training will run on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(next(model.parameters()).device)

    # Train the DNN
    #train_dnn(model, X_train, y_train, X_val, y_val, num_epochs=50, learning_rate=0.005)
    # Run Cross-Validation
    cross_validate_dnn(model, X_train, y_train, num_epochs=50, learning_rate=0.005, k=5)

    # Load and preprocess test data
    test_file = "Host Events/EVSE-B-HPC-cleaned.csv"  # Adjust with your test file
    X_test_tensor, y_test_tensor, _, _, _, _ = prepare_data(test_file, "test-cleaned.csv")

    # Remove outliers from the test set
    X_test_cleaned, y_test_cleaned = remove_outliers(X_test_tensor, y_test_tensor)

    # Debug the test set
    numerical_cols = [i for i in range(X_test_cleaned.shape[1])]  # Use indices for numerical columns
    # Convert numerical_cols to indices if they are column names
    if isinstance(numerical_cols[0], str):
        numerical_cols = [i for i, col in enumerate(numerical_cols)]
        
    # Apply the function to the test set
    debug_test_set(X_train, y_train, X_test_cleaned.cpu().numpy(), y_test_cleaned.cpu().numpy(), numerical_cols=numerical_cols, categorical_cols=[])

    # Test the trained model
    test_dnn(model, X_test_cleaned, y_test_cleaned)
