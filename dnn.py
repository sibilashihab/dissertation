import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, make_scorer, accuracy_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skorch import NeuralNetClassifier
from sklearn.manifold import TSNE

class DNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.bn1 = nn.BatchNorm1d(hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.bn2 = nn.BatchNorm1d(hiddenSize // 2)
        self.fc3 = nn.Linear(hiddenSize // 2, hiddenSize // 4)
        self.bn3 = nn.BatchNorm1d(hiddenSize // 4) #trying to change layer size to see if outputs change
        self.fc4 = nn.Linear(hiddenSize // 4, outputSize) 

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # Increase dropout slightly

        self.softmax = nn.Softmax(dim=1)  # For multi-class classification (final output layer)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)  # No activation; handled in CrossEntropyLoss
        return x

# Function to prepare data for the DNN
def dataPrep(inputFile, outputFile, fitScaler=None, fitEncoder=None, mode="train", selectedCols=None):

    # Check if the augmented file already exists to use for testing or validation
    if os.path.exists(outputFile) and mode in ["val", "test"]:
        print(f"Loading pre-augmented data from {outputFile} for {mode} set.")
        df = pd.read_csv(outputFile)
        
        # Separate features and target
        X = df.drop(columns=['Attack'])
        y = df['Attack']
        
        if selectedCols is not None:
            # Filter selectedCols to only include columns present in the dataset
            selectedCols = [col for col in selectedCols if col in X.columns]
            X = X[selectedCols]

        numCols = X.select_dtypes(include=['int64', 'float64']).columns  # Select numerical cols
        catCols = X.select_dtypes(include=['object']).columns  # Select categorical cols

        # Encode categorical columns in X
        labelEnc = {}
        for col in catCols:
            if fitEncoder is None:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                labelEnc[col] = le
            else:
                le = fitEncoder[col]
                X[col] = le.transform(X[col].astype(str))
        
        # Encode the target variable y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Initialize and fit the scaler on the loaded data
        if fitScaler is None:
            scaler = StandardScaler()
            X[numCols] = scaler.fit_transform(X[numCols])
        else:
            scaler = fitScaler
            X[numCols] = scaler.transform(X[numCols])
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        
        return X_tensor, y_tensor, X.shape[1], le, scaler, labelEnc

    # If the augmented file does not exist or mode is "train", perform augmentation
    print(f"Augmented file not found or mode is 'train'. Performing augmentation and saving to {outputFile}")
    
    df = pd.read_csv(inputFile)

    # Clean data by dropping columns with all NaN/Zero values
    df = df.loc[:, ~((df.eq(0) | df.isna()).all())]

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert all object-type columns to lowercase and strip whitespace
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype(str).str.strip().str.lower()

    # Column selection mapping for specific input files
    fileFeatures = {
        "Host Events/EVSE-B-HPC.csv": [ "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "State", "Attack", "Scenario", "Label", "interface"],
        "Host Events/EVSE-B-Kernel-Events.csv": ["time", "cpu_cycles", "cpu-cycles", "cpu-migrations", "context-switches", "cgroup-switches", "syscalls:sys_enter_close", "syscalls:sys_exit_close", "State", "Attack", "Attack-Group", "Label", "interface"],
        "Power Consumption/EVSE-B-PowerCombined.csv": ["time", "shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State", "Attack", "Attack-Group", "Label", "interface"],
    }

    # Filter columns based on the specific file
    selectedCols = fileFeatures.get(inputFile, df.columns)
    if selectedCols is None:
        selectedCols = fileFeatures.get(inputFile, df.columns)
    df = df[selectedCols]

    targetCol = 'Attack'
    X = df.drop(columns=[targetCol])
    y = df[targetCol]

    # Clean and preprocess numerical columns
    numCol = X.select_dtypes(include=['int64', 'float64']).columns
    catCols = X.select_dtypes(include=['object']).columns

    print(f"Class distribution for {inputFile} before resampling: \n{y.value_counts()}")

    # Convert numerical columns to float, handling errors
    for col in numCol:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace 0 values with the mean of the column
    for col in numCol:
        X[col] = X[col].astype(float)
        for i in range(len(X)):
            if X.at[i, col] == 0:
                col_mean = X[col].replace(0, np.nan).mean()
                X.at[i, col] = col_mean

    # Apply scaling
    if fitScaler is None:
        scaler = StandardScaler()
        X[numCol] = scaler.fit_transform(X[numCol])
    else:
        scaler = fitScaler
        X[numCol] = scaler.transform(X[numCol])

    # Encode categorical columns in X
    labelEnc = {}
    for col in catCols:
        if fitEncoder is None:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            labelEnc[col] = le
        else:
            le = fitEncoder[col]
            X[col] = le.transform(X[col].astype(str))
    
    # Identify categorical feature indices for SMOTENC
    catIndex = [X.columns.get_loc(col) for col in catCols]

    # Perform resampling using SMOTENC if necessary (only for training)
    if mode == "train":
        if y.value_counts().min() <= 1:
            print(f"Skipping SMOTE for {inputFile} due to insufficient samples in a class.")
        else:
            # Use SMOTENC or SMOTE depending on whether categorical features are present
            if catIndex:
                k_neighbors = max(1, min(3, y.value_counts().min() - 1))
                smote = SMOTENC(categorical_features=catIndex, random_state=42, k_neighbors=k_neighbors)
            else:
                smote = SMOTE(random_state=42)

            # Resample and save the modified data
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Combine original and augmented data
            X_combined = pd.concat([X, X_resampled], axis=0)
            y_combined = pd.concat([y, y_resampled], axis=0)

            # Shuffle the combined dataset
            combinedDf = pd.concat([X_combined, y_combined], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)

            # Save to file
            combinedDf.to_csv(outputFile, index=False)

            print(f"Class distribution after augmentation:\n{y_combined.value_counts()}")

            # Encode the target variable y
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_combined)  # Ensure y is treated as strings

            input_size = X_combined.shape[1]  # Access the number of columns (features) from the numpy array    

            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_combined.values.copy(), dtype=torch.float32)
            y_tensor = torch.tensor(y_encoded.copy(), dtype=torch.long)  # Use long dtype for integer labels

            return X_tensor, y_tensor, input_size, le, scaler, labelEnc, selectedCols

    # If mode is "val" or "test", return the original data without augmentation
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Encode y
    X_tensor = torch.tensor(X.values.copy(), dtype=torch.float32)  # Make the array writable
    y_tensor = torch.tensor(y_encoded.copy(), dtype=torch.long)    # Make the array writable
    return X_tensor, y_tensor, X.shape[1], le, scaler, labelEnc, selectedCols  # Return the tensors and encoded labels

def removeZeroVar(X_train, X_test):
    # Calculate variance along the columns (features)
    variances = np.var(X_train, axis=0, ddof=0)  # ddof=0 for population variance
    
    # Identify indices of features with zero variance
    zeroVarIndex = np.where(variances == 0)[0]
    
    # Remove zero-variance features from both training and test sets
    X_train = np.delete(X_train, zeroVarIndex, axis=1)
    X_test = np.delete(X_test, zeroVarIndex, axis=1)
    
    return X_train, X_test, zeroVarIndex

# Function to remove outliers using Z-score
def removeOutliers(X, y, threshold=3):
    # Calculate mean and standard deviation
    # Ensure X is a NumPy array
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Calculate mean and standard deviation
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Handle zero standard deviation (replace with a small value to avoid division by zero)
    std[std == 0] = 1e-8  # Replace zero std with a small epsilon (e.g., 1e-8)
    
    # Compute z-scores manually
    zScore = np.abs((X - mean) / std)
    
    # Identify outliers
    outlierIndex = np.where(zScore > threshold)
    X_cleaned = np.delete(X, outlierIndex[0], axis=0)
    y_cleaned = np.delete(y, outlierIndex[0], axis=0)
    
    # Convert back to PyTorch tensors if necessary
    X_cleaned = torch.tensor(X_cleaned, dtype=torch.float32)
    y_cleaned = torch.tensor(y_cleaned, dtype=torch.long)
    
    return X_cleaned, y_cleaned
 
# Function to debug the test set
#Remove later
def debugTestSet(X_train, y_train, X_test, y_test, numCols, catCols):

    # Ensuring X_train and X_test are NumPy arrays
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.cpu().numpy()

    # Debug: Print input shapes
    print(f"X_train shape before removal: {X_train.shape}")
    print(f"X_test shape before removal: {X_test.shape}")
    print(f"numCols: {numCols}")

    # Remove zero-variance features
    X_train, X_test, zeroVarIndex = removeZeroVar(X_train, X_test)
    print(f"Removed zero-variance features at indices: {zeroVarIndex}")

    # Printing shapes after removal
    print(f"X_train shape after removal: {X_train.shape}")
    print(f"X_test shape after removal: {X_test.shape}")

    # Comparing feature distributions
    # Create a list of remaining column indices after removing zero-variance features
    finalIndex = [i for i in range(X_train.shape[1])]
    #print(f"Remaining column indices: {finalIndex}")

    # Filter numCols to only include indices that are in finalIndex
    numCols = [colId for colId in numCols if colId in finalIndex]
    #print(f"Updated numCols: {numCols}")

    #Checking for each feature
    '''for colId in numCols:
        plt.figure(figsize=(10, 4))
        # Update the KDE plot code
        sns.kdeplot(X_train[:, colId], label="Train")
        sns.kdeplot(X_test[:, colId], label="Test")
        plt.legend(loc="upper right")
        plt.title(f"Distribution of {colId}")
        plt.legend()
        plt.show()
    '''
    
    # Compare class distributions
    '''
    plt.figure(figsize=(10, 4))
    sns.countplot(y_train, label="Train")
    sns.countplot(y_test, label="Test")
    plt.title("Class Distribution")
    plt.legend(loc="upper right")
    plt.show()
    '''

    # Check for missing values
    print("Missing values in test set:", np.isnan(X_test).sum())

    # Check for invalid labels
    validLabels = np.unique(y_train)
    invalidLabels = [label for label in np.unique(y_test) if label not in validLabels]
    print("Invalid labels in test set:", invalidLabels)

    # Detect outliers using Z-score
    zScore = np.abs(zscore(X_test))
    outliers = np.where(zScore > 3)
    #print("Outliers in test set:", outliers)

    # Visualize test set using PCA to see clustering
    pca = PCA(n_components=2)
    XtrainPCA = pca.fit_transform(X_train)
    XTestPCA = pca.transform(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(XtrainPCA[:, 0], XtrainPCA[:, 1], label="Train", alpha=0.5)
    plt.scatter(XTestPCA[:, 0], XTestPCA[:, 1], label="Test", alpha=0.5)
    plt.legend()
    plt.title("PCA Visualization of Training and Test Sets")
    plt.show()

    # Compare with a baseline model to see how the dataset performs with another ml model 3Using a logistic regression model 
    baseModel = LogisticRegression(max_iter=1000)
    baseModel.fit(X_train, y_train)
    yBaselinePred = baseModel.predict(X_test)
    print("Baseline model accuracy:", accuracy_score(y_test, yBaselinePred))

def errorAnalysis(model, X_test, y_test, targetVarEnc):
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    y_true = y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    # Identify misclassified instances
    misclassified = np.where(y_pred != y_true)[0]
    print(f"Number of misclassified instances: {len(misclassified)}")

    # Analyze misclassified instances
    for idx in misclassified[:10]:  # Print details for the first 10 misclassified instances
        print(f"True: {targetVarEnc.inverse_transform([y_true[idx]])}, Predicted: {targetVarEnc.inverse_transform([y_pred[idx]])}")
        print(f"Features: {X_test[idx].cpu().numpy()}")

# Function to train the DNN
def trainModel(model, X_train, y_train, X_val, y_val, epochNum=100, learnRate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Using gpu if available for faster performance
    model.to(device)

     # Calculate class weights
    class_counts = pd.Series(y_train.cpu().numpy()).value_counts().sort_index().values
    class_weights = 1.0 / class_counts  # Inverse of class counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Cross entropy loss for multi-class classification
    optimizer = optim.AdamW(model.parameters(), lr=learnRate, weight_decay=1e-4) #L2 regularization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs
    scaler = GradScaler()  # For mixed precision training

    trainingSet = torch.utils.data.TensorDataset(X_train, y_train)
    trainingLoader = torch.utils.data.DataLoader(trainingSet, batch_size=min(32, len(trainingSet)), shuffle=True, drop_last=True)

    bestLoss = float('inf')
    bestWeight = None    
    patience = 10
    counter = 0

    for epoch in range(epochNum):
        # Training Phase
        model.train()
        epLoss = 0

        for X_batch, y_batch in trainingLoader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):  # Mixed precision
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epLoss += loss.item()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            validationOutput = model(X_val.to(device))
            valLoss = criterion(validationOutput, y_val.to(device))

            # Save the best model
            if valLoss.item() < bestLoss:
                bestLoss = valLoss.item()
                bestWeight = model.state_dict().copy()

            # Convert logits to predicted class labels
            _, predicted = torch.max(validationOutput, 1)  
            valAccuracy = (predicted == y_val.to(device)).float().mean().item()

            # Convert tensors to numpy arrays for metric calculation
            y_true = y_val.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            # Compute Precision, Recall, and F1-score
            valPrecision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            valRecall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            valf1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            scheduler.step()

        # Print metrics for each epoch
        print(f"Epoch [{epoch+1}/{epochNum}], "
              f"Loss: {loss.item():.4f}, "
              f"Val Loss: {valLoss.item():.4f}, "
              f"Accuracy: {valAccuracy:.4f}, "
              f"Precision: {valPrecision:.4f}, "
              f"Recall: {valRecall:.4f}, "
              f"F1-score: {valf1:.4f}")

        # Early Stopping
        if valLoss.item() < bestLoss:
            bestLoss = valLoss.item()
            counter = 0
        else: 
            counter += 1
            if counter >= patience: # If model loss doesnt improve after 10 epochs stop the training to prevent overfitting
                print("Early stopping triggered!")
                break   
             
        model.load_state_dict(bestWeight) #Loading best model weights
        scheduler.step() #reduce learning rate

#Using cross validation
def crossValModel(model, X, y, epochNum=50, learnRate=0.005, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    evalMetrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
    
    for fold, (tID, vID) in enumerate(kf.split(X)):
        print(f"\n==== Fold {fold+1}/{k} ====")
        
        # Split data into training and validation for this fold
        X_train, X_val = X[tID], X[vID]
        y_train, y_val = y[tID], y[vID]
        
        # Check if validation set is too small
        if len(X_val) < 2:
            print(f"Skipping fold {fold+1} because validation set is too small.")
            continue

        # Initialize model for each fold
        model = DNN(inputSize, hiddenSize, outputSize)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        # Train model
        trainModel(model, X_train, y_train, X_val, y_val, epochNum=epochNum, learnRate=learnRate)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            validationOutput = model(X_val)
            _, predicted = torch.max(validationOutput, 1)
        
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

    # Visualize metrics using box plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[evalMetrics["accuracy"], evalMetrics["precision"], evalMetrics["recall"], evalMetrics["f1_score"]],
                notch=True, patch_artist=True)
    plt.xticks([0, 1, 2, 3], ["Accuracy", "Precision", "Recall", "F1-Score"])  # Set labels for x-axis
    plt.title("Cross-Validation Metrics Across Folds")
    plt.ylabel("Score")
    plt.show()

def testModel(model, X_test, y_test):
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

    #Confusion matrix for evaluation
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=targetVarEnc.classes_, yticklabels=targetVarEnc.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return y_pred  # Return predictions

# File paths for input and output
inputFiles = ["Host Events/EVSE-B-HPC.csv"]
outputFiles = ["Host Events/EVSE-B-HPC-cleaned.csv"]

for inputFile, outputFile in zip(inputFiles, outputFiles):
    X_tensor, y_tensor, inputSize, targetVarEnc, scaler, labelEnc, selectedCols = dataPrep(inputFile, outputFile, mode="train")

    # Analyze class distribution
    #print("Class Distribution Analysis:")
    classCounter = pd.Series(y_tensor.numpy()).value_counts()
    classProportion = classCounter / len(y_tensor)
    print("Class Proportions:\n", classProportion)

    majClass = classCounter.max()
    minClass = classCounter.min()
    imbRatio = majClass / minClass
    print(f"Imbalance Ratio: {imbRatio}")

    # Remove outliers from the entire dataset before splitting
    #X_cleaned, y_cleaned = removeOutliers(X_tensor, y_tensor)

    # Split the cleaned data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Determine output size based on the number of unique classes
    outputSize = len(targetVarEnc.classes_)  # For multiclass classification, this is the number of attack types

    # Initialize the DNN model
    hiddenSize = 512  # might need to adjust
    model = DNN(inputSize, hiddenSize, outputSize)

    # Check if CUDA is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is NOT available. Training will run on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(next(model.parameters()).device)

    # Train the DNN
    #trainModel(model, X_train, y_train, X_val, y_val, epochNum=50, learnRate=0.005)

    # Running Cross-Validation instead of training directly
    crossValModel(model, X_train, y_train, epochNum=50, learnRate=0.005, k=5)

    # Load and preprocess test data
    testFile = "Host Events/EVSE-B-HPC-cleaned.csv"
    XtestTensor, ytestTensor, _, _, _, _ = dataPrep(testFile, "test-cleaned.csv", fitScaler=scaler, fitEncoder=labelEnc, mode="test", selectedCols=selectedCols)

    # Remove outliers from the test set
    XtestClean, ytestClean = removeOutliers(XtestTensor, ytestTensor)

    numCols = [i for i in range(X_train.shape[1])]  # Use indices for numerical columns

    if isinstance(numCols[0], str): # Convert numCols to indices if they are column names
        numCols = [i for i, col in enumerate(numCols)]
        
    # Perform error analysis #just to find why model isnt performing well #remove later
    errorAnalysis(model, X_val, y_val, targetVarEnc)

    # Only for debugging to see how to improve model #remove later
    debugTestSet(X_train, y_train, XtestClean.cpu().numpy(), ytestClean.cpu().numpy(), numCols=numCols, catCols=[])

    # Test the trained model
    testModel(model, XtestClean, ytestClean)
