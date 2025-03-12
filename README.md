Deep Neural Network (DNN) for Classification
============================================
This repository contains a PyTorch-based Deep Neural Network (DNN) implementation for multi-class classification tasks. The code includes preprocessing, training, evaluation, and testing functionalities. Below is a detailed explanation of the code and its components.

##### The main dataset used in this code is the EVSE-B-HPC.csv file located in the Host Events directory, and only the nonaugdnn.py works correctly for now

Overview
------------
The code implements a DNN model for classifying network data based on attack types into multiple categories. It includes:

Data preprocessing (cleaning, scaling, encoding, feature selection).
A DNN architecture with dropout and batch normalization.
Training with early stopping, learning rate scheduling, and mixed precision training.
Cross-validation for model evaluation.
Error analysis and debugging tools.
Visualization of results (confusion matrix, PCA, etc.).

Dependencies
--------------
The code requires the following Python libraries:

**PyTorch**: For building and training the DNN.
**Pandas:** For data manipulation and preprocessing.
**NumPy:** For numerical operations.
**Scikit-learn:** For preprocessing, feature selection, and evaluation metrics.
**Imbalanced-learn:** For handling imbalanced datasets (e.g., SMOTE).
**Matplotlib and Seaborn:** For visualization.
**SciPy:** For statistical operations (e.g., Z-score).

Install the dependencies using: 
'''
bash
pip install torch pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy
'''

Code Structure
-------------------

Below is a detailed explanation of each function.

---

### **1. `DNN` Class**
This class defines the architecture of the Deep Neural Network.

#### **Attributes:**
- **`fc1`, `fc2`, `fc3`, `fc4`**: Fully connected (linear) layers.
- **`bn1`, `bn2`, `bn3`**: Batch normalization layers to stabilize and speed up training.
- **`relu`**: ReLU activation function for introducing non-linearity.
- **`dropout`**: Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 during training.

#### **Methods:**
- **`__init__(self, inputSize, hiddenSize, outputSize)`**:
  - Initializes the layers of the DNN.
  - `inputSize`: Number of input features.
  - `hiddenSize`: Number of neurons in the first hidden layer.
  - `outputSize`: Number of output classes.
- **`forward(self, x)`**:
  - Defines the forward pass of the network.
  - Applies ReLU activation, batch normalization, and dropout after each layer.
  - The final layer (`fc4`) outputs raw logits (no activation) for use with `CrossEntropyLoss`.

---

### **2. `dataPrep` Function**
Prepares the dataset for training by cleaning, encoding, scaling, and feature selection.

#### **Parameters:**
- **`inputFile`**: Path to the input CSV file.
- **`outputFile`**: Path to save the cleaned dataset.
- **`fitScaler`**: Pre-fitted scaler.
- **`fitEncoder`**: Pre-fitted label encoder.

#### **Steps:**
1. **Load Data**:
   - Reads the CSV file into a Pandas DataFrame.
2. **Clean Data**:
   - Drops columns with all NaN or zero values.
   - Removes duplicate columns.
   - Converts categorical columns to lowercase and strips whitespace.
3. **Feature Selection**:
   - Selects specific columns based on the input file using a `fileFeatures` dictionary.
4. **Separate Features and Target**:
   - `X`: Features (all columns except the target).
   - `y`: Target column (`Attack`).
5. **Preprocess Numerical Columns**:
   - Converts numerical columns to float.
   - Scales numerical features using `MinMaxScaler`.
6. **Encode Categorical Columns**:
   - Uses `LabelEncoder` to encode categorical features.
7. **Feature Selection**:
   - Selects the top 20 features using mutual information (`SelectKBest`).
8. **Encode Target Variable**:
   - Encodes the target variable (`y`) using `LabelEncoder`.
9. **Convert to PyTorch Tensors**:
   - Converts `X` and `y` to PyTorch tensors for training.

#### **Returns:**
- `X_tensor`: Features as a PyTorch tensor.
- `y_tensor`: Target labels as a PyTorch tensor.
- `inputSize`: Number of input features.
- `targetVarEnc`: Fitted `LabelEncoder` for the target variable.
- `scaler`: Fitted `MinMaxScaler` for numerical features.
- `labelEnc`: Dictionary of fitted `LabelEncoder` objects for categorical features.

---

### **3. `removeZeroVar` Function**
Removes features with zero variance from the dataset.

#### **Parameters:**
- **`X_train`**: Training features.
- **`X_test`**: Test features.

#### **Steps:**
1. Calculates the variance of each feature in the training set.
2. Identifies features with zero variance.
3. Removes these features from both the training and test sets.

#### **Returns:**
- `X_train`: Training features with zero-variance features removed.
- `X_test`: Test features with zero-variance features removed.
- `zeroVarIndex`: Indices of the removed features.

---

### **4. `removeOutliers` Function**
Removes outliers from the dataset using Z-score.

#### **Parameters:**
- **`X`**: Features.
- **`y`**: Target labels.
- **`threshold`**: Z-score threshold for identifying outliers (default: 3).

#### **Steps:**
1. Converts `X` and `y` to NumPy arrays if they are PyTorch tensors.
2. Computes the mean and standard deviation of each feature.
3. Calculates the Z-score for each feature.
4. Identifies and removes outliers based on the Z-score threshold.
5. Converts the cleaned data back to PyTorch tensors.

#### **Returns:**
- `X_cleaned`: Features with outliers removed.
- `y_cleaned`: Target labels with outliers removed.

---

### **5. `debugTestSet` Function**
Debugs the test set by comparing it with the training set.

#### **Parameters:**
- **`X_train`**: Training features.
- **`y_train`**: Training labels.
- **`X_test`**: Test features.
- **`y_test`**: Test labels.
- **`numCols`**: Indices of numerical columns.
- **`catCols`**: Indices of categorical columns.

#### **Steps:**
1. Converts `X_train` and `X_test` to NumPy arrays if they are PyTorch tensors.
2. Removes zero-variance features.
3. Compares feature distributions using KDE plots.
4. Compares class distributions using count plots.
5. Checks for missing values and invalid labels.
6. Detects outliers using Z-scores.
7. Visualizes the data using PCA.
8. Compares performance with a baseline logistic regression model.

---

### **6. `errorAnalysis` Function**
Analyzes misclassified instances to check model errors.

#### **Parameters:**
- **`model`**: Trained DNN model.
- **`X_test`**: Test features.
- **`y_test`**: Test labels.
- **`targetVarEnc`**: Fitted `LabelEncoder` for the target variable.

#### **Steps:**
1. Moves `X_test` and `y_test` to the same device as the model.
2. Computes predictions using the model.
3. Identifies misclassified instances.
4. Prints details of the first 10 misclassified instances.

---

### **7. `trainModel` Function**
Trains the DNN model.

#### **Parameters:**
- **`model`**: DNN model to train.
- **`X_train`**: Training features.
- **`y_train`**: Training labels.
- **`X_val`**: Validation features.
- **`y_val`**: Validation labels.
- **`epochNum`**: Number of epochs.
- **`learnRate`**: Learning rate.

#### **Steps:**
1. Moves the model and data to the appropriate device (GPU if available).
2. Computes class weights to handle imbalanced data.
3. Defines the loss function (`CrossEntropyLoss`), optimizer (`AdamW`), and learning rate scheduler.
4. Trains the model using mini-batch gradient descent.
5. Implements early stopping if validation loss does not improve for 5 epochs.
6. Prints metrics (loss, accuracy, precision, recall, F1-score) for each epoch.

---

### **8. `crossValModel` Function**
Performs k-fold cross-validation to evaluate the model.

#### **Parameters:**
- **`model`**: DNN model.
- **`X`**: Features.
- **`y`**: Labels.
- **`epochNum`**: Number of epochs.
- **`learnRate`**: Learning rate.
- **`k`**: Number of folds.

#### **Steps:**
1. Splits the data into `k` folds.
2. Trains and evaluates the model on each fold.
3. Computes metrics (accuracy, precision, recall, F1-score) for each fold.
4. Prints average metrics across all folds.
5. Visualizes metrics using box plots.

---

### **9. `testModel` Function**
Evaluates the trained model on the test set.

#### **Parameters:**
- **`model`**: Trained DNN model.
- **`X_test`**: Test features.
- **`y_test`**: Test labels.

#### **Steps:**
1. Moves `X_test` and `y_test` to the same device as the model.
2. Computes predictions using the model.
3. Calculates metrics (accuracy, precision, recall, F1-score).
4. Generates a confusion matrix for visualization.

#### **Returns:**
- `y_pred`: Predicted labels for further analysis.

---

### **10. Main**
- Loads and preprocesses the dataset.
- Splits the data into training and validation sets.
- Initializes and trains the DNN model.
- Performs cross-validation.
- Tests the model on the test set.
- Performs error analysis and debugging.

Usage
--------------

> Run the _nonaugdnn.py_ file to run the model on the original dataset with no augmentation.

Key Features
Handles Imbalanced Data: Uses class weighting for the imbalanced dataset.
Feature Selection: Selects top features using mutual information.
Mixed Precision Training: Training runs on GPUs.
Early Stopping: Prevents overfitting by checking validation loss.
Visualization: Provides insights into data distribution, model performance, and errors.

Results and Evaluation
---------------------
Metrics: Accuracy, precision, recall, and F1-score are computed for each fold during cross-validation.

Confusion Matrix: Visualizes the model's performance on the test set.

PCA Visualization: Helps understand the clustering of data points.

