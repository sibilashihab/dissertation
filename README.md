Deep Neural Network (DNN) for Classification
============================================
This repository contains a PyTorch-based Deep Neural Network (DNN) implementation for multi-class classification tasks. The code includes preprocessing, training, evaluation, and testing functionalities. Below is a detailed explanation of the code and its components.

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
> pip install torch pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy

Code Structure
-------------------
1. DNN Model (DNN class)
A fully connected neural network with:
Input layer, hidden layers, and output layer.
Batch normalization and dropout for regularization.
ReLU activation and softmax for multi-class classification.

2. Data Preparation (dataPrep function)
Cleans and preprocesses input data:

Drops columns with all NaN/zero values.
Converts categorical columns to lowercase and strips whitespace.
Encodes categorical variables using LabelEncoder.
Scales numerical features using MinMaxScaler.
Performs feature selection using mutual information.
Converts data into PyTorch tensors.

3. Training (trainModel function)
Trains the DNN using:

Cross-entropy loss with class weighting for imbalanced datasets.
AdamW optimizer with L2 regularization.
Learning rate scheduling and early stopping.
Mixed precision training for faster performance on GPUs.

4. Cross-Validation (crossValModel function)
Evaluates the model using k-fold cross-validation.
Computes and visualizes metrics (accuracy, precision, recall, F1-score).

5. Testing (testModel function)
Evaluates the trained model on a test set.
Computes metrics and generates a confusion matrix.

6. Error Analysis and Debugging
errorAnalysis: Identifies misclassified instances and analyzes errors.
debugTestSet: Compares training and test set distributions, detects outliers, and visualizes data using PCA.

7. Extra Functions
removeZeroVar: Removes features with zero variance.
removeOutliers: Removes outliers using Z-score.

Usage
--------------
Train the Model:
Run the script to preprocess data, train the DNN, and evaluate it using cross-validation.

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

