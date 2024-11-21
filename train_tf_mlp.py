import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import random
import os

# Set random seeds for reproducibility
def set_seeds(seed=42):
    """
    Sets random seeds for reproducibility across Numpy, Python's random module, and TensorFlow.
    Also enables deterministic operations in TensorFlow.

    Args:
        seed (int): Seed value for reproducibility. Default is 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

# MLP model creation function
def create_mlp(input_dim, hidden_neurons):
    """
    Creates a Multilayer Perceptron (MLP) model with three hidden layers.

    Args:
        input_dim (int): Number of input features.
        hidden_neurons (int): Number of neurons in each hidden layer.

    Returns:
        tensorflow.keras.Sequential: Compiled MLP model.
    """
    model = Sequential()
    # Add three hidden layers with ReLU activation
    model.add(Dense(hidden_neurons, activation='relu', input_dim=input_dim))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dense(hidden_neurons, activation='relu'))
    # Add the output layer with linear activation (for regression tasks)
    model.add(Dense(1))
    # Compile the model using Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Function to count the number of trainable parameters in a model
def count_trainable_params(model):
    """
    Counts the total number of trainable parameters in the given model.

    Args:
        model (tensorflow.keras.Model): The Keras model.

    Returns:
        int: Total number of trainable parameters.
    """
    return np.sum([np.prod(v.shape) for v in model.trainable_weights])

# Training and testing loop
def train_mlp_with_neurons(X_train, y_train, X_test, y_test, neurons, onnx_model_path, batch_size):
    """
    Trains an MLP model with the specified number of neurons and saves the trained model.

    Args:
        X_train (array-like): Training input features.
        y_train (array-like): Training target values.
        X_test (array-like): Testing input features.
        y_test (array-like): Testing target values.
        neurons (int): Number of neurons in each hidden layer.
        onnx_model_path (str): Path to save the trained model.
        batch_size (int): Batch size for training.
    """
    input_dim = X_train.shape[1]  # Number of features in the input data

    print(f"\nTraining with {neurons} neurons in the hidden layer...")

    # Create the MLP model
    model = create_mlp(input_dim, neurons)

    # Count and display the number of trainable parameters
    trainable_params = count_trainable_params(model)
    print(f"Trainable parameters: {trainable_params}")

    # Train the model for 100 epochs
    history = model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=batch_size, 
                        validation_data=(X_test, y_test))

    # Display final training and validation RMSE
    print('Final Training RMSE: ', history.history['root_mean_squared_error'][-1])
    print('Final Validation RMSE: ', history.history['val_root_mean_squared_error'][-1])

    # Save the model to the specified path
    print(f"Saving the model to {onnx_model_path}.")
    model.save(onnx_model_path)  # Save the current model

# Function to load data from CSV files
def load_data_from_csv(train_csv, test_csv):
    """
    Loads training and testing data from CSV files.

    Assumes the last column in the CSV files is the target variable (y), and all other columns are features (X).

    Args:
        train_csv (str): Path to the training data CSV file.
        test_csv (str): Path to the testing data CSV file.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) - Input features and target values for training and testing.
    """
    # Load the CSV data into Pandas DataFrames
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    # Split data into features (X) and target (y)
    X_train = train_data.iloc[:, :-1].values  # Features (all columns except the last)
    y_train = train_data.iloc[:, -1].values   # Target (last column)
    
    X_test = test_data.iloc[:, :-1].values  # Features (all columns except the last)
    y_test = test_data.iloc[:, -1].values   # Target (last column)

    return X_train, y_train, X_test, y_test

# Example usage of the script
if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seeds(42)  # You can use any seed value for reproducibility

    # File paths for training and testing datasets
    train_csv = 'peaks_train.csv'  # Path to the training dataset
    test_csv = 'peaks_test.csv'  # Path to the testing dataset

    # Load the data from CSV files
    X_train, y_train, X_test, y_test = load_data_from_csv(train_csv, test_csv)

    # Path to save the trained ONNX model
    model_path = "peaks_mlp_relu_3_64.keras"

    # Number of neurons in each hidden layer
    neurons = 64

    # Batch size for training
    batch_size = 80

    # Train the MLP model
    train_mlp_with_neurons(X_train, y_train, X_test, y_test, neurons, model_path, batch_size)