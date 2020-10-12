if __name__ == "__main__":
 
    import pandas as pd 
    import os 
    import pickle
    from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
    from acc_calc import accuracy
    from MLP_NeuralNet import MLPNeuralNetwork

    data_train = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/657_ECE/Assignments/Assignment_1/Q4_Folder/train_data.csv')
    data_train_labels = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/657_ECE/Assignments/Assignment_1/Q4_Folder/train_labels.csv')
    X_train = data_train.iloc[:17000, :]
    y_train = data_train_labels.iloc[:17000, :]
    X_valid = data_train.iloc[17000:, :]
    y_valid = data_train_labels.iloc[17000:, :]
    X_train_val = X_train.values
    y_train_val = y_train.values 
    X_valid_val = X_valid.values 
    y_valid_val = y_valid.values

    y_pred = test_mlp(X_train_val)
    test_accuracy = accuracy(y_train_val, y_pred)
    print("Accuracy Score for Validation Set: {0:.3f}".format(test_accuracy * 100))
    