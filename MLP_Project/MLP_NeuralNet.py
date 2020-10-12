import numpy as np 
import sys

class MLPNeuralNetwork(object):
    """ Multilayer Perceptron Neural Network with 1 input layer, 1 hidden layer and 
        Output layer 
    
    Input Parameters:
    ------------------ 
    
    n_hidden_unit: Number of Neurons(Unit) in the hidden layer default
                   dtype: int, default value = 60

    epochs: Number of iterations that the Network iterates over the training dataset
            dtype: int, default value = 200

    eta: Learning rate for the Network
         dtype: float, default value = 0.001

    shuffle: This condition enables the dataset to be shuffled every epoch
             dtype: bool, defalut value = True

    batch_size: The size of each batch used while training
                dtype: int, default value = 1

    RandomState: Immutable value. Mainly used for initializing the weights and biases if value is 1
                 the value of the randomly generated weights becomes fixed.
                 If value is 0 then each time it randomly generated the weights
                 dtype: int, default value = 1

    Return:
    -------
    self: all the parameters accessed by class methods
          
    """

    def __init__(self, n_hidden_unit = 100, epochs=200, eta=0.001, shuffle=False, batch_size=100):
                 """ Initialization for the NeuralNetwork """

                 self.random = np.random.RandomState(1)
                 self.n_hidden_unit = n_hidden_unit
                 self.epochs = epochs 
                 self.eta = eta 
                 self.batch_size = batch_size 
                 self.shuffle = shuffle
                 output_unit = y_train.shape[1]
                 feature_set = X_train.shape[1]
                 self.b_hidden = np.zeros(self.n_hidden_unit)
                 self.w_hidden = self.random.normal(loc=0.0, scale=0.1, size=(feature_set, self.n_hidden_unit))
                 self.b_output = np.zeros(output_unit)
                 self.w_output = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden_unit, output_unit))
                 self.cost_tot = []
                 self.train_accuracy = []


    def sigmoid(self, x):
        """ Activation function for the Hidden layer
        Input Parameters: 
        -----------------
        x: Input for the activation function

        Returns: 
        --------
        value after calculating the activation using the sigmoid formula
        """
        return (1.0 / (1.0 + np.exp(-x)))

    def sigmoid_der(self, x):
        """ Derivative of Activation function used while BackPropagation
        Input Parameter: 
        ----------------
        x: Output from the hidden layer

        Return: 
        -------
        der_value: Derivative value for the sigmoid function
         """
        der_value = x * (1.0 - x)
        return der_value

    def softmax(self, x):
        """ Activation function for the output layer
        Input Parameters: 
        -----------------
        x: Input for the activation function

        Returns: 
        --------
        a_output: value after calculating the activation using the softmax
        formula """

        z_op_atcv = np.exp(x)
        a_output = z_op_atcv/z_op_atcv.sum(axis=1, keepdims=True)
        return a_output

    def decode_onehot_encoding(self, x):
        """ Used for decoding onehot encoding for the training label
        Input Parameter:
        -----------------
        x: Onehot Encoded y_label values along axis = 1/ columns.

        Return: 
        --------
        y_label_int: Decoded Onehot encoding values 
        """
        y_label_int = np.argmax(x, axis=1)
        return y_label_int

    def forward_calculation(self, x):
        """Calculation of weights and other value when propagating 
            forward
        Input Parameter
        ---------------
        x: dataset for calculation

        Returns: 
        ---------
        hl_value: Value that goes to the input of hidden layer
        
        hl_output: Value after applying Sigmoid activation 
                   function on the hidden layer
        
        ol_value: Value that goes to the input of the Output layer

        ol_output: Value after applying softmax activation fucntion
                    on the output layer
        """

        # hidden layer calculation
        hl_value = np.dot(x, self.w_hidden) + self.b_hidden

        # Applying Sigmoid Activation Function
        hl_output = self.sigmoid(hl_value)

        # Output layer calculation
        ol_value = np.dot(hl_output, self.w_output) +self.b_output

        # Applying Softmax Activation Function
        ol_output = self.softmax(ol_value)

        return hl_value, hl_output, ol_value, ol_output

    def cost_calculation(self, y_label, ol_output):
        """ Calculate the Cost for the Network 
        
        Input Parameters: 
        -----------------
        y_label: The original target label from the dataset

        ol_output: Value from the output layer 

        Returns:
        --------- 
        cost: Total cost incured by the network

        """

        # using Cross Entropy function for calculation of cost 
        cost = np.sum(-y_label * np.log(ol_output))

        return cost

    def predict(self, x):
        """ Predict the values for the given input dataset

        Input Parameters:
        -----------------
        x: The dataset for which prediction is to be made

        Returns:
        --------
        y_pred: predicted y labels for the dataset by the network

        """
        # Calculate the values for prediction using Forward Function 
        hl_value, hl_ouput, ol_value, ol_output = self.forward_calculation(x)

        # convert back to integer mode along column wise
        y_pred = self.decode_onehot_encoding(ol_value)

        return y_pred

    def accuracy_score(self, y_label, y_pred):
        """ Calculate the Accuracy score after Prediction
        Input Parameters: 
        -----------------
        y_label: Original target labels 

        y_pred: Predicted values by the Network
        
        Returns: 
        --------
        acc_score: Accuracy score in percent
        """

        # Decode the One-hot Encoded Values in the y_label
        y_label = self.decode_onehot_encoding(y_label)

        # Compare and sum the number of same values in prediction and label 
        acc_score = ((np.sum(y_label == y_pred)).astype(np.float)/y_label.shape[0])

        return (acc_score * 100)

    def onehot_encoding(self, y_test):
        """ For Encoding y_labels to OneHot 
        
        """
        onehot = np.zeros((4, y_test.shape[0]))
        for idx, val in enumerate(y_test.astype(int)):
            onehot[val, idx] = 1.

        return onehot.T
    
    def train(self, X_train, y_train):
        """ Train Network for weights using the input datasets
        Input Parameter:
        ----------------
        X_train: Training dataset with only features
        y_train: Training dataset with only labels

        Returns:
        --------
        self: all the parameters the can be accessed
        """
        

        for index_i in range(self.epochs):

            index_val = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(index_val)

            for index_j in range(0, index_val.shape[0]- self.batch_size + 1, self.batch_size):
                batch = index_val[index_j:index_j + self.batch_size]

                # Step Forward Propagation
                hl_value, hl_output, ol_value, ol_output = self.forward_calculation(X_train[batch])

                # Backward Propagation
                del_output = ol_output - y_train[batch]
                sigmoid_der_hidden = self.sigmoid_der(hl_output)
                del_hidden = np.dot(del_output, self.w_output.T) * sigmoid_der_hidden
                gradient_w_hidden = np.dot(X_train[batch].T, del_hidden)
                gradient_b_hidden = np.sum(del_hidden, axis=0)
                gradient_w_output = np.dot(hl_output.T, del_output)
                gradient_b_output = np.sum(del_output, axis=0)

                # Weight and Bias Update for hidden layer 
                self.w_hidden -= self.eta * (gradient_w_hidden + self.w_hidden)
                self.b_hidden -= self.eta * gradient_b_hidden 

                # Weight and Bias Update for the Output layer
                self.w_output -= self.eta * (gradient_w_output + self.w_output)
                self.b_output -= self.eta * gradient_b_output

            # Accuracy evaluation for the training dataset after each epoch
            hl_value, hl_output, ol_value, ol_output = self.forward_calculation(X_train)
            cost = self.cost_calculation(y_label = y_train, ol_output = ol_output)
            y_train_pred = self.predict(X_train)
            training_accuracy = self.accuracy_score(y_label=y_train, y_pred=y_train_pred)
            self.cost_tot.append(cost)
            self.train_accuracy.append(training_accuracy)
            sys.stderr.write("Epoch: {0}/{1} | Cost: {2:.3f} | Training Accuracy: {3:.3f} \n".format(index_i+1 , self.epochs, cost, training_accuracy ))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            sys.stderr.flush()

        return self

if __name__ == "__main__": 
    
    import pandas as pd 
    import os 
    import pickle
    from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
    from acc_calc import accuracy
    
    data_train = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/657_ECE/Assignments/Assignment_1/Q4_Folder/train_data.csv')
    data_train_labels = pd.read_csv('~/Documents/Uwaterloo_Study_Docs/657_ECE/Assignments/Assignment_1/Q4_Folder/train_labels.csv')
    print(data_train.shape)
    # X_train = data_train.iloc[:17000, :]
    # y_train = data_train_labels.iloc[:17000, :]
    # X_valid = data_train.iloc[17000:, :]
    # y_valid = data_train_labels.iloc[17000:, :]
    # X_train_val = X_train.values
    # y_train_val = y_train.values 
    # X_valid_val = X_valid.values 
    # y_valid_val = y_valid.values
    # mlp_neuralnet = MLPNeuralNetwork(n_hidden_unit=100, epochs=200, eta=0.0004, shuffle=True, batch_size=1000)
    # mlp_neuralnet.train(X_train = X_train_val, y_train = y_train_val)

    # # Test for the Validation Accuracy
    # y_valid_pred = mlp_neuralnet.predict(X_valid_val)
    # accuracy_score = mlp_neuralnet.accuracy_score(y_label = y_valid_val, y_pred=y_valid_pred)
    # print("Accuracy Score for Validation Set: {0:.3f}".format(accuracy_score))
    
############################################################# Code For Pickling this File ############################    
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # dest = os.path.join(dir_path,'MLP_Model')
    # if not os.path.exists(dest):
    #     os.makedirs(dest)

    # pickle.dump(mlp_neuralnet, open(os.path.join(dest, 'mlpmodel.pkl'), 'wb'), protocol=4)
                                                                                                                
    #mlp_neuralnet = pickle.load(open(os.path.join('MLP_Model', 'mlpmodel.pkl'), 'rb'))                         
######################################################################################################################





