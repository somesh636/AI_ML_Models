# import required packages
import pickle 
import pandas as pd 
import os 
import sys 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from train_NLP import NLP

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

if __name__ == "__main__": 

	# Load your saved model
	nlp = pickle.load(open(os.path.join('models', '20817245_NLP_model.pkl'), 'rb'))  

	X_train = pd.read_csv('data/X_train_NLP.csv')
	X_test =  pd.read_csv('data/X_test_NLP.csv')
	y_train = pd.read_csv('data/y_train_NLP.csv')
	y_test =  pd.read_csv('data/y_test_NLP.csv')

	X_train_new = list(X_train.values.ravel())
	X_test_new = list(X_test.values.ravel())

	# Tokenize the 
	X_train_pad, X_test_pad = nlp.Tokenizer_Padding(X_train_new, X_test_new, y_train, y_test)

	y_test_new = y_test.values.flatten()

	# Accuracy Score for the Neural Network on Test Dataset  
	nlp.predict(X_test_pad, y_test_new)

