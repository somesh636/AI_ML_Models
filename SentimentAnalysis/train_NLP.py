# import required packages
import pandas as pd 
import numpy as np 
import time 
from tqdm import tqdm 
import re as regex 
import pickle
import sys 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf 
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

class NLP(object):
	""" This class is mainly used for Sentiment Analysis for the IMDB Large 
		Dataset. 
	Parameters: 
	-----------
	base_folder: string(default: 'data/aclImdb')
	   As per the assignment it is the data folder but can be changed 
	   as required
		
	folder_labels: dict(default: 'pos': 1, 'neg': 0)
	   Folders inside aclImdb keep default
	
	batch_size: int(default: 100)
		Size of the Batch for Training and Validation 
	
	epochs: int(default: 20)
		Number of epochs to train the Neural Network.

	Attributes:
	-----------
	self: all the variables. 
	"""
	def __init__(self, base_folder = 'data/aclImdb', folder_labels = {'pos': 1, 'neg': 0}, batch_size = 100, epochs = 20):

		# Initialization for the Variables 
		self.base_folder = base_folder
		self.folder_labels = folder_labels
		self.data_imdbLarge = pd.DataFrame()
		self.batch_size = batch_size
		self.epochs = epochs

	def Load_data(self):
		""" Function for Loading the data from the aclImdb Folder
		Parameters: 
		-----------
		self: all the variable of the Class 

		Attributes: 
		----------
		self.data_imdbLarge: Returns Pandas DataFrame.
		"""
		sys.stderr.write("\n Loading Large IMDB Dataset.... \n")
		sys.stderr.flush()
		
		# Initialize the Progress Bar for the number of Samples 
		tqdmBar = tqdm(total = 50000)

		# Iterate over the Train Folder 

		for index_1 in ('test', 'train'):
			for index_2 in ('neg','pos'):
				# Get the path for the text data
				TextPath  = os.path.join(self.base_folder,index_1, index_2)
				# Iterate over the Text file and read the data from the file 
				for index_3 in sorted(os.listdir(TextPath)):
					Text_File = open(os.path.join(TextPath, index_3), 'r', encoding = 'utf-8') 
					data_text = Text_File.read() 
					self.data_imdbLarge = self.data_imdbLarge.append([[data_text, self.folder_labels[index_2]]], ignore_index = True)
					tqdmBar.update()

		# Name the coloumns of the Train Dataframe 
		self.data_imdbLarge.columns = ['Review', 'Sentiments']

		return self.data_imdbLarge

	def PreProcess_Data(self, data):
		""" PreProcessing the data in order to remove the html tags and emojis
		Parameter: 
		-----------
		data: Text data to be cleaned 
		
		Attributes:
		------------
		data: Cleaned text data
		"""

		data = regex.sub('<[^>]*>', '', data)
		exp = regex.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', data)
		data = (regex.sub('[\W]+', ' ', data.lower()) + ' '.join(exp).replace('-', ''))

		return data 

	def data_coversion(self):
		""" Saving Data for Testing the Model.

		Parameters: 
		------------
		self

		Attributes:
		------------ 
		X_train: Training Dataset with only Features 

		X_test: Testing Dataset with only Features 

		y_train: Training Target Labels 

		y_test: Testing Target Labels 
		
		"""

		# Load the Dataset from the Folder 
		data = self.Load_data()
		np.random.seed(0)
		IMDBmovie_data = data.reindex(np.random.permutation(data.index))
				
		# Clean the Train and Test Dataset 
		IMDBmovie_data['Review'] = IMDBmovie_data['Review'].apply(self.PreProcess_Data)

		# Feature and Target Extraction from the dataset
		data_X = IMDBmovie_data['Review']
		data_y = IMDBmovie_data['Sentiments']

		# Split the dataset in Training and Testing set 
		X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.50, random_state = 42)

		self.X_train = list(X_train.values.ravel())
		self.X_test = list(X_test.values.ravel())
		self.y_train = y_train.values
		self.y_test = y_test.values 

		X_train_df = pd.DataFrame(data = self.X_train)
		X_test_df = pd.DataFrame(data = self.X_test)
		y_train_df = pd.DataFrame(data = self.y_train)
		y_test_df = pd.DataFrame(data= self.y_test)

		X_train_df.to_csv('data/X_train_NLP.csv', index = False)
		X_test_df.to_csv('data/X_test_NLP.csv', index = False)
		y_train_df.to_csv('data/y_train_NLP.csv', index = False)
		y_test_df.to_csv('data/y_test_NLP.csv', index = False)

		return self.X_train, self.X_test, self.y_train, self.y_test 


	def Tokenizer_Padding(self, X_train, X_test, y_train, y_test): 
		""" Tokenization and Padding of the Train and Test dataset. 
		Parameters: 
		------------
		X_train: Training Dataset with only Features 

		X_test: Testing Dataset with only Features 

		y_train: Training Target Labels 

		y_test: Testing Target Labels  

		Attributes:
		------------ 
		X_train_pad: Padded Training Data 

		X_test_pad: Padded Testing Data 

		"""

		# Tokenization of the Text Data 
		tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 10000) 
		tokenizer.fit_on_texts(X_train)
		token_train = tokenizer.texts_to_sequences(X_train)
		token_test = tokenizer.texts_to_sequences(X_test)

		self.max_words = 400 
		self.X_train_pad = sequence.pad_sequences(token_train, maxlen = self.max_words)
		self.X_test_pad = sequence.pad_sequences(token_test, maxlen = self.max_words)

		return self.X_train_pad, self.X_test_pad  

	def LSTM_init(self): 

		""" Initialization Function for the NLP Network 
		Attributes: 
		-----------
		self 
		"""
		vocabulary_size = 10000  
		embedding_size = 16	

		# Initialize the Model 
		self.LSTM_model = Sequential()
		self.LSTM_model.add(Embedding(vocabulary_size, embedding_size )) 
		self.LSTM_model.add(LSTM(100))
		self.LSTM_model.add(Dense(1, activation='sigmoid'))

		return self 

	def LSTM_train(self): 
		""" Trainig Function for the NLP Network 

		Attributes: 
		-----------
		history_LSTM: Attributes of the Model  
		"""

		X_train, X_test, y_train, y_test = self.data_coversion()
		self.Tokenizer_Padding(X_train, X_test, y_train, y_test)
		self.LSTM_init()
		self.LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		history_LSTM = self.LSTM_model.fit(self.X_train_pad, self.y_train, validation_data=(self.X_test_pad, self.y_test), batch_size = self.batch_size, epochs = self.epochs)

		return history_LSTM 

	def predict(self, X_test_pad, y_test):
		""" Prediction Function for the Class 
		
		X_test_pad: Padded data 

		y_test: Target data
		Attributes:
		------------
		self 

		""" 
		scores = self.LSTM_model.evaluate(X_test_pad, y_test)
		acc_value = scores[1]
		sys.stderr.write(" \n Test Accuracy: {0:.3f} \n".format(acc_value * 100))
		sys.stderr.flush()

		return self 

if __name__ == "__main__": 

	# Initilaze the Class 
	nlp = NLP(batch_size = 64, epochs = 5)	

	# Train the Model 
	history = nlp.LSTM_train()


	# ########################################################### Code For Pickling this File ############################    
	# dir_path = os.path.dirname(os.path.realpath(__file__))
	# dest = os.path.join(dir_path,'models')
	# if not os.path.exists(dest):
	# 	os.makedirs(dest)

	# pickle.dump(nlp, open(os.path.join(dest, '20817245_NLP_model.pkl'), 'wb'), protocol=4)                                                                                                                   
	# ###################################################################################################################
