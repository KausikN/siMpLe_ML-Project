# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm_notebook 

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

import ann_network as ann

np.random.seed(100)


number_of_features = 0
data = []
labels = []


def JSONHandler(data={}, filename="input_json.json", option='r'):
	if option == 'c': # Clear
		f = open(filename, 'w+')
		f.write("")
		f.close()
		JSONHandler('a')
		return 1

	elif option == 'r': # Read
		f = open(filename, 'r')
		json_str = f.read()
		f.close()
		if json_str == None:
			return None
		json_contents = json.loads(json_str)
		return json_contents


def LoadData():
	global data
	global labels
	json_contents = JSONHandler('r')

	data = pd.read_csv(json_contents['InputDataLocation'][0]).iloc[:,:json_contents['InputSize'][0]]
	number_of_features = data.shape[1]
	labels = pd.read_csv(json_contents['OutputDataLocation'][0])
	labels = labels.iloc[:,labels.shape[1] - json_contents['OutputSize'][0]:]






# Set the number of features we want


# Load feature and target data
#(train_data, train_target_vector), (test_data, test_target_vector) = reuters.load_data(num_words=number_of_features)
train_data, test_data, train_target_vector, test_target_vector = train_test_split(data, labels, stratify=labels, random_state=0)

# Convert feature data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode='binary')
test_features = tokenizer.sequences_to_matrix(test_data, mode='binary')

# One-hot encode target vector to create a target matrix
train_target = to_categorical(train_target_vector)
test_target = to_categorical(test_target_vector)