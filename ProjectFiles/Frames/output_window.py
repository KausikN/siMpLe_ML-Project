# Load libraries
import tkinter as tk
import json

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm_notebook 

# from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

import ann_network_improved as ann

np.random.seed(100)

# Essential Initis
fields = 'Epochs', 'LearningRate', 'Optim', 'LossFn'
filename = "training_input_json.json"
first_input = True


number_of_features = 0
number_of_outputs = 0
hidden_layer_config = []
data = []
labels = []

epochs = 1
learning_rate = 1
display_loss = True
opt = None
loss_fn = None

Y_map = []

# Optional Inits
first_input_optionalparams = True

optional_fields_underuse = 'WeightDecay'

optional_fields_Adagrad = 'LRDecay', 'WeightDecay'
optional_fields_Adam = 'Beta1', 'Beta2', 'WeightDecay'

def makeform(root, fields):
	entries = []
	for field in fields:
		row = tk.Frame(root)
		lab = tk.Label(row, width=15, text=field, anchor='w')
		ent = tk.Entry(row)
		row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
		lab.pack(side=tk.LEFT)
		ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
		entries.append((field, ent))
	return entries

#--2--
def fetch_inputs(entries):
	global first_input
	input_json = {}

	for entry in entries:
		field = entry[0]
		text  = entry[1].get()
		if field == "HiddenLayerConfig":
			text = text.split(",")
		input_json[field] = text
		#print('%s: "%s"' % (field, text))
	print(input_json)
	JSONHandlerFrame(data=input_json, filename=filename, option='a')
	first_input = False
	

#--3--
def JSONHandlerFrame(data={}, filename="training_input_json.json", option='r'):
	global first_input
	new_json = {}
	if option == 'c': # Clear
		f = open(filename, 'w+')
		f.write("")
		f.close()
		JSONHandler('a')
		first_input = True
		return 1

	elif option == 'r': # Read
		f = open(filename, 'r')
		json_str = f.read()
		f.close()
		if json_str == "":
			first_input = True
			return {}
		else:
			json_contents = json.loads(json_str)
			return json_contents

	elif option == 'a': # Append

		prev_json = JSONHandler('r')
		if first_input == True:
			for field in fields:
				new_json[field] = [data[field]]
		else :
			new_json = prev_json
			for field in fields:
				new_json[field].insert(0, data[field])
		print(new_json)

		json_str = json.dumps(new_json)
		f = open(filename, 'w+')
		f.write(json_str)
		f.close()
		return 2



def JSONHandler(data={}, filename="network_input_json.json", option='r'):
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
	global number_of_features
	global number_of_outputs
	global hidden_layer_config

	json_contents = JSONHandler('r')

	data = pd.read_csv(json_contents['InputDataLocation'][0])
	data = data.iloc[:,:int(json_contents['InputSize'][0])]

	labels = pd.read_csv(json_contents['OutputDataLocation'][0])
	labels = labels.iloc[:,labels.shape[1] - 1:]

	number_of_features = int(json_contents['InputSize'][0])

	for i in json_contents['HiddenLayerConfig'][0]:
		hidden_layer_config.append(int(i))
	print("HiddenLayerConfig: ", hidden_layer_config)

def FindUniqueElements(X):
	X_unique = []
	for i in X:
		if i not in X_unique:
			X_unique.append(i)
	return X_unique

def OneHotEncoder(X, update_values=False):
	Y = []
	global Y_map
	X_unique = FindUniqueElements(X)
	unique_size = len(X_unique)

	if update_values:
		global number_of_outputs
		number_of_outputs = unique_size
		Y_map = X_unique

	for i in X:
		X_temp = np.zeros(unique_size)
		X_temp[X_unique.index(i)] = 1
		Y.append(X_temp)
	return Y

def Inverse_OneHotEncoder(Y):
	X = []

	numpy_example = np.asarray([])
	tensor_example = torch.tensor(numpy_example)

	for j in Y:
		if type(j) == type(numpy_example):
			Y_temp = Y_map[list(j).index(np.max(j))]

		elif type(j) == type(tensor_example):
			Y_temp = Y_map[torch.argmax(j)]

		X.append(Y_temp)
	return X

def getParameterValues(fn, parameters={}):
	global epochs
	global learning_rate
	global opt
	global loss_fn

	global optional_fields_underuse
	global optional_fields_Adagrad
	global optional_fields_Adam

	# Default Values
	optimizer = "SGD"
	lossfn = "CE"
	# Default Values

	# Essential Values
	epochs = int(parameters['Epochs'][0])
	learning_rate = float(parameters['LearningRate'][0])
	optimizer = parameters['Optim'][0]
	lossfn = parameters['LossFn'][0]

	if optimizer == "SGD":
		opt = optim.SGD(fn.parameters(), lr=learning_rate)

	else:

		if optimizer == "Adagrad":
			optional_fields_underuse = optional_fields_Adagrad

			OptionalParametersWindow(optional_fields_Adagrad)
			optparams = JSONHandlerOptionalFrame('r')

			learning_rate_decay = float(optparams['LRDecay'][0])
			weight_decay = float(optparams['WeightDecay'][0])
			opt = optim.Adagrad(fn.parameters(), lr=learning_rate, lr_decay=learning_rate_decay, weight_decay=weight_decay)

		elif optimizer == "Adam":
			optional_fields_underuse = optional_fields_Adam

			OptionalParametersWindow(optional_fields_Adam)
			optparams = JSONHandlerOptionalFrame('r')

			betas = (float(optparams['Beta1'][0]), float(optparams['Beta2'][0]))
			weight_decay = float(optparams['WeightDecay'][0])
			opt = optim.Adam(fn.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)


	if lossfn == "CE":
		loss_fn = F.cross_entropy

def JSONHandlerOptionalFrame(data={}, filename="opt_params_input_json.json", option='r'):
	global first_input_optionalparams
	new_json = {}

	if option == 'r': # Read
		f = open(filename, 'r')
		json_str = f.read()
		f.close()
		if json_str == "":
			first_input_optionalparams = True
			return {}
		else:
			json_contents = json.loads(json_str)
			return json_contents

	elif option == 'a': # Append
		for field in optional_fields_underuse:
			new_json[field] = [data[field]]
		print(new_json)

		json_str = json.dumps(new_json)
		f = open(filename, 'w+')
		f.write(json_str)
		f.close()
		return 2

def fetch_inputs_optionalparams(entries):
	global first_input_optionalparams
	input_json = {}

	for entry in entries:
		field = entry[0]
		text  = entry[1].get()
		if field == "HiddenLayerConfig":
			text = text.split(",")
		input_json[field] = text
	print(input_json)
	JSONHandlerOptionalFrame(data=input_json, filename="opt_params_input_json.json", option='a')
	first_input_optionalparams = False

def OptionalParametersWindow(optional_fields={}):
	root = tk.Tk()
	ents = makeform(root, optional_fields)
	root.bind('<Return>', (lambda event, e=ents: fetch_inputs_optionalparams(e)))   
	b1 = tk.Button(root, text='Train',
		command=(lambda e=ents: fetch_inputs_optionalparams(e)))
	b1.pack(side=tk.LEFT, padx=5, pady=5)
	b2 = tk.Button(root, text='Quit', command=root.quit)
	b2.pack(side=tk.LEFT, padx=5, pady=5)
	root.mainloop()

def MainFunction():
	global number_of_features
	global number_of_outputs
	global hidden_layer_config
	global data
	global labels
	global epochs
	global learning_rate
	global opt
	global loss_fn
	global display_loss
	global Y_map

	#Load the data
	LoadData()

	print("\nlabels: ", labels, "\nlabelsize: ", labels.shape)
	print("\ndata: ", data, "\ndatasize: ", data.shape, "\n")

	#Split to test-train split
	X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
	print("\nX_train: ", X_train.shape, "\nX_val: ", X_val.shape, "\nY_train: ", Y_train.shape, "\nY_val: ", Y_val.shape, "\n")

	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	X_val = np.array(X_val)
	Y_val = np.array(Y_val)


	#One Hot Encoding
	Y_OH_val = OneHotEncoder(Y_val)
	Y_OH_train = OneHotEncoder(Y_train, update_values=True)
	if number_of_outputs <= 1:
		Y_OH_train = Y_train
		Y_OH_val = Y_val

	print("\nY_map: ", Y_map, "\n")
	print("\nNumber of possible output values: ", number_of_outputs, "\n")

	Y_OH_train_orig = Y_OH_train
	Y_OH_val_orig = Y_OH_val

	X_train =  torch.tensor(X_train, dtype=torch.float)
	Y_OH_train =  torch.tensor(Y_OH_train, dtype=torch.float)
	X_val =  torch.tensor(X_val, dtype=torch.float)
	Y_OH_val =  torch.tensor(Y_OH_val, dtype=torch.float)

	#Apply Network

	fn = ann.NeuralNetwork(number_of_features, number_of_outputs, hidden_layer_config)

	parameters = JSONHandlerFrame('r')
	
	getParameterValues(fn=fn, parameters=parameters)

	fn.fit(x=X_train, y=Y_OH_train, opt=opt, loss_fn=loss_fn, epochs=epochs, display_loss=display_loss)

	Y_pred_train = fn.predict(X_train)
	print(Y_pred_train)
	Y_pred_train = Inverse_OneHotEncoder(Y_pred_train)

	Y_pred_val = fn.predict(X_val)
	print(Y_pred_val)
	Y_pred_val = Inverse_OneHotEncoder(Y_pred_val)

	accuracy_train = accuracy_score(Y_pred_train, Y_train)
	accuracy_val = accuracy_score(Y_pred_val, Y_val)

	print("\nTraining accuracy", round(accuracy_train, 2))
	print("\nValidation accuracy", round(accuracy_val, 2), "\n")


#Main Code

root = tk.Tk()
ents = makeform(root, fields)
root.bind('<Return>', (lambda event, e=ents: fetch_inputs(e)))   
b1 = tk.Button(root, text='Train',
	command=(lambda e=ents: fetch_inputs(e)))
b1.pack(side=tk.LEFT, padx=5, pady=5)
b2 = tk.Button(root, text='Quit', command=root.quit)
b2.pack(side=tk.LEFT, padx=5, pady=5)
root.mainloop()

MainFunction()