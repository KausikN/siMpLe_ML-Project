# Load libraries
import tkinter as tk
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm_notebook 

# from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

import ann_network as ann

np.random.seed(100)

fields = 'Epochs', 'LearningRate'
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

Y_map = []



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
	for j in Y:
		Y_temp = Y_map[list(j).index(np.max(j))]
		X.append(Y_temp)
	return X



def MainFunction():
	#Load the data
	LoadData()

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

	print("Y_map: ", Y_map)
	print("Number of possible output values: ", number_of_outputs)
	# print("\nY_OH_train: ", len(Y_OH_train), "\nY_OH_val: ", len(Y_OH_val), "\n")

	#Apply Network
	ffsn_multi = ann.FFSN_MultiClass(number_of_features, number_of_outputs, hidden_layer_config)

	#Change as required
	epochs = 1000
	learning_rate = 0.5
	display_loss = True
	parameters = JSONHandlerFrame('r')
	epochs = int(parameters['Epochs'][0])
	learning_rate = float(parameters['LearningRate'][0])
	#Change as required

	ffsn_multi.fit(X_train, Y_OH_train, epochs=epochs, learning_rate=learning_rate, display_loss=display_loss)

	Y_pred_train = ffsn_multi.predict(X_train)
	Y_pred_train = Inverse_OneHotEncoder(Y_pred_train)

	Y_pred_val = ffsn_multi.predict(X_val)
	Y_pred_val = Inverse_OneHotEncoder(Y_pred_val)

	# print("Y_OH_train: ", Y_OH_train)
	# print("Y_pred_train: ", Y_pred_train)
	# print("Y_OH_train: ", Y_OH_train)
	# print("Y_OH_val: ", Y_OH_val)

	accuracy_train = accuracy_score(Y_pred_train, Y_train)
	accuracy_val = accuracy_score(Y_pred_val, Y_val)

	print("Training accuracy", round(accuracy_train, 2))
	print("Validation accuracy", round(accuracy_val, 2))


#Main Code

# if __name__ == '__main__':
# 	root = tk.Tk()
# 	ents = makeform(root, fields)
# 	root.bind('<Return>', (lambda event, e=ents: fetch_inputs(e)))   
# 	b1 = tk.Button(root, text='Show',
# 		command=(lambda e=ents: fetch_inputs(e)))
# 	b1.pack(side=tk.LEFT, padx=5, pady=5)
# 	b2 = tk.Button(root, text='Quit', command=root.quit)
# 	b2.pack(side=tk.LEFT, padx=5, pady=5)
# 	root.mainloop()

root = tk.Tk()
ents = makeform(root, fields)
root.bind('<Return>', (lambda event, e=ents: fetch_inputs(e)))   
b1 = tk.Button(root, text='Show',
	command=(lambda e=ents: fetch_inputs(e)))
b1.pack(side=tk.LEFT, padx=5, pady=5)
b2 = tk.Button(root, text='Quit', command=root.quit)
b2.pack(side=tk.LEFT, padx=5, pady=5)
root.mainloop()

MainFunction()