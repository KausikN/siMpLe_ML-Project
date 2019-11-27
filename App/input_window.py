#--Imports------------------------------------------------------------------
import tkinter as tk
import json
#---------------------------------------------------------------------------

#--Initialisations----------------------------------------------------------
first_input = True

filename = "network_input_json.json"

fields = 'InputSize', 'HiddenLayerConfig', 'InputDataLocation', 'OutputDataLocation'


input_layer_size = 0
output_layer_size = 0
hidden_layers = []
#---------------------------------------------------------------------------

#--Functions----------------------------------------------------------------
#--1--
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
	print(input_json)
	JSONHandler(data=input_json, filename=filename, option='a')
	first_input = False


#--3--
def JSONHandler(data={}, filename="network_input_json.json", option='r'):
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

#---------------------------------------------------------------------------

#--Main Code----------------------------------------------------------------

root = tk.Tk()
ents = makeform(root, fields)
root.bind('<Return>', (lambda event, e=ents: fetch_inputs(e)))   
b1 = tk.Button(root, text='Done',
	command=(lambda e=ents: fetch_inputs(e)))
b1.pack(side=tk.LEFT, padx=5, pady=5)
b2 = tk.Button(root, text='Quit', command=root.quit)
b2.pack(side=tk.LEFT, padx=5, pady=5)
root.mainloop()