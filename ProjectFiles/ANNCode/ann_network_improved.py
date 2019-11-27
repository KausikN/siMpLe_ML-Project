import torch.nn as nn            
class NeuralNetwork(nn.Module):
  
	def __init__(self, n_inputs, n_outputs, hidden_sizes=[2,3], activation_function="Sigmoid"):
		import torch
		import torch.nn as nn
		from collections import OrderedDict

		super().__init__()
		torch.manual_seed(0)

		act_func = nn.Sigmoid()

		if activation_function == "Sigmoid":
			act_func = nn.Sigmoid()

		seq = []
		seq.append(('input_layer', nn.Linear(n_inputs, hidden_sizes[0])))
		seq.append(('act_1', act_func))
		for i in range(len(hidden_sizes)):
			if i > 0:
				seq.append(('hidden_'+str(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i])))
				seq.append(('act_'+str(i), act_func))
		seq.append(('output_layer', nn.Linear(hidden_sizes[len(hidden_sizes)-1], n_outputs)))
		seq.append(('softmax', nn.Softmax()))

		print("\nNetwork Sequence: ", OrderedDict(seq), "\n")

		self.net = nn.Sequential(
			OrderedDict(seq)
		)

	def forward(self, X):
		return self.net(X)

	def loss_fn(self, y_hat, y):
		return -(y_hat[range(y.shape[0]), y].log()).mean()

	def fit(self, x, y, opt, loss_fn, epochs = 1000, display_loss=True):
		from torch import optim
		import matplotlib.pyplot as plt
		import matplotlib.colors

		#for i in y_OH:
		y = y.nonzero()[:, 1]

		loss_arr = []
		for epoch in range(epochs):
			loss = self.loss_fn(self.forward(x), y)
			loss_temp = loss.item()
			loss_arr.append(loss_temp)

			loss.backward()
			opt.step()
			opt.zero_grad()

		if display_loss:
			plt.plot(loss_arr)
			plt.xlabel('Epochs')
			plt.ylabel('CE')
			plt.show()

		return loss.item()

	def predict(self, X):
		import numpy as np

		Y_pred = self.net(X)

		Y_pred = Y_pred.detach().numpy()

		return np.array(Y_pred).squeeze()



