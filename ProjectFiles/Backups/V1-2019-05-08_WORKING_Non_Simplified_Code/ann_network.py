
class FFSN_MultiClass:
	
	def __init__(self, n_inputs, n_outputs, hidden_sizes=[2,3]):

		import numpy as np
		#Initialisations
		self.nx = n_inputs
		self.ny = n_outputs
		self.nh = len(hidden_sizes)
		self.sizes = [self.nx] + hidden_sizes + [self.ny] 

		self.W = {}
		self.B = {}
		for i in range(self.nh+1):
			self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
			self.B[i+1] = np.zeros((1, self.sizes[i+1]))

	def sigmoid(self, x):
		import numpy as np
		return 1.0/(1.0 + np.exp(-x))

	def softmax(self, x):
		import numpy as np
		exps = np.exp(x)
		return exps / np.sum(exps)

	def forward_pass(self, x):
		import numpy as np

		self.A = {}
		self.H = {}
		self.H[0] = x.reshape(1, -1)
		for i in range(self.nh):
			self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
			self.H[i+1] = self.sigmoid(self.A[i+1])
		self.A[self.nh+1] = np.matmul(self.H[self.nh], self.W[self.nh+1]) + self.B[self.nh+1]
		self.H[self.nh+1] = self.softmax(self.A[self.nh+1])
		return self.H[self.nh+1]

	def predict(self, X):
		import numpy as np

		Y_pred = []
		for x in X:
			y_pred = self.forward_pass(x)
			Y_pred.append(y_pred)
		return np.array(Y_pred).squeeze()

	def grad_sigmoid(self, x):
		return x*(1-x) 

	def cross_entropy(self,label,pred):
		import numpy as np

		yl=np.multiply(pred,label)
		yl=yl[yl!=0]
		yl=-np.log(yl)
		yl=np.mean(yl)
		return yl

	def grad(self, x, y):
		import numpy as np

		self.forward_pass(x)
		self.dW = {}
		self.dB = {}
		self.dH = {}
		self.dA = {}
		L = self.nh + 1
		self.dA[L] = (self.H[L] - y)
		for k in range(L, 0, -1):
			self.dW[k] = np.matmul(self.H[k-1].T, self.dA[k])
			self.dB[k] = self.dA[k]
			self.dH[k-1] = np.matmul(self.dA[k], self.W[k].T)
			self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1])) 

	def fit(self, X, Y, epochs=100, initialize='True', learning_rate=0.01, display_loss=False):
		import numpy as np
		from tqdm import tqdm_notebook
		import matplotlib.pyplot as plt
		import matplotlib.colors

		if display_loss:
			loss = {}

		if initialize:
			for i in range(self.nh+1):
				self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
				self.B[i+1] = np.zeros((1, self.sizes[i+1]))

		for epoch in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
			dW = {}
			dB = {}
			for i in range(self.nh+1):
				dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
				dB[i+1] = np.zeros((1, self.sizes[i+1]))
			for x, y in zip(X, Y):
				self.grad(x, y)
				for i in range(self.nh+1):
					dW[i+1] += self.dW[i+1]
					dB[i+1] += self.dB[i+1]

			m = X.shape[1]
			for i in range(self.nh+1):
				self.W[i+1] -= learning_rate * (dW[i+1]/m)
				self.B[i+1] -= learning_rate * (dB[i+1]/m)

			if display_loss:
				Y_pred = self.predict(X) 
				loss[epoch] = self.cross_entropy(Y, Y_pred)

		if display_loss:
			plt.plot(loss.values())
			plt.xlabel('Epochs')
			plt.ylabel('CE')
			plt.show()

