import numpy as np
import os.path

class normal_equation():
	def __init__ (self, reg):
		self.max_examples = 50000
		self.regularization = reg
		self.lamb = 1	
			
	def solve(self, x_train, y_train):
		print "Training..."
		
		X, y = self.reduce_examples(x_train, y_train)
		
		n = len(X[0]) #Number of features
		m = len(X) #Number of examples
			
	
		x_bias = np.ones((m, 1))
		X = np.append(x_bias, X, 1)
		X_transpose = np.transpose(X) 
		if self.regularization:
			#theta = inv(X^T * X + lamb * [reg matrix]) * X^T * y			
			reg_matrix = np.identity(X.shape[1])
			reg_matrix[0,0] = 0		
			theta = np.linalg.pinv(X_transpose.dot(X) + self.lamb*reg_matrix)					
		else:
			#theta = inv(X^T * X) * X^T * y
			theta = np.linalg.pinv(X_transpose.dot(X))		
					
		theta = theta.dot(X_transpose)
		theta = theta.dot(y)
		
				
		print theta
		
		file_name = raw_input("Insert the file name to save the normal equation data\n")
		self.save_results(theta, file_name)
		return theta
		
	def reduce_examples(self, X, y):
		"""
		Reduce the number of examples, so the algorithm doesn't takes too long to finish
		"""
		m = len(X) #number of examples
		new_X = []
		new_y = []
		if m > self.max_examples:
			for i in range(m):
				if(i < self.max_examples):
					new_X.append(X[i])
					new_y.append(y[i])
				else:
					break
			
			return new_X, np.transpose(new_y)
		else:
			return X, y

	def save_results(self, weights, file_name):
		if not os.path.isdir('results'):
			os.mkdir('results')
		
		limit = '-----------------------------------------------\n'			
		np.savetxt('results/'+file_name, weights, fmt = '%10.3f' ,header = limit, footer = limit)


