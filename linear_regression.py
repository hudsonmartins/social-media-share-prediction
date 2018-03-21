import random
import numpy as np

class linear_regression():

	def __init__ (self):
		self.alpha = 0.3 #Learning rate
		
	def fit(self, x_train, y_train):
		print "Training..."
		bias = random.randint(-200,200)	#Initialize the bias as a random value
		theta = []

		
		for i in range(len(x_train[0])):
			theta.append(random.randint(-200,200)) #Initialize theta as random values
			
		convergence = False
		prev_error = 0
		convergence = 0
		while(convergence < 200):	
			h = []
			for i in range(len(x_train)):
				h.append(bias)  # h = theta0
				for j in range(len(theta)):
					h[i] += theta[j]*x_train[i][j] # h = theta0 + sum(theta_i * x_i)
		
			error = self.calc_error(h, y_train) #Calculates the error	
			print "Error: ", error
			#print "Anterior: ", prev_error
			
			if round(prev_error,3) == round(error,3):
				convergence += 1
			else:
				convergence = 0				
			gd = self.gradient(h, x_train, y_train) #Calculates the gradient descent
			#print gd
			bias = bias - self.alpha*gd[0]
			for i in range(len(gd)-1):
				theta[i] = theta[i] - self.alpha*gd[i+1]
			prev_error = error

			
		#print "Pesos ", bias, theta			
		return [bias, theta]
		
	def calc_error(self, h, y):	
		j = 0 #The error J
		m = len(h) #The number of examples
		
		for i in range(m):
			j += (h[i] - y[i])**2
			
		j = j/(2*m)
		
		return j
		
	def gradient(self, h, x, y):
		gd = []
		m = len(h) #The number of examples

		#1/m * SUM of (hi -yi)
		bias_err = 0
		for i in range(m):
			bias_err += h[i]-y[i]

		gd.append(bias_err/m)
		
		for i in range(len(x[0])):
			err = 0
			for j in range(m):
				err += (h[j]-y[j])*x[j][i]
			gd.append(err/m)	
			
		return gd
