import random, os.path, plot_error
import numpy as np

class gradient_descent():

	def __init__ (self, reg):
		self.alpha = 0.001	 #Learning rate
		self.lamb = 10		#Regularization param
		self.regularization = reg  #True if there is regularization
		
	def fit(self, x_train, y_train):
		print "Training..."

		vec_error = []
		bias = random.randint(-100, 100)	#Initialize the bias as a random value
		theta = []


		for i in range(len(x_train[0])):
			theta.append(random.randint(-100, 100)) #Initialize theta as random values
			
		convergence = False
		prev_error = 0
		convergence = 0
		n_iterations = 0
		
		while(convergence < 200):	
			n_iterations += 1
			h = []
			m = len(x_train) #number of examples
			for i in range(m):
				h.append(bias)  # h = theta0
				for j in range(len(theta)):
					h[i] += theta[j]*x_train[i][j] # h = theta0 + sum(theta_i * x_i)
		
			error = self.calc_error(h, y_train) #Calculates the error	
			if n_iterations % 10 == 0:
				print "Epoch: ", n_iterations ," Error: ", error
				
			vec_error.append(error)
						
			if round(prev_error, 1) == round(error, 1):
				convergence += 1
			else:
				convergence = 0				
			gd = self.gradient(h, x_train, y_train) #Calculates the gradient descent
			
			bias = bias - self.alpha*gd[0]
			if self.regularization:
				for i in range(len(gd)-1):
					theta[i] = theta[i] - self.alpha*(gd[i+1] + (self.lamb/m) * theta[i])
			else:
				for i in range(len(gd)-1):
					theta[i] = theta[i] - self.alpha*gd[i+1]
			
			prev_error = error

			
		print "Pesos ", bias, theta			
		
		file_name = raw_input("Insert the file name to save the linear regression data\n")

		theta = np.append([bias], theta, axis=0)
		self.save_results(theta, vec_error, file_name, n_iterations)
		plot_error.plot(vec_error)
		
		return theta
		
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
	
	def save_results(self, weights, error, file_name, n_iterations):
		if not os.path.isdir('results'):
			os.mkdir('results')
			
		if os.path.isfile('results/'+file_name):
			myfile = open('results/'+file_name, 'a+')
			myfile.write('\n-----------------------------------------------\n')
			myfile.write('Learning Rate: '+ str(self.alpha))
			myfile.write('\nWeights: '+str(weights))
			myfile.write('\nNumber of iterations: '+str(n_iterations))
			myfile.write('\nError over time: '+str(error))
			

			myfile.close()
		else:
			myfile = open('results/'+file_name, 'w+')
			myfile.write('\n-----------------------------------------------\n')
			myfile.write('Learning Rate: ')
			myfile.write('Learning Rate: '+ str(self.alpha))
			myfile.write('\nWeights: '+str(weights))
			myfile.write('\nNumber of iterations: '+str(n_iterations))
			myfile.write('\nError over time: '+str(error))
			myfile.close()
			
