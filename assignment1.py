import random, linear_regression
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def training_data():
	"""
		Get the training data (features and target) from the files
	"""
	begin = True
	train_file = np.genfromtxt('dataset/train.csv',delimiter=',')
	
	features = []
	targets = []
	
	for row in train_file:
		if begin:
			begin = False
			continue
			
		targets.append(row[len(row)-1])    #The target number of shares, the last column
		
		#features.append([row[2], row[3], row[7], row[9], row[10], row[12], row[30], row[46], row[47]]) #Excluding some attributes
		features.append(row[2:len(row)-1])
		"""
				
		features.append(row[2:4])
		"""
	return features, targets			

def test_data():
	"""
		Get the test data (features and target) from the files
	"""
	begin = True
	test_file = np.genfromtxt('dataset/test.csv',delimiter=',')
	target_file = np.genfromtxt('dataset/test_target.csv',delimiter=',')
	features = []
	targets = []
	
	for row in test_file:
		if begin:
			begin = False
			continue
		features.append(row[2:len(row)]) #Excluding the non-predictive attributes
		#features.append(row[2:4]) #Excluding the non-predictive attributes
		#features.append([row[2], row[3], row[7], row[9], row[10], row[12], row[30], row[46], row[47]]) #Excluding some attributes
		
	begin = True		
	for row in target_file:
		if begin:
			begin = False
			continue
		targets.append(row) #Excluding the non-predictive attributes
		
	return features, targets

def z_norm(x, mean, std):
	"""
		Calculates the z-norm
	"""			
	x = (x - mean)/std

	return x
	
def normalize_targ(v):
	n_examples = len(v)

	#print col
	feat = []
	for example in v:
		feat.append(example)
	mean=(np.mean(feat))
	std=(np.std(feat))
	
	for i in range(n_examples):
		v[i] = z_norm(v[i], mean, std)
	
	return v
	
def normalize(dataset):	
	"""
		Returns the z-norm of the dataset
	"""
	n_feat = len(dataset[0])
	n_examples = len(dataset)
	mean = []
	std = []
	
	for col in range(n_feat):
		#print col
		feat = []
		for example in dataset:
			feat.append(example[col])
		mean.append(np.mean(feat))
		std.append(np.std(feat))
	
	for i in range(n_examples):
		for j in range(n_feat):
			dataset[i][j] = z_norm(dataset[i][j], mean[j], std[j])
	
	return dataset

def predict(model, x, y):
	
	outputs = []
	
	for i in range(len(x)):
		outputs.append(model[0])
		for j in range(len(x[0])):
			outputs[i] += model[1][j] * x[i][j]
	
	for i in range(len(outputs)):		
		print "Predict = ", outputs[i], " Real = ", y[i]
	
	plt.plot(outputs, 'bo', y, 'ro')
	
	
	'''
	plt.plot(x, outputs, 'bo')
	plt.plot(x, y, 'ro')

	'''
	plt.show()
	
	
#Get the data	
train_feat, train_targ = training_data()			
test_feat, test_targ = test_data()			



#Normalize the data
train_feat = normalize(train_feat)
test_feat = normalize(test_feat)
test_targ = normalize_targ(test_targ)
train_targ = normalize_targ(train_targ)

#print train_feat

#Find the weights!
lr = linear_regression.linear_regression()
model = lr.fit(train_feat, train_targ)
print "Modelo: ", model

predict(model, test_feat, test_targ)

"""
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(train_feat, train_targ)
#predict data
pred = regr.predict(test_feat)

# The coefficients
print('Thetas: ', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_targ, pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_targ, pred))

# Plot outputs

#print test_feat
#plt.scatter(train_feat, train_targ,  color='black')
#plt.plot(train_feat, pred, color='blue', linewidth=1)
#plt.plot(pred, 'bo', test_targ, 'ro')

#plt.show()
"""
