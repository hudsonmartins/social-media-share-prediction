import random, linear_regression, normal_equation
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

feat_std=[]
feat_mean=[]
targ_std=0
targ_mean=0


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
		#features.append([row[2], row[3]])
		#features.append([row[2], row[3], row[7], row[9], row[10], row[12], row[30], row[46], row[47]]) #Excluding some attributes
		features.append(row[2:len(row)-1])

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
		#features.append([row[2], row[3]])
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

def get_stdnmean(dataset, n_feat):
	
	if n_feat > 1:
		mean = []
		std = []
		for col in range(n_feat):
			#print col
			feat = []
			for example in dataset:
				feat.append(example[col])
			mean.append(np.mean(feat))
			std.append(np.std(feat))
	else:
		feat = []
		for example in dataset:
			feat.append(example)
		mean = np.mean(feat)
		std = np.std(feat)
		
	return std, mean
	
def normalize(dataset, mean, std, n_feat):	
	"""
		Returns the z-norm of the dataset
	"""

	n_examples = len(dataset)
	if n_feat > 1:	
		for i in range(n_examples):
			for j in range(n_feat):
				dataset[i][j] = z_norm(dataset[i][j], mean[j], std[j])
	else:
		for i in range(n_examples):
			dataset[i] = z_norm(dataset[i], mean, std)

		
	return dataset

def predict(model, x, y):
	
	outputs = []
	
	for i in range(len(x)):
		outputs.append(model[0])
		for j in range(len(x[0])):
			outputs[i] += model[j+1] * x[i][j]
	
	for i in range(len(outputs)):		
		print "Predict = ", outputs[i], " Real = ", y[i]
	
	plt.plot(outputs, 'bo', y, 'ro')
	
	
	'''
	plt.plot(x, outputs, 'bo')
	plt.plot(x, y, 'ro')

	'''
	plt.show()
	

def remove_outliers(dataset, targ):
	n_examples = len(dataset)
	n_feat = len(dataset[0])
	new_dataset = []
	new_targ = []
	remove = False
		
	if n_feat > 1:
		
		
		for i in range(n_examples):
			remove = False
			for j in range(n_feat):
				#print "(",i,", ",j,")"
				if (dataset[i][j] < feat_mean[j] - 5*feat_std[j]) or (dataset[i][j] > feat_mean[j] + 5*feat_std[j]):
					if dataset[i][j] != 0.0 and dataset[i][j] != 1.0:
						#print "outlier in ", i
						remove = True
						break
				elif (targ[i] < targ_mean - 5*targ_std) or (targ[i] > targ_mean + 5*targ_std):
					remove = True
					break
			if not remove:
				#print "Added ", i
				new_dataset.append(dataset[i])
				new_targ.append(targ[i])
							
	return new_dataset, new_targ

#------------------- Getting the data and removing outliers ----------------------------------------
train_feat, train_targ = training_data()

feat_std, feat_mean = get_stdnmean(train_feat, len(train_feat[0]))
targ_std, targ_mean = get_stdnmean(train_targ, 1)
	
train_feat, train_targ = remove_outliers(train_feat, train_targ)

feat_std, feat_mean = get_stdnmean(train_feat, len(train_feat[0]))
targ_std, targ_mean = get_stdnmean(train_targ, 1)

test_feat, test_targ = test_data()			

#-------------------------- Normalizing the features -----------------------------------------------
train_feat = normalize(train_feat, feat_mean, feat_std, len(train_feat[0]))
test_feat = normalize(test_feat, feat_mean, feat_std, len(test_feat[0]))


#-------------------------- Finding the weights ---------------------------------------------------
"""
# Linear Regression
lr = linear_regression.linear_regression()
model = lr.fit(train_feat, train_targ)
print "Modelo: ", model
predict(model, train_feat, train_targ)

"""
#Normal Equations
ne = normal_equation.normal_equation()
model = ne.solve(train_feat, train_targ)
predict(model, test_feat, test_targ)

