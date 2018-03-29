import random, math, gradient_descent, normal_equation
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
	n_rows = 0
	
	for row in train_file:
		n_rows += 1
		if n_rows > 6000:
			break
			
		if begin:
			begin = False
			continue
			
		targets.append(row[len(row)-1])    #The target number of shares, the last column		
		treated_row = []
		
		for i in range(2,len(row)-1): #removing non-predictive attributes
			#excluding discrete data
			if (i > 12 and i < 19) or (i > 30 and i < 39):
				continue
			treated_row.append(row[i])
			
		features.append(treated_row)

	return features, targets, len(features[0])

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
		treated_row = []
		
		for i in range(2,len(row)): #removing non-predictive attributes
			#excluding discrete data
			if (i > 12 and i < 19) or (i > 30 and i < 39):
				continue
			treated_row.append(row[i])	
		features.append(treated_row)
				
	begin = True		
	for row in target_file:
		if begin:
			begin = False
			continue
		targets.append(row)
		
	return features, targets

def increase_features(features):
	"""
		Increase the number of features to get a better model
	"""
	
	new_features = []
	for row in features:
		increased = []
		for i in range(len(row)):
			increased.append(row[i])
		
			for j in range(2, 21):
				increased.append(row[i]**j) #880 feat
			
			for k in range (2, len(row)-1):
				if(i > k) and (i-k-1 >= 0):
					increased.append(row[i]*row[i-k-1])	#1741 feat
			
					for j in range(2,4):  #2: 4324, 3: 6907
						increased.append((row[i]**j)*row[i-k-1])
						increased.append(row[i]*row[i-k-1]**j)
						increased.append((row[i]**j)*row[i-k-1]**j)
				
				else:
					break


		new_features.append(increased)

	return new_features



def z_norm(x, mean, std):
	"""
		Calculates the z-norm
	"""			
	x = (x - mean)/std

	return x

def get_stdnmean(dataset, n_feat):
	"""
		Calculates the standard deviation and mean for a given dataset
	"""
	
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
	"""
		Finds the output for the model and compare to the ground truth
	"""
	
	outputs = []
	
	
	for i in range(len(x)):
		outputs.append(model[0])
		for j in range(len(x[0])):
			outputs[i] += model[j+1] * x[i][j]
	
	diff = []
	for i in range(len(outputs)):		
		#print "Predict = ", outputs[i], " Real = ", y[i]
		#print "percentage error = ", abs(outputs[i]-y[i])/y[i]
		diff.append(abs(outputs[i]-y[i])/y[i])

	n_correct = 0
	for value in diff:
		if value < 0.1:
			n_correct += 1
	
	print "Number of correct predictions = ", n_correct
	print "Data size = ", len(y)
	#plt.plot(outputs, 'bo', y, 'ro')
	
	plt.show()
	

def remove_outliers(dataset, targ, n_feat):
	"""
		Remove the rows that contains outliers in both features and target
	"""
	
	n_examples = len(dataset)
	new_dataset = []
	new_targ = []
	remove = False
	
	if n_feat > 1:
		for i in range(n_examples):
			remove = False
			for j in range(n_feat):
				#print "(",i,", ",j,")"
				if (dataset[i][j] < feat_mean[j] - 4*feat_std[j]) or (dataset[i][j] > feat_mean[j] + 4*feat_std[j]):
					if dataset[i][j] != 0.0 and dataset[i][j] != 1.0:
						#print "outlier in ", i
						remove = True
						break
				elif (targ[i] < targ_mean - 4*targ_std) or (targ[i] > targ_mean + 4*targ_std):
					remove = True
					break
			if not remove:
				#print "Added ", i
				new_dataset.append(dataset[i])
				new_targ.append(targ[i])
							
	return new_dataset, new_targ

#------------------- Getting the data and removing outliers ----------------------------------------

train_feat, train_targ, n_feat_init = training_data()
test_feat, test_targ = test_data()

print "Preparing data..."
feat_std, feat_mean = get_stdnmean(train_feat, len(train_feat[0]))
targ_std, targ_mean = get_stdnmean(train_targ, 1)

print "Removing outliers"
train_feat, train_targ = remove_outliers(train_feat, train_targ, n_feat_init)

print "Number of examples: ", len(train_feat)

print "Increasing features"
train_feat = increase_features(train_feat)
test_feat = increase_features(test_feat)

print "Number of features: ", len(train_feat[0])

#-------------------------- Normalizing the features -----------------------------------------------
print "Normalizing..."
feat_std, feat_mean = get_stdnmean(train_feat, len(train_feat[0]))
train_feat = normalize(train_feat, feat_mean, feat_std, len(train_feat[0]))
test_feat = normalize(test_feat, feat_mean, feat_std, len(test_feat[0]))


#-------------------------- Finding the weights ---------------------------------------------------

# Gradient Descent
gd = gradient_descent.gradient_descent()
model = gd.fit(train_feat, train_targ)
print "Modelo: ", model

print "Results on training data"
predict(model, train_feat, train_targ)

print "Results on testing data"
predict(model, test_feat, test_targ)


"""
#Normal Equations
ne = normal_equation.normal_equation()
model = ne.solve(train_feat, train_targ)

print "Results on training data"
predict(model, train_feat, train_targ)

print "Results on testing data"
predict(model, test_feat, test_targ)
"""

"""
#SKLearn Model
print "Training..."
regr = linear_model.LinearRegression(fit_intercept=True)
# Train the model using the training sets
regr.fit(train_feat, train_targ)
# Make predictions using the testing set
pred = regr.predict(train_feat)
# The coefficients
print('Coefficients: \n', regr.coef_)
model = regr.coef_

predict(model, train_feat, train_targ)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(train_targ, pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(train_targ, pred))
"""
