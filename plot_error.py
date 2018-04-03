import matplotlib.pyplot as plt

def plot(cost):
	fig = plt.figure()
	fig.suptitle('Number of Iterations x Cost', fontsize=14, fontweight='bold')

	ax = fig.add_subplot(111)
	fig.subplots_adjust(top=0.85)

	ax.set_xlabel('Iterations')
	ax.set_ylabel('Cost')

	
	ax.plot(cost)
	
	plt.show()
