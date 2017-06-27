import numpy as np 
import scipy.optimize as opt

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# prediction about class y based on value h(X)
def predict(theta,X):
	return sigmoid(np.dot(theta,X)) > 0.5 or -1

def log_likelihood(X,Y,theta,alpha=0.1):
	return np.sum(np.log(sigmoid(Y * np.dot(X,theta)))) - alpha/2 * np.dot(theta,theta)

def log_likelihood_grad(X,Y,theta,alpha=0.1):
	n = len(theta)
	m = len(X)
	grad = np.zeros(n)

	for i in range(m):
		grad += Y[i] * X[i] * sigmoid(-Y[i] * np.dot(X[i],theta))

	grad -= alpha * theta

	return grad

# Computing numerical gradient ( ( f(a+b)-f(a-b) ) / 2*(a-b) ) 
# to make sure gradient function works
def grad_num(X,Y,theta,f,eps=0.00001):
	n = len(theta)
	ident = np.identity(n)
	g = np.zeros(n)

	for i in range(n):
		g[i] += f(X, Y, theta + eps * ident[i])
		g[i] -= f(X, Y, theta - eps * ident[i])
		g[i] /= 2*eps 

	return g

def test_log_likelihood_grad(X,Y):
	n_attr = X.shape[1]
	theta = np.array([1.0 / n_attr] * n_attr)

	print("grad asc")
	print(log_likelihood_grad(X,Y,theta))

	print("grad num")
	print(grad_num(X,Y,theta,log_likelihood_grad))

# gradient ascent
def train_theta(X,Y,alpha=0.1):
	def f(theta):
		return -log_likelihood(X,Y,theta,alpha)
	def fprime(theta):
		return -log_likelihood_grad(X,Y,theta,alpha)

	n = X.shape[1]
	initial_guess = np.zeros(n)

	return opt.fmin_bfgs(f, initial_guess,fprime, disp=False)

# gauge predicted vs real on each training example to test accuracy
def accuracy(X,Y,theta):
	n_correct = 0
	for i in range(len(X)):
		if predict(theta,X[i]) == Y[i]:
			n_correct += 1
	return n_correct * 1.0 / len(X)

# Evaluating regularization penalty alpha with K-fold cross validation set
def fold(arr, K, i):
	N = len(arr)
	size = np.ceil(1.0 * N / K)
	arange = np.arange(N) # all indices
	heldout = np.logical_and(i * size <= arange, arange < (i+1) * size)
	rest = np.logical_not(heldout)
	return arr[heldout], arr[rest]

# Returns K distinct folds for arr
def kfold(arr,K):
	return [fold(arr,K,i) for i in range(K)]

def avg_accuracy(all_X,all_Y,alpha):
	s = 0
	K = len(all_X)
	for i in range(K):
		X_heldout, X_rest = all_X[i]
		Y_heldout, Y_rest = all_Y[i]
		theta = train_theta(X_rest,Y_rest,alpha)
		s += accuracy(X_heldout,Y_heldout,theta)
	return s * 1.0/ K

def train_alpha(X,Y,K=10):
	all_alpha = np.arange(0,1,0.1)
	all_X = kfold(X,K)
	all_Y = kfold(Y,K)
	all_acc = np.array([avg_accuracy(all_X,all_Y,alpha) for alpha in all_alpha])
	# returns alpha with the highest avg accuracy score
	return all_alpha[all_acc.argmax()]

def read_data(filename, sep=",", filt=int):

	def split_line(line):
		return line.split(sep)

	def apply_filt(values):
		return map(filt, values)

	def process_line(line):
		return apply_filt(split_line(line))

	f = open(filename)
	lines = map(process_line, f.readlines())
	# "[1]" below corresponds to x0
	X = np.array([[1] + l[1:] for l in lines])
	# "or -1" converts 0 values to -1
	Y = np.array([l[0] or -1 for l in lines])
	f.close()

	return X, Y

def main():
	# 80 training examples and 23 features
	#X_train, Y_train = read_data("SPECT.train")
	data = np.loadtxt('ex2data2.txt', delimiter=',')
	X_train = data[:,0:2]
	Y_train = data[:,2]

	#test_log_likelihood_grad(X_train, Y_train); exit()

	alpha = train_alpha(X_train, Y_train)
	print("alpha = ")
	print(alpha)

	theta = train_theta(X_train, Y_train, alpha)
	print("theta = ")
	print(theta)

	X_test, Y_test = read_data("SPECT.test")
	print("accuracy = ")
	print(accuracy(X_train,Y_train,theta))


if __name__ == "__main__":
	main()