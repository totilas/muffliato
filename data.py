from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def load(dataset="Houses"):
	"""
	Load and preprocess Houses dataset
	Parameters
	----------
	dataset : str, default: "Houses"
		Name of the dataset to be fetched (other dataset currently not implemented)
	Returns
	-------
	X_train: array
		the samples use for training
	X_test: array
		the samples for testing
	y_train: array
		labels of the training set
	y_test: array
		labels of the testing set
	"""
	if dataset == "Houses":
		X, y = fetch_openml(name='houses', version=2, return_X_y=True, as_frame=False)
		n, d = X.shape
		c = np.unique(y)    
		y[y==c[0]] = -1
		y[y==c[1]] = 1
		y = y.astype(float)

		standardizer = StandardScaler()
		X = standardizer.fit_transform(X)

	normalizer = Normalizer()
	X = normalizer.transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7755, random_state=42, stratify=y)

	return X_train, X_test, y_train, y_test