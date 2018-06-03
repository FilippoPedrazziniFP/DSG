from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import islice
import pickle
import operator
from itertools import chain
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

class BaselineClassier(object):
	def __init__(self):
		super(BaselineClassier, self).__init__()
		self.model = self.build_model()

	def build_model(self):
		return

	def features_labels_split(self, X):
		"""
			The method splits the data into Features and Labels
			returning a numpy array
			@args
				X : numpy array
			@return
				features : numpy array
				labels : numpy array
		"""
		features = X[:,0]
		labels = X[:,1]
		return features, labels

	def take(self, n, iterable):
		return list(islice(iterable, n))

	def create_dictionary(self, train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is a list with 2 features 
			[Normalized Frequency, 
				Sum of Interest of the interests]
		"""

		# Creating Frequency Dictionary 
		customers_ids = np.unique(train[:,1])
		cidx_freq = {}
		for c in customers_ids:
		    i = 0
		    for sample in train:
		        if sample[1] == c:
		            i=i+1
		    cidx_freq[c] = i

		# Creating Interest Dictionary 
		last_date = train[-1,0]
		print(last_date)
		cidx_int = {}
		for c in customers_ids:
			for sample in train:
				interest = 0
				if sample[0] == last_date:
					interest = interest + sample[4]
			cidx_int[c] = interest

		# Merge the 2 dictionaries
		cidx_freq_int = defaultdict(list)
		for k, v in chain(cidx_freq.items(), cidx_int.items()):
			cidx_freq_int[k].append(v)

		# Creating Normalized Frequency Dictionary
		max_value = self.get_max_value(cidx_freq_int)
		print("MAX VALUE: ", max_value)
		for k, v in cidx_freq_int.items():
			cidx_freq_int[k] = [v[0]/max_value, v[1]]
		
		return cidx_freq_int

	def load_dictionary(self, X):
		# Create Frequency Dictionary
		try:
			self.cidx_freq_int = self.restore_dictionary()
		except:
			self.cidx_freq_int = self.create_dictionary(X)
		return

	def fit(self, train):
		# Load Dictionary or create a new one
		self.load_dictionary(train)
		
		# Create Train set having a dictionary
		train = self.create_set(self.cidx_freq_int)
		print(train[0:10])
		
		# Split Features and Labels
		X, y = self.features_labels_split(train)

		# Reshape Features
		X = np.reshape(X, (-1, 1))

		# Train Classifier
		self.classifier = self.train_classifier(X, y)
		return

	def create_set(self, dictionary):
		"""
			The method creates a matrix with 2 columns:
			CustomerIdx, NormalizedFrequency given a dictionary
			of CustomersIdx : Normalized Values.
		"""
		train = []
		for k, v in dictionary.items():
			train.append(v)
		train = np.asarray(train)
		return train

	def train_classifier(self, X_train, y_train):
		"""
			Simple classifier to put a weight 
			to the frequency feature.
		"""
		model = LogisticRegression()
		model.fit(X_train, y_train)
		return model

	def get_max_value(self, dictionary):
		max_value = 0
		for k, v in dictionary.items():
			if v[0] > max_value:
				max_value = v[0]
		return max_value

	def print_dictionary(self, dictionary=None):
		if dictionary is not None:
			print_dict = dictionary
		else:
			print_dict = self.cidx_freq_int
		n_items = self.take(10, print_dict.items())
		for key, val in n_items:
			print(key, val)
		return

	def save_dictionary(self):
		"""
			The method saves the training 
			dictionary in a pickle file.

		"""
		try:
			pickle.dump(self.cidx_freq_int, open("./weights/cidx_freq_int.p", "wb"))
			print("Dictionaries saved...")
		except:
			pass
		return

	def restore_dictionary(self):
		"""
			The method restore the dictionary
			from a pickle file.
		"""
		try:
			cidx_freq_int = pickle.load(open("./weights/cidx_freq_int.p", "rb"))
			print("Model restored...")
		except:
			pass
		return cidx_freq_int

	def predict(self, X):
		predictions = []
		for sample in X:
			try: 
				features = self.cidx_freq_int[sample[1]]
				pred = self.classifier.predict(features)
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions

	def evaluate(self, test):
		y_pred = self.predict(X)
		score = roc_auc_score(y_test, y_pred)
		return score

	def predict_for_submission(self, X):
		predictions = []
		for sample in X:
			try:
				features = self.cidx_freq_int[sample[2]]
				pred = self.classifier.predict(features)
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions
		