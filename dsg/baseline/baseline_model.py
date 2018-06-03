from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import islice
import pickle
import operator
from itertools import chain
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

class Baseline(object):
	def __init__(self):
		super(Baseline, self).__init__()
		self.model = self.build_model()

	def build_model(self):
		return

	def take(self, n, iterable):
		return list(islice(iterable, n))

	def create_dictionary(self, train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and
			[Normalized Frequency]
		"""
		# Creating Frequency Dictionary 
		customers_ids = np.unique(train[:,1])
		cidx_freq = {}
		for c in customers_ids:
		    i = 0
		    discounted_reward = 0
		    # Looping over the date
		    for sample in train:
		        if sample[1] == c:
		            i=i+1*discounted_reward
		        discounted_reward = discounted_reward + 0.1
		    cidx_freq[c] = i

		# Creating Normalized Frequency Dictionary
		max_value = self.get_max_value(cidx_freq)
		print("MAX VALUE: ", max_value)
		for k, v in cidx_freq.items():
			cidx_freq[k] = v/max_value
		
		return cidx_freq

	def load_dictionary(self, X):
		# Create Frequency Dictionary
		try:
			self.cidx_freq = self.restore_dictionary()
		except:
			self.cidx_freq = self.create_dictionary(X)
		return

	def fit(self, train):
		# Load Dictionary or create a new one
		self.load_dictionary(train)
		return

	def get_max_value(self, dictionary):
		max_value = 0
		for k, v in dictionary.items():
			if v > max_value:
				max_value = v
		return max_value

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
		features = X[:,0:4]
		labels = X[:,4]
		return features, labels

	def print_dictionary(self, dictionary=None):
		if dictionary is not None:
			print_dict = dictionary
		else:
			print_dict = self.cidx_freq
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
			pickle.dump(self.cidx_freq, open("./weights/cidx_freq.p", "wb"))
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
			cidx_freq = pickle.load(open("./weights/cidx_freq.p", "rb"))
			print("Model restored...")
		except:
			pass
		return cidx_freq_int

	def predict(self, X):
		predictions = []
		for sample in X:
			try: 
				pred = self.cidx_freq[sample[1]]
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions

	def evaluate(self, test):
		raise NotImplementedError

	def predict_for_submission(self, X):
		predictions = []
		for sample in X:
			try:
				pred = self.cidx_freq[sample[2]]
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions
		