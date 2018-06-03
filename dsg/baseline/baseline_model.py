from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import islice
import pickle
import operator

class Baseline(object):
	def __init__(self):
		super(Baseline, self).__init__()
		self.model = self.build_model()

	def build_model(self):
		return

	def take(self, n, iterable):
		return list(islice(iterable, n))

	def create_frequency_dictionary(self, X_train, y_train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is the normalized frequency of the interactions.
		"""
		# Creating Frequency Dictionary
		customers_ids = np.unique(X_train[:,1])
		self.cidx_freq = {}
		for c in customers_ids:
		    i = 0
		    for sample in X_train:
		        if sample[1] == c:
		            i=i+1
		    self.cidx_freq[c] = i

		# Creating Normalized Frequency Dictionary
		max_value = self.get_max_value()
		print("MAX VALUE: ", max_value)
		for k, v in self.cidx_freq.items():
			self.cidx_freq[k] = v/max_value
		return

	def create_train_set(self):
		
		return

	def fit(self, X_train, y_train):
		"""
			Simple classifier to put a weight 
			to the frequency feature.
		"""

		return

	def get_max_value(self):
		max_value = 0
		for k, v in self.cidx_freq.items():
			if v > max_value:
				max_value = v
		return max_value

	def print_dictionary(self):
		n_items = self.take(10, self.cidx_freq.items())
		for key, val in n_items:
			print(key, val)
		return

	def save_dictionary(self):
		"""
			The method saves the training 
			dictionary in a pickle file.

		"""
		try:
			pickle.dump(self.cidx_freq, open("./weights/cust_dict.p", "wb"))
			print("Model saved...")
		except:
			pass
		return

	def restore_dictionary(self):
		"""
			The method restore the dictionary
			from a pickle file.
		"""
		try:
			self.cidx_freq = pickle.load(open("./weights/cust_dict.p", "rb"))
			print("Model restored...")
		except:
			pass
		return

	def predict(self, X):
		predictions = []
		for sample in X:
			try: 
				pred = self.cidx_freq[sample[1]]
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, y_pred)
		return score

	def predict_for_submission(self, X):
		predictions = []
		for sample in X:
			try:
				pred = self.cidx_freq[sample[2]]
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions
		