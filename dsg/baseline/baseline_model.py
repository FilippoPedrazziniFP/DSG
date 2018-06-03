from sklearn.metrics import roc_auc_score
import numpy as np
import pickle

class Baseline(object):
	def __init__(self):
		super(Baseline, self).__init__()
		self.model = self.build_model()

	def build_model(self):
		return

	def fit(self, X_train, y_train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is the normalized frequency of the interactions.
		"""
		customers_ids = np.unique(X_train[:,1])
		self.cidx_freq = {}
		for c in customers_ids:
		    i = 0
		    for sample in X_train:
		        if sample[1] == c:
		            i=i+1
		    self.cidx_freq[c] = i
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
			pickle.load(open("./weights/cust_dict.p", "rb"))
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
		