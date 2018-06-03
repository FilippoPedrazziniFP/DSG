from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import islice
import pickle
import operator
from itertools import chain
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from dsg.data_loader import DataLoader

class Classier(object):
	def __init__(self):
		super(Classier, self).__init__()

	def create_cus_bond_freq_dictionary(self, df):
		"""
			The method creates the dictionary in which each
			key is represented by (CustomerIdx, BondIdx) and 
			the value is the number of historical occurrences 
			betweem the two. 
		"""
		return

	def create_cus_freq_dictionary(self, df):
		"""		
			The method creates a dictionary based on the frequence
			of interaction during the considered period.
			CustomerIdx : Frequency
		"""
		dictionary = df.groupby('CustomerIdx').count()['TradeDateKey'].to_dict()
		return dictionary

	def create_cus_int_dictionary(self, df):
		"""		
			The method creates a dictionary based on the frequence
			of interaction during the considered period.
		"""
		max_value = df_trade["TradeDateKey"].max()
		dictionary = df[df["TradeDateKey"] == max_value].groupby("CustomerIdx").count()["CustomerInterest"].to_dict()
		return dictionary

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

	def create_dictionary(self, df_train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is a list with 2 features 
			[Normalized Frequency, 
				Sum of Interest]
		"""
		# Creating Frequency Dictionary 
		cidx_freq = self.create_cus_freq_dictionary(df_train)

		# Creating Interest Dictionary
		cidx_int = self.create_cus_int_dictionary(df_train)

		# Merge the 2 dictionaries
		cidx_freq_int = defaultdict(list)
		for k, v in chain(cidx_freq.items(), cidx_int.items()):
			cidx_freq_int[k].append(v)

		# Normalize Frequency 
		max_value = self.get_max_value(cidx_freq_int)
		print("MAX VALUE: ", max_value)
		for k, v in cidx_freq_int.items():
			cidx_freq_int[k] = [v[0]/max_value, v[1]]
		
		return cidx_freq_int

	def fit(self, df_train):

		# Create Customer Freq Dictionary
		features_dictionary = self.create_dictionary(df_train)

		# Create Train set having a dictionary
		train = self.create_set(features_dictionary)
				
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
			[CustomerIdx, NormalizedFrequency] given a dictionary
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
		