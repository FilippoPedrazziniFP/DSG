from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import islice
import pickle
import operator
from itertools import chain
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from dsg.data_loader import DataLoader

class Classifier(object):
	def __init__(self):
		super(Classifier, self).__init__()

	def create_cus_bond_freq_dictionary(self, df):
		"""
			The method creates the dictionary in which each
			key is represented by (CustomerIdx, BondIdx) and 
			the value is the number of historical occurrences 
			betweem the two. 
		"""
		return

	def create_bond_freq_dictionary(self, df):
		"""
			The method creates a dictionary with 
			BondsIdx : frequency
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
			The method creates a dictionary with 
			CustomerIdx : Interest
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

	def create_customer_features_dict(self, df_train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is a list with 2 features 
			[Normalized Frequency, 
				Sum of Interest]
		"""
		# Creating Customer - Frequency Dictionary 
		cidx_freq = self.create_cus_freq_dictionary(df_train)

		# Normalize Frequency Dictionary for better fitting.
		max_value = self.get_max_value(cidx_freq)
		print("MAX VALUE: ", max_value)
		for k, v in cidx_freq.items():
			cidx_freq[k] = v/max_value
		
		return cidx_freq

	def fit(self, df_train):

		# Create Customer Dictionary
		self.customer_dictionary = self.create_customer_features_dict(df_train)

		# Create Bond Dictionary


		# Create Train set with the dictionaries
		train = self.create_set(df_train, self.customer_dictionary)
				
		# Split Features and Labels
		X, y = self.features_labels_split(train)

		# Reshape Features
		X = np.reshape(X, (-1, 1))

		# Train Classifier
		self.classifier = self.train_classifier(X, y)
		return

	def create_set(self, df, dictionary):
		"""
			The method creates a matrix with 2 columns:
			[CustomerIdx, NormalizedFrequency] given a dictionary
			of CustomersIdx : Normalized Values.
		"""
		train = df.values
		train_set = []
		for sample in train:
			row = []
			customer_features = self.customer_dictionary[sample[0]]
			row.append(customer_features)
			label = sample[3]
			row.append(label)
			train_set.append(row)
		train_set = np.asarray(train_set)
		return train_set

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
			if v > max_value:
				max_value = v
		return max_value

	def print_dictionary(self, dictionary=None):
		n_items = self.take(10, dictionary.items())
		for key, val in n_items:
			print(key, val)
		return

	def predict(self, X):
		predictions = []
		for sample in X:
			try: 
				features = self.customer_dictionary[sample[0]]
				pred = self.classifier.predict(features)
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions

	def evaluate(self, test):
		X_test, y_test = self.features_labels_split_df(test)
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, y_pred)
		return score

	def features_labels_split_df(self, test_df):
		labels = test_df["CustomerInterest"]
		features = test_df.drop(["CustomerInterest"], axis=1)
		return features.values, labels.values

	def predict_for_submission(self, X):
		predictions = []
		for sample in X:
			try:
				features = self.customer_dictionary[sample[2]]
				pred = self.classifier.predict(features)
			except KeyError:
				pred = 0
			predictions.append(pred)
		return predictions
		