from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import islice
import pickle
import operator
from itertools import chain
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression
from dsg.data_loader import DataLoader
from sklearn.preprocessing import StandardScaler
from dsg.visualizer import Explorer
from catboost import CatBoostClassifier

class Classifier(object):
	def __init__(self, max_rating=5):
		super(Classifier, self).__init__()
		self.max_rating = max_rating

	def create_cus_bond_freq_dictionary(self, df):
		"""
			The method creates the dictionary in which each
			key is represented by (CustomerIdx, BondIdx) and 
			the value is the number of historical occurrences 
			betweem the two. 
		"""
		dictionary = df.groupby(['CustomerIdx', 'IsinIdx']).count()['TradeDateKey'].to_dict()
		return dictionary

	def create_bond_freq_dictionary(self, df):
		"""
			The method creates a dictionary with 
			BondsIdx : frequency
		"""
		dictionary = df.groupby('IsinIdx').count()['TradeDateKey'].to_dict()
		return dictionary

	def create_cus_freq_dictionary(self, df):
		"""		
			The method creates a dictionary based on the frequence
			of interaction during the considered period.
			CustomerIdx : Frequency
		"""
		dictionary = df.groupby('CustomerIdx').count()['TradeDateKey'].to_dict()
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
		features = X[:,0:-1]
		labels = X[:,-1]
		return features, labels

	def take(self, n, iterable):
		return list(islice(iterable, n))

	def create_bonds_features_dict(self, df_train):
		"""
			The method creates a dictionary with
			BondIdx : features.
		"""
		bidx_freq = self.create_bond_freq_dictionary(df_train)
		return bidx_freq

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
		return cidx_freq

	def create_cus_bond_features_dict(self, df_train):
		"""
			The method creates the dictionary with all the features related 
			to the pair custonmer - bond.
		"""
		cbidx_freq = self.create_cus_bond_freq_dictionary(df_train)
		return cbidx_freq

	def fit(self, X_train_df, y_train_df):

		# Create Customer Dictionary
		self.customer_dictionary = self.create_customer_features_dict(X_train_df)

		# Create Bond Dictionary
		self.bond_dictionary = self.create_bonds_features_dict(X_train_df)

		# Create Customer - Bond Dictionary
		self.cus_bond_dictionary = self.create_cus_bond_features_dict(X_train_df)

		# Create Train set with the dictionaries
		train = self.create_set(y_train_df)

		print("CREATED TRAIN SET; STARTING TO FIT THE MODEL..")

		# Split Features and Labels
		X, y = self.features_labels_split(train)

		# Train Classifier
		self.classifier = self.train_classifier(X, y)

		return

	def create_set(self, df):
		"""
			The method creates a matrix with 2 columns:
			[CustomerIdx, NormalizedFrequency] given a dictionary
			of CustomersIdx : Normalized Values.
		"""
		train = df.values
		train_set = []
		for sample in train:
			row = []
			
			try:
				customer_features = self.customer_dictionary[sample[0]]
			except KeyError:
				customer_features = 0
			row.append(customer_features)
			
			try: 
				bond_features = self.bond_dictionary[sample[1]]
			except KeyError:
				bond_features = 0
			row.append(bond_features)

			try: 
				cus_bond_features = self.cus_bond_dictionary[(sample[0], sample[1])]
			except KeyError:
				cus_bond_features = 0
			row.append(cus_bond_features)
			
			label = sample[2]
			row.append(label)
			train_set.append(row)
		train_set = np.asarray(train_set)
		return train_set

	def train_classifier(self, X_train, y_train):
		"""
			Simple classifier to put a weight 
			to the frequency feature.
		"""
		self.scaler = StandardScaler()
		self.scaler.fit(X_train)
		X_train = self.scaler.transform(X_train)

		# Plot Distribution of Labels
		# Explorer.plot_array(y_train)
		# print(y_train.mean())
		# print(y_train.std())
		# print(y_train.max())

		# Fit the model
		model = CatBoostClassifier(verbose=False)
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
			features = []
			
			try: 
				cust_features = self.customer_dictionary[sample[0]]
			except KeyError:
				cust_features = 0
			
			features.append(cust_features)

			try:
				bond_features = self.bond_dictionary[sample[1]]
			except KeyError:
				bond_features = 0

			features.append(bond_features)

			try:
				cus_bond_features = self.cus_bond_dictionary[(sample[0], sample[1])]
			except KeyError:
				cus_bond_features = 0
			
			features.append(cus_bond_features)
			
			features = np.asarray(features)
			features = np.reshape(features, (1, -1))
			features = self.scaler.transform(features)
			pred = 1 - self.classifier.predict_proba(features)[0][0]
			predictions.append(pred)
		predictions = np.array(predictions)
		print(predictions.max())
		print(predictions[0:10])
		return predictions

	def evaluate(self, y_test_df):
		X_test, y_test = self.features_labels_split_df(y_test_df)
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, list(y_pred))
		return score

	def features_labels_split_df(self, test_df):
		labels = test_df["CustomerInterest"]
		features = test_df.drop(["CustomerInterest"], axis=1)
		return features.values, labels.values

	def predict_baseline(self, X):
		predictions = []
		for sample in X:
			features = 0
			
			try: 
				cust_features = self.customer_dictionary[sample[0]]
			except KeyError:
				cust_features = 0
			
			features = features + cust_features

			try:
				bond_features = self.bond_dictionary[sample[1]]
			except KeyError:
				bond_features = 0

			features = features + bond_features

			try:
				cus_bond_features = self.cus_bond_dictionary[(sample[0], sample[1])]
			except KeyError:
				cus_bond_features = 0
			
			features = features + cus_bond_features
			pred = features
			predictions.append(pred)
		predictions = np.array(predictions)
		predictions = predictions/predictions.max()
		return predictions
		