from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from itertools import islice
import os
from sklearn.preprocessing import StandardScaler
from dsg.visualizer import Explorer
from catboost import CatBoostClassifier
from dsg.data_loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

class Classifier(object):
	def __init__(self):
		super(Classifier, self).__init__()
	
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
		features = X[:, :-1]
		labels = X[:, -1]
		return features, labels

	def take(self, n, iterable):
		return list(islice(iterable, n))

	def create_bonds_features(self):
		"""
			The method creates a dictionary with
			BondIdx : features.
		"""
		if os.path.isfile("data/bond_frequency_features2018.csv"):
			bond_features = pd.read_csv("data/bond_frequency_features2018.csv")
		else:
			print("FILE NOT FOUND: data/bond_frequency_features2018.csv")

		return bond_features


	def create_customer_features(self):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is a list with some features:

			CustomerIdx : Freq, Last Int, Last Month, Last Week, 
				Last 2 Week, Last 2 Months
		"""
		if os.path.isfile("data/cust_frequency_features2018.csv"):
			cust_features = pd.read_csv("data/cust_frequency_features2018.csv")
		else:
			print("FILE NOT FOUND: data/cust_frequency_features2018.csv")
		return cust_features

	def create_cus_bond_features(self):
		"""
			The method creates the dictionary with all the features related 
			to the pair custonmer - bond.
		"""

		if os.path.isfile("data/bondcust_frequency_features2018.csv"):
			bondcust_features = pd.read_csv("data/bondcust_frequency_features2018.csv")
		else:
			print("FILE NOT FOUND: data/bondcust_frequency_features2018.csv")

		return bondcust_features


	def fit(self, train):

		# Create Customer Dictionary
		self.customer_dictionary = self.create_customer_features()

		# Create Bond Dictionary
		self.bond_dictionary = self.create_bonds_features()

		# Create Customer - Bond Dictionary
		self.cus_bond_dictionary = self.create_cus_bond_features()

		# Create Train set with the dictionaries
		train = self.create_set(train)


		"""
			Save the train set into a Data Frame
		"""

		print(train.columns)
		print(train[0:5])
		print(train.shape)

		print("CREATED TRAIN SET; STARTING TO FIT THE MODEL..")

		# Split Features and Labels
		print(train.describe())
		X, y = self.features_labels_split_df(train)

		# Train Classifier
		print("QUA")
		print(np.mean(y))
		self.classifier = self.train_classifier(X, y)

		return

	def create_set(self, train):
		"""
			The method creates a matrix with 2 columns:
			[CustomerIdx, NormalizedFrequency] given a dictionary
			of CustomersIdx : Normalized Values.
		"""
		train = train.merge(self.customer_dictionary, on=["CustomerIdx", "DateKey"], how='left')
		train = train.merge(self.bond_dictionary, on=["IsinIdx", "DateKey"], how='left')
		train = train.merge(self.cus_bond_dictionary, on=["CustomerIdx","IsinIdx", "DateKey"], how='left')
		train = train.drop(["Date_BuySell","Date_BuySell_x","Date_BuySell_y","CustomerIdx","IsinIdx", "DateKey"], axis=1)

		if "PredictionIdx" in train.columns:
			train = train.drop(["PredictionIdx"], axis=1)

		train.loc[train["BuySell"] == "Buy", "BuySell" ] = 1
		train.loc[train["BuySell"] == "Sell", "BuySell"] = 0

		train = train.fillna(0)

		return train

	def train_classifier(self, X_train, y_train):
		raise NotImplementedError

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

	def predict(self, test):
		#test = self.create_set(test)
		#X_test, y_test = self.features_labels_split(test)
		y_pred = self.classifier.predict_proba(test)[:,1]

		return y_pred

	def predict_test(self, test):
		test = self.create_set(test)
		X_test, y_test = self.features_labels_split_df(test)
		y_pred = self.predict(X_test)

		return y_pred
	def evaluate(self, test):
		test = self.create_set(test)
		X_test, y_test = self.features_labels_split_df(test)
		y_pred = self.predict(X_test)
		print(y_pred)
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

class CATBoost(Classifier):
	def __init__(self):
		super(CATBoost, self).__init__()

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
		print(np.mean(y_train))
		print(y_train)
		print(y_train.std())
		print(y_train.max())

		# Fit the model
		model = CatBoostClassifier(verbose=False)
		model.fit(X_train, y_train)
		return model

class RF(Classifier):
	def __init__(self):
		super(RF, self).__init__()

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
		print(y_train.mean())
		print(y_train.std())
		print(y_train.max())

		# Fit the model
		model = RandomForestClassifier()
		model.fit(X_train, y_train)
		return model

class KNN(Classifier):
	def __init__(self):
		super(KNN, self).__init__()

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
		print(y_train.mean())
		print(y_train.std())
		print(y_train.max())

		# Fit the model
		model = KNeighborsClassifier()
		model.fit(X_train, y_train)
		return model
		
		