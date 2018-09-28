from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
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
from dsg.data_loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

class Classifier(object):
	def __init__(self):
		super(Classifier, self).__init__()
	
	def take(self, n, iterable):
		return list(islice(iterable, n))

	def fit(self, X_train, y_train):
		# Train Classifier
		self.classifier = self.train_classifier(X_train, y_train)
		return

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

	def predict_by_line(self, X):
		predictions = []
		for sample in X:
			features = self.scaler.transform(sample)
			features = np.reshape(features, (1, -1))
			pred = 1 - self.classifier.predict_proba(features)[0][0]
			predictions.append(pred)
		predictions = np.array(predictions)
		print(predictions.max())
		print(predictions[0:10])
		return predictions

	def predict(self, X):
		X = self.scaler.transform(X)
		predictions = self.classifier.predict_proba(X)[:,1]
		print(predictions.max())
		print(predictions[0:10])
		return predictions

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, y_pred)
		return score

	def features_labels_split_df(self, test_df):
		labels = test_df["CustomerInterest"]
		features = test_df.drop(["CustomerInterest"], axis=1)
		return features.values, labels.values

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
		print(y_train.mean())
		print(y_train.std())
		print(y_train.max())

		# Fit the model
		model = CatBoostClassifier(
			verbose=False, 
			custom_metric="AUC"
			)
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

class LR(Classifier):
	def __init__(self):
		super(LR, self).__init__()

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
		model = LogisticRegression()
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
		
		