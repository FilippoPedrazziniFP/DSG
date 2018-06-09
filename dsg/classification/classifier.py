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
		max_date = df_train["TradeDateKey"].max()

		# Number of Interactions during the period
		num_int = df_train.groupby('IsinIdx').count()
		num_int["NumInt"] = num_int["TradeDateKey"]
		num_int = num_int.reset_index()
		num_int = num_int[["IsinIdx", "NumInt"]]

		# Last interactions
		last_int = df_train.groupby('IsinIdx').max()
		last_int["LastInt"] = last_int["TradeDateKey"].apply(lambda x: max_date - x)
		last_int = last_int.reset_index()
		last_int = last_int[["IsinIdx", "LastInt"]]

		# First Merge
		df = pd.merge(num_int, last_int, on=['IsinIdx'], how='left')

		# Last month Interactions
		last_month_df = df_train[df_train["TradeDateKey"] >= max_date -30]
		last_month_df = last_month_df.groupby('IsinIdx').count()
		last_month_df["LastMonthInt"] = last_month_df["TradeDateKey"]
		last_month_df = last_month_df.reset_index()
		last_month_df = last_month_df[["IsinIdx", "LastMonthInt"]]

		# Second Merge
		df = pd.merge(df, last_month_df, on=['IsinIdx'], how='left')

		# Last Week Interactions
		last_week_df = df_train[df_train["TradeDateKey"] >= max_date -7]
		last_week_df = last_week_df.groupby('IsinIdx').count()
		last_week_df["LastWeekInt"] = last_week_df["TradeDateKey"]
		last_week_df = last_week_df.reset_index()
		last_week_df = last_week_df[["IsinIdx", "LastWeekInt"]]

		# Third Merge
		df = pd.merge(df, last_week_df, on=['IsinIdx'], how='left')

		# Last 2 Week Interactions
		last_2_week_df = df_train[df_train["TradeDateKey"] >= max_date -15]
		last_2_week_df = last_2_week_df.groupby('IsinIdx').count()
		last_2_week_df["Last2WeekInt"] = last_2_week_df["TradeDateKey"]
		last_2_week_df = last_2_week_df.reset_index()
		last_2_week_df = last_2_week_df[["IsinIdx", "Last2WeekInt"]]

		# Fourth Merge
		df = pd.merge(df, last_2_week_df, on=['IsinIdx'], how='left')

		# Last 2 Month Interactions
		last_2_month_df = df_train[df_train["TradeDateKey"] >= max_date -60]
		last_2_month_df = last_2_month_df.groupby('IsinIdx').count()
		last_2_month_df["Last2MonthInt"] = last_2_month_df["TradeDateKey"]
		last_2_month_df = last_2_month_df.reset_index()
		last_2_month_df = last_2_month_df[["IsinIdx", "Last2MonthInt"]]

		# Final Merge
		df = pd.merge(df, last_2_month_df, on=['IsinIdx'], how='left')

		# Fill NAN with Zeros
		df.fillna(0, inplace=True)

		print(df.head())
		print(df.describe())
		
		df['Features'] = list(zip(df['NumInt'], df['LastInt'], df["LastMonthInt"], df["LastWeekInt"], df["Last2WeekInt"], df["Last2MonthInt"]))
		df_dict = df.groupby("IsinIdx")["Features"].apply(list).to_dict()
		
		return df_dict

	def create_customer_features_dict(self, df_train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is a list with some features:

			CustomerIdx : Freq, Last Int, Last Month, Last Week, 
				Last 2 Week, Last 2 Months
		"""
		max_date = df_train["TradeDateKey"].max()

		# Number of Interactions during the period
		num_int = df_train.groupby('CustomerIdx').count()
		num_int["NumInt"] = num_int["TradeDateKey"]
		num_int = num_int.reset_index()
		num_int = num_int[["CustomerIdx", "NumInt"]]

		# Last interactions
		last_int = df_train.groupby('CustomerIdx').max()
		last_int["LastInt"] = last_int["TradeDateKey"].apply(lambda x: max_date - x)
		last_int = last_int.reset_index()
		last_int = last_int[["CustomerIdx", "LastInt"]]

		# First Merge
		df = pd.merge(num_int, last_int, on=['CustomerIdx'], how='left')

		# Last month Interactions
		last_month_df = df_train[df_train["TradeDateKey"] >= max_date -30]
		last_month_df = last_month_df.groupby('CustomerIdx').count()
		last_month_df["LastMonthInt"] = last_month_df["TradeDateKey"]
		last_month_df = last_month_df.reset_index()
		last_month_df = last_month_df[["CustomerIdx", "LastMonthInt"]]

		# Second Merge
		df = pd.merge(df, last_month_df, on=['CustomerIdx'], how='left')

		# Last Week Interactions
		last_week_df = df_train[df_train["TradeDateKey"] >= max_date -7]
		last_week_df = last_week_df.groupby('CustomerIdx').count()
		last_week_df["LastWeekInt"] = last_week_df["TradeDateKey"]
		last_week_df = last_week_df.reset_index()
		last_week_df = last_week_df[["CustomerIdx", "LastWeekInt"]]

		# Third Merge
		df = pd.merge(df, last_week_df, on=['CustomerIdx'], how='left')

		# Last 2 Week Interactions
		last_2_week_df = df_train[df_train["TradeDateKey"] >= max_date -15]
		last_2_week_df = last_2_week_df.groupby('CustomerIdx').count()
		last_2_week_df["Last2WeekInt"] = last_2_week_df["TradeDateKey"]
		last_2_week_df = last_2_week_df.reset_index()
		last_2_week_df = last_2_week_df[["CustomerIdx", "Last2WeekInt"]]

		# Fourth Merge
		df = pd.merge(df, last_2_week_df, on=['CustomerIdx'], how='left')

		# Last 2 Month Interactions
		last_2_month_df = df_train[df_train["TradeDateKey"] >= max_date -60]
		last_2_month_df = last_2_month_df.groupby('CustomerIdx').count()
		last_2_month_df["Last2MonthInt"] = last_2_month_df["TradeDateKey"]
		last_2_month_df = last_2_month_df.reset_index()
		last_2_month_df = last_2_month_df[["CustomerIdx", "Last2MonthInt"]]

		# Final Merge
		df = pd.merge(df, last_2_month_df, on=['CustomerIdx'], how='left')

		# Fill NAN with Zeros
		df.fillna(0, inplace=True)

		print(df.head())
		print(df.describe())
		
		df['Features'] = list(zip(df['NumInt'], df['LastInt'], df["LastMonthInt"], df["LastWeekInt"], df["Last2WeekInt"], df["Last2MonthInt"]))
		df_dict = df.groupby("CustomerIdx")["Features"].apply(list).to_dict()
		
		return df_dict

	def create_cus_bond_features_dict(self, df_train):
		"""
			The method creates the dictionary with all the features related 
			to the pair custonmer - bond.
		"""
		max_date = df_train["TradeDateKey"].max()

		# Number of Interactions during the period
		num_int = df_train.groupby(['CustomerIdx', 'IsinIdx']).count()
		num_int["NumInt"] = num_int["TradeDateKey"]
		num_int = num_int.reset_index(level=['CustomerIdx', 'IsinIdx'])
		num_int = num_int[["CustomerIdx", 'IsinIdx', "NumInt"]]

		# Last interactions
		last_int = df_train.groupby(['CustomerIdx', 'IsinIdx']).max()
		last_int["LastInt"] = last_int["TradeDateKey"].apply(lambda x: max_date - x)
		last_int = last_int.reset_index(level=['CustomerIdx', 'IsinIdx'])
		last_int = last_int[["CustomerIdx", 'IsinIdx', "LastInt"]]

		# First Merge
		df = pd.merge(num_int, last_int, on=['CustomerIdx', "IsinIdx"], how='left')

		# Fill NAN with Zeros
		df.fillna(100, inplace=True)

		# Last month Interactions
		last_month_df = df_train[df_train["TradeDateKey"] >= max_date -30]
		last_month_df = last_month_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_month_df["LastMonthInt"] = last_month_df["TradeDateKey"]
		last_month_df = last_month_df.reset_index()
		last_month_df = last_month_df[["CustomerIdx", 'IsinIdx', "LastMonthInt"]]

		# Second Merge
		df = pd.merge(df, last_month_df, on=['CustomerIdx', "IsinIdx"], how='left')

		# Last Week Interactions
		last_week_df = df_train[df_train["TradeDateKey"] >= max_date -7]
		last_week_df = last_week_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_week_df["LastWeekInt"] = last_week_df["TradeDateKey"]
		last_week_df = last_week_df.reset_index()
		last_week_df = last_week_df[["CustomerIdx", 'IsinIdx', "LastWeekInt"]]

		# Third Merge
		df = pd.merge(df, last_week_df, on=['CustomerIdx', "IsinIdx"], how='left')

		# Last 2 Week Interactions
		last_2_week_df = df_train[df_train["TradeDateKey"] >= max_date -15]
		last_2_week_df = last_2_week_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_2_week_df["Last2WeekInt"] = last_2_week_df["TradeDateKey"]
		last_2_week_df = last_2_week_df.reset_index()
		last_2_week_df = last_2_week_df[["CustomerIdx", 'IsinIdx', "Last2WeekInt"]]

		# Fourth Merge
		df = pd.merge(df, last_2_week_df, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Last 2 Month Interactions
		last_2_month_df = df_train[df_train["TradeDateKey"] >= max_date -60]
		last_2_month_df = last_2_month_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_2_month_df["Last2MonthInt"] = last_2_month_df["TradeDateKey"]
		last_2_month_df = last_2_month_df.reset_index()
		last_2_month_df = last_2_month_df[["CustomerIdx", 'IsinIdx', "Last2MonthInt"]]

		# Final Merge
		df = pd.merge(df, last_2_month_df, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Fill NAN with Zeros
		df.fillna(0, inplace=True)

		print(df.head())
		print(df.describe())
		
		df['Features'] = list(zip(df['NumInt'], df['LastInt'], df["LastMonthInt"], df["LastWeekInt"], df["Last2WeekInt"], df["Last2MonthInt"]))
		df_dict = df.groupby(['CustomerIdx', "IsinIdx"])["Features"].apply(list).to_dict()
		
		return df_dict

	def fit(self, X_train_df, y_train_df):

		# Create Customer Dictionary
		self.customer_dictionary = self.create_customer_features_dict(X_train_df)

		# Create Bond Dictionary
		self.bond_dictionary = self.create_bonds_features_dict(X_train_df)

		# Create Customer - Bond Dictionary
		self.cus_bond_dictionary = self.create_cus_bond_features_dict(X_train_df)

		# Create Train set with the dictionaries
		train = self.create_set(y_train_df)

		"""
			Save the train set into a Data Frame
		"""

		print(train[0:5])
		print(train.shape)

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
			try:
				customer_features = self.customer_dictionary[sample[0]]
				customer_features = np.asarray(customer_features)
			except KeyError:
				customer_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
			
			try: 
				bond_features = self.bond_dictionary[sample[1]]
				bond_features = np.asarray(bond_features)
			except KeyError:
				bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])

			try: 
				cus_bond_features = self.cus_bond_dictionary[(sample[0], sample[1])]
				cus_bond_features = np.asarray(cus_bond_features)
			except KeyError:
				cus_bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
			
			label = sample[2]
			row = np.append(customer_features, bond_features)
			row = np.append(row, cus_bond_features)
			row = np.append(row, label)
			train_set.append(row)
		train_set = np.asarray(train_set)
		return train_set

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

	def predict(self, X):
		predictions = []
		for sample in X:
			try:
				customer_features = self.customer_dictionary[sample[0]]
				customer_features = np.asarray(customer_features)
			except KeyError:
				customer_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
			
			try: 
				bond_features = self.bond_dictionary[sample[1]]
				bond_features = np.asarray(bond_features)
			except KeyError:
				bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])

			try: 
				cus_bond_features = self.cus_bond_dictionary[(sample[0], sample[1])]
				cus_bond_features = np.asarray(cus_bond_features)
			except KeyError:
				cus_bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
			
			features = np.append(customer_features, bond_features)
			features = np.append(features, cus_bond_features)
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
		
		