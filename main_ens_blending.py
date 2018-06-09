import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
from dsg.data_loader import DataLoader
from dsg.recommenders.urm_preprocessing import URMPreprocessing
from dsg.recommenders.collaborative import SVDRec, AsynchSVDRec
from dsg.regressor.regressor_preprocessing import RegressorPreprocessor
from dsg.classification.classifier_preprocessing import ClassifierPreprocessor
from dsg.classification.classifier import CATBoost, KNN
from dsg.regressor.regressor import Regressor
from sklearn.linear_model import LogisticRegression

import dsg.util as util

parser = argparse.ArgumentParser()

""" General Parameters """
args = parser.parse_args()

def create_submission_file(loader, preprocessor, meta_class, models):
	test = loader.load_challenge_data()
	print(test.head())
	X = preprocessor.test_transform(test)
	print(X[0:5])
	predictions = []
	for model in models:
		preds = model.predict(X)
		predictions.append(preds)
	predictions = np.asarray(predictions)
	predictions = np.sum(predictions, axis=0) / len(models)
	submission = loader.load_submission_file()
	submission["CustomerInterest"] = predictions
	submission.to_csv(util.SUBMISSION, index=False)
	return

def train_meta_classifier(val, models):
	X, y = model[0].features_label_split_df(val)
	print(X[0:5])
	print(x.shape)
	predictions = []
	for model in models:
		pred = model.predict(X)
		predictions.append(pred)

	predictions = np.asarray(predictions)
	print(predictions.shape)
	predictions = predictions.T
	print(predictions.shape)
	meta_class = LogisticRegression()
	meta_class.fit(predictions, y)
	return meta_class

def evaluate_meta_classifier(test, meta_class, models):
	X, y = model[0].features_label_split_df(test)
	print(X[0:5])
	print(x.shape)
	predictions = []
	for model in models:
		pred = model.predict(X)
		predictions.append(pred)

	predictions = np.asarray(predictions)
	print(predictions.shape)
	predictions = predictions.T
	print(predictions.shape)
	meta_class = LogisticRegression()
	final_predictions = meta_class.predict(predictions)
	score = roc_auc_score(y_test, list(final_predictions))
	return score

def main():

	# Fixing the seed
	np.random.seed(0)
	tf.set_random_seed(0)

	start = time.clock()

	# Load the Data
	loader = DataLoader()
	df = loader.load_trade_data()

	###### REGRESSORS

	# Clean Trade Data
	preprocessor_urm = URMPreprocessing(
		test_date=20180416,
		val_date=20180409,
		train_date=20180402
		)
	train_svd, test, val, data_svd = preprocessor_urm.fit_transform(df)

	# SVD
	model_svd = SVDRec()
	model_svd.fit(train)

	# Clean Trade Data
	preprocessor_reg = RegressorPreprocessor(
		test_date=20180416,
		val_date=20180409,
		train_date=20180402
		)
	X_train_lin, y_train_lin, test, val, X_lin, y_lin = preprocessor_reg.fit_transform(df)

	# Linear Regression
	model_reg = Regressor()
	model_reg.fit(X_train_lin, y_train_lin)

	###### CLASSIFIERS	

	# Clean Trade Data
	preprocessor_class = ClassifierPreprocessor(
		test_date=20180416,
		val_date=20180409,
		train_date=20180402
		)
	X_train_class, y_train_class, test, val, X_class, y_class = preprocessor_class.fit_transform(df)

	# CatBoost
	cat_model = CATBoost()
	cat_model.fit(X_train_class, y_train_class)

	# KNN
	knn_model = KNN()
	knn_model.fit(X_train_class, y_train_class)

	# Train Meta Classifier
	meta_class = train_meta_classifier(val, [model_reg, model_svd, 
		knn_model, cat_model])

	# Evaluate Meta Classifier
	score = evaluate_meta_classifier(test, meta_class, [model_reg, model_svd, 
		knn_model, cat_model])
	print("TEST SCORE: ", score)

	# Train the models on entire data
	model_svd.fit(data_svd)
	model_reg.fit(X_lin, y_lin)
	cat_model.fit(X_class, y_class)
	knn_model.fit(X_class, y_class)

	# Create the submission file
	create_submission_file(loader, preprocessor_reg, meta_class, 
		[model_reg, model_svd, knn_model, cat_model])
	return

main()