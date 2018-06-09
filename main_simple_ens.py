import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd

from dsg.data_loader import DataLoader
from dsg.recommenders.urm_preprocessing import URMPreprocessing
from dsg.recommenders.collaborative import SVDRec, AsynchSVDRec
from dsg.regressor.regressor_preprocessing import RegressorPreprocessor
from dsg.classification.classifier_preprocessing import ClassifierPreprocessor
from dsg.classification.classifier import CATBoost, KNN
from dsg.regressor.regressor import Regressor
import dsg.util as util

parser = argparse.ArgumentParser()

""" General Parameters """
args = parser.parse_args()

def create_submission_file(loader, preprocessor, models):
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
	train, test, val, data = preprocessor_urm.fit_transform(df)

	# SVD
	model_svd = SVDRec()
	model_svd.fit(data)

	# Clean Trade Data
	preprocessor_reg = RegressorPreprocessor(
		test_date=20180416,
		val_date=20180409,
		train_date=20180402
		)
	X_train, y_train, test, val, X, y = preprocessor_reg.fit_transform(df)

	# Linear Regression
	model_reg = Regressor()
	model_reg.fit(X, y)

	###### CLASSIFIERS	

	# Clean Trade Data
	preprocessor_class = ClassifierPreprocessor(
		test_date=20180416,
		val_date=20180409,
		train_date=20180402
		)
	X_train, y_train, test, val, X, y = preprocessor_class.fit_transform(df)

	# CatBoost
	cat_model = CATBoost()
	cat_model.fit(X, y)

	# KNN
	knn_model = KNN()
	knn_model.fit(X, y)

	# Create the submission file
	create_submission_file(loader, preprocessor_reg, [model_reg, model_svd, knn_model, cat_model])
	return

main()