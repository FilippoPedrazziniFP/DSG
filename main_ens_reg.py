import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd

from dsg.data_loader import DataLoader
from dsg.recommenders.urm_preprocessing import URMPreprocessing
from dsg.recommenders.collaborative import SVDRec, AsynchSVDRec
from dsg.regressor.regressor_preprocessing import RegressorPreprocessor
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

	# Clean Trade Data
	preprocessor_urm = URMPreprocessing(
		from_date=20180101,
		test_date=20180412,
		val_date=20180405,
		train_date=20180328
		)
	train, test, val, data = preprocessor_urm.fit_transform(df)

	# Asynch SVD
	# model_asynch_svd = AsynchSVDRec()
	# model_asynch_svd.fit(data)

	# SVD
	model_svd = SVDRec()
	model_svd.fit(data)

	# Clean Trade Data
	preprocessor_reg = RegressorPreprocessor(
		from_date=20180101,
		test_date=20180412,
		val_date=20180405,
		train_date=20180328
		)
	X_train, y_train, X, y = preprocessor_reg.fit_transform(df)

	# Fit and Evaluate the model
	model_reg = Regressor()
	model_reg.fit(X, y)
		
	# Create the submission file
	create_submission_file(loader, preprocessor_reg, [model_reg, model_svd])

	submission_time = time.clock() - fit_model
	print("TIME TO PREDICT AND CREATE SUBMISSION FILE: ", submission_time)

	return

main()