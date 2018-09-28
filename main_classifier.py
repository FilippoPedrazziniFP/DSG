import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd

from dsg.data_loader import DataLoader
from dsg.classifier.classifier_preprocessing import ClassifierPreprocessor
from dsg.classifier.classifier import Classifier
import dsg.util as util

parser = argparse.ArgumentParser()

""" General Parameters """
args = parser.parse_args()

def create_submission_file(loader, preprocessor, model):
	test = loader.load_challenge_data()
	print(test.head())
	X = preprocessor.test_transform(test)
	print(X[0:5])
	preds = model.predict(X)
	submission = loader.load_submission_file()
	submission["CustomerInterest"] = preds
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

	FROM_DATE = 20170420

	# Clean Trade Data
	preprocessor = ClassifierPreprocessor(
		from_date=20180101,
		test_date=20180412,
		val_date=20180405,
		train_date=20180328
		)
	X_train, y_train, y_test, y_val, X, y = preprocessor.fit_transform(df)

	print("TRAIN")
	print(X_train.head())
	print(y_train.head())

	preproc_time = time.clock() - start
	print("TIME TO LOAD AND PREPROCESS THE MODEL: ", preproc_time)

	# Fit and Evaluate the model
	model = Classifier()
	model.fit(X_train, y_train)

	# Evaluate the model
	score = model.evaluate(y_test)
	print("TEST SCORE: ", score)
	
	fit_model = time.clock() - preproc_time
	print("TIME TO FIT AND EVALUATE THE MODEL: ", fit_model)

	exit()

	# Fit on the entire data 
	model.fit(X, y)
	
	fit_data = time.clock() - fit_model
	print("TIME TO FIT THE ENTIRE DATA: ", fit_data)
	
	# Create the submission file
	create_submission_file(loader, preprocessor, model)

	submission_time = time.clock() - fit_model
	print("TIME TO PREDICT AND CREATE SUBMISSION FILE: ", submission_time)

	return

main()