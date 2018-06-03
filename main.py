import argparse
import numpy as np
import tensorflow as tf
import time

from dsg.data_loader import DataLoader
from dsg.baseline.baseline_preprocessing import BaselinePreprocessor
from dsg.baseline.baseline_model import Baseline
import dsg.util as util

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--train_samples', type=int, default=300, 
	help='number of training examples.')
parser.add_argument('--test_samples', type=int, default=100, 
	help='number of test examples.')
parser.add_argument('--val_samples', type=int, default=100, 
	help='number of validation examples.')
args = parser.parse_args()

def create_submission_file(loader, preprocessor, model):
	test = loader.load_challenge_data()
	X = preprocessor.test_transform(test)
	preds = model.predict_for_submission(X)
	submission = loader.load_submission_file()
	submission["CustomerInterest"] = preds
	pd.to_csv(submission, util.SUBMISSION)
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
	preprocessor = BaselinePreprocessor(
		from_date=20180101,
		train_samples=args.train_samples,
		test_samples=args.test_samples,
		val_samples=args.val_samples
		)
	X_train, y_train, X_test, y_test, X_val, y_val, X, y = preprocessor.fit_transform(df)

	preproc_time = time.clock() - start
	print("TIME TO LOAD AND PREPROCESS THE MODEL: ", preproc_time)

	# Fit and Evaluate the model
	model = Baseline()
	model.fit(X_train, y_train)
	
	score = model.evaluate(X_test, y_test)
	print("TEST ERROR: ", score)

	fit_model = time.clock() - preproc_time
	print("TIME TO FIT AND EVALUATE THE MODEL: ", fit_model)

	# Fit on the entire data 
	model.fit(X, y)
	model.save_dictionary()

	fit_data = time.clock() - fit_model
	print("TIME TO FIT THE ENTIRE DATA: ", fit_data)
	
	# Create the submission file
	create_submission_file(loader, preprocessor, model)

	submission_time = time.clock() - fit_model
	print("TIME TO PREDICT AND CREATE SUBMISSION FILE: ", submission_time)

	return

main()