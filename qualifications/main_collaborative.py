import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd

from dsg.data_loader import DataLoader
from dsg.data_generation.data_generator import FakeGeneratorFilo
from dsg.recommenders.urm_preprocessing import URMPreprocessing
from dsg.recommenders.collaborative import SVDRec, AsynchSVDRec, SimpleKNN, NMFAlgo, Slope, BaselineKNN, Clustering
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

	# Clean Trade Data
	preprocessor = URMPreprocessing(
		test_date=20180415,
		val_date=20180408,
		train_date=20180401
		)
	train, test, val, data = preprocessor.fit_transform(df)

	print("TRAIN")
	print(train.head())
	print(train.describe())
	print("TEST")
	print(test.head())
	print(test.describe())

	preproc_time = time.clock() - start
	print("TIME TO LOAD AND PREPROCESS THE MODEL: ", preproc_time)

	# Fit and Evaluate the model
	model = SVDRec()
	model.fit(train)

	print("TRAINING FINISHED; STARTING VALIDATION..")

	print(val.describe())
	print(val.head())
	# Evaluate the model
	score = model.evaluate(val)
	print("TEST SCORE: ", score)

	# Evaluate the model on Claudio Test Set
	generator = FakeGeneratorFilo()
	test_claudio = generator.generate_test_claudio(df)
	print(test_claudio.describe())
	print(test_claudio.head())

	# Evaluate the model
	score = model.evaluate(test_claudio)
	print("TEST SCORE CLAUDIO: ", score)
	
	fit_model = time.clock() - preproc_time
	print("TIME TO FIT AND EVALUATE THE MODEL: ", fit_model)

	exit()

	# Fit on the entire data 
	model.fit(data)
	
	fit_data = time.clock() - fit_model
	print("TIME TO FIT THE ENTIRE DATA: ", fit_data)
	
	# Create the submission file
	create_submission_file(loader, preprocessor, model)

	submission_time = time.clock() - fit_model
	print("TIME TO PREDICT AND CREATE SUBMISSION FILE: ", submission_time)

	return

main()