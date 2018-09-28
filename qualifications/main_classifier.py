import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd

from dsg.classification.classifier_preprocessing import ClassifierPreprocessor
from dsg.classification.classifier import CATBoost, LR
import dsg.util as util

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True, 
	help='If True train and validate the model locally.')
parser.add_argument('--sub', type=bool, default=True, 
	help='If True train the model on the entire data and creates the submission.')
args = parser.parse_args()

def main():

	# Fixing the seed
	np.random.seed(0)
	tf.set_random_seed(0)

	t = time.clock()

	# Clean Trade Data
	preprocessor = ClassifierPreprocessor()
	X_train, y_train, X_test, y_test, X_val, y_val, X, y, X_challenge, submission \
		= preprocessor.fit_transform()

	exit()

	print("TIME TO LOAD AND PREPROCESS THE MODEL: ", time.clock() - t)

	# Fit and Evaluate the model
	model = CATBoost()

	if args.train == True:

		t = time.clock()
		
		print(X_train.shape)
		print(y_train.shape)
		model.fit(X_train, y_train)
		
		print("TRAINED FINISHED, STARTING TEST..")

		# Evaluate the model
		score = model.evaluate(X_test, y_test)
		print("TEST SCORE: ", score)
		
		print("TIME TO FIT AND EVALUATE THE MODEL: ", time.clock() - t)

	if args.sub == True:

		t = time.clock()

		# Fit on the entire data 
		print(X.shape)
		print(y.shape)
		model.fit(X, y)
				
		# Create the submission file
		preds = model.predict(X_challenge)
		submission["CustomerInterest"] = preds
		submission.to_csv(util.SUBMISSION, index=False)

		print("TIME TO FIT THE ENTIRE DATA and CREATE SUBMISSION: ", time.clock() - t)

	return

main()