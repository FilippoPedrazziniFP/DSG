import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd

from dsg.data_loader import DataLoader
from dsg.recurrent.lstm_model import LSTMModel
from dsg.recurrent.lstm_preprocessing import SequencePreprocessor
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

	# Load the Data
	loader = DataLoader()
	df = loader.load_trade_data()

	# Clean Trade Data
	preprocessor = SequencePreprocessor()
	X_train, y_train, X_test, y_test, X_val, y_val, X, y, X_challenge, submission \
		= preprocessor.fit_transform()

	print("TIME TO LOAD AND PREPROCESS THE MODEL: ", time.clock() - t)

	# Fit and Evaluate the model
	model = LSTMModel()

	if args.train == True:

		t = time.clock()
		
		print(X_train.shape)
		print(y_train.shape)
		model.fit(X_train, y_train, X_val, y_val)
		
		print("TRAINED FINISHED, STARTING TEST..")

		# Evaluate the model
		score = model.evaluate(X_test[0:200], y_test[0:200])
		print("TEST SCORE: ", score)
		
		print("TIME TO FIT AND EVALUATE THE MODEL: ", time.clock() - t)

	if args.sub == True:

		t = time.clock()

		# Restore model
		model.restore()
				
		# Create the submission file
		preds = model.predict(X_challenge)
		submission["CustomerInterest"] = preds
		submission.to_csv(util.SUBMISSION, index=False)

		print("TIME TO FIT THE ENTIRE DATA and CREATE SUBMISSION: ", time.clock() - t)

	return

main()