import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd

from dsg.data_loader import DataLoader
from dsg.trend_predictor.trend_preprocessing import TrendPreprocessor
from dsg.trend_predictor.trend_model import LSTMModel
import dsg.util as util

parser = argparse.ArgumentParser()

""" General Parameters """
args = parser.parse_args()

def main():

	# Fixing the seed
	np.random.seed(0)
	tf.set_random_seed(0)

	start = time.clock()

	# Load the Data
	loader = DataLoader()
	df = loader.load_trade_data()

	# Clean Trade Data
	preprocessor = TrendPreprocessor(
		from_date=20170420,
		test_samples=7,
		val_samples=14
		)
	X_train, y_train, X_test, y_test, X_val, y_val, last_sample = preprocessor.fit_transform(df)

	print("TRAIN")
	print(X_train[0:10])

	preproc_time = time.clock() - start
	print("TIME TO LOAD AND PREPROCESS THE MODEL: ", preproc_time)

	# Fit and Evaluate the model
	model = LSTMModel()
	model.fit(X_train, y_train, X_val, y_val)

	# Evaluate the model
	loss, score = model.evaluate(X_test, y_test)
	print("TEST SCORE: ", score)
	
	fit_model = time.clock() - preproc_time
	print("TIME TO FIT AND EVALUATE THE MODEL: ", fit_model)

	final_pred = model.predict(last_sample)
	final_pred = np.asarray(final_pred).sum()
	print("FINAL PREDICTION: ", final_pred)
	return

main()