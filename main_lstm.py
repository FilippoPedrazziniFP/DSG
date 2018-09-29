import numpy as np
import tensorflow as tf
import time

from dsg.models.model import RecurrentModel
from dsg.models.classifier import CatBoost
from dsg.seq_preprocessor import SeqPreprocessor
from dsg.loader import DataLoader, Util

def main():
	
	# Fixing the seed
	np.random.seed(0)
	tf.set_random_seed(0)
	
	start = time.clock()

	preprocessor = SeqPreprocessor()

	print("FILE NOT FOUND, GENERATING THE TRAINIG DATA")
	X_train, y_train, X_test, y_test, X_val, y_val, X, y = preprocessor.fit_transform()
	DataLoader.save_into_pickle(Util.AFTER_PREPROCESSING_LSTM, 
	[X_train, y_train, X_test, y_test, X_val, y_val, X, y])

	preproc_time = time.clock() - start

	print("TIME TO LOAD AND PREPROCESS THE DATA: "+ str(preproc_time))

	start = time.clock()

	# fit and evaluate the model
	model = RecurrentModel()
	# model.tune(X_train, y_train, X_val, y_val)

	fit_model = time.clock() - preproc_time
	print("TIME TO FIT THE MODEL: "+ str(fit_model))

	# evaluate the model
	# model.evaluate(X_test, y_test)

	# model fit on the entire data
	model.fit(X, y)
	print("FITTED ON ENTIRE DATA..")

	# generate submission file
	Util.generate_submission_file(model, preprocessor)

	return

main()