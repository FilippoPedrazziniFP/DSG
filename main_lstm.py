import numpy as np
import tensorflow as tf
import time

from dsg.models.model import RecurrentModel
from dsg.preprocessor import SequentialPreprocessor

def main():
	
	# Fixing the seed
	np.random.seed(0)
	tf.set_random_seed(0)
	
	start = time.clock()

	preprocessor = SequentialPreprocessor()

	print("FILE NOT FOUND, GENERATING THE TRAINIG DATA")
    X_train, y_train, X_test, y_test, X_val, y_val, X, y = preprocessor.fit_transform(train=True)
    DataLoader.save_into_pickle(Util.AFTER_PREPROCESSING, 
    [X_train, y_train, X_test, y_test, X_val, y_val, X, y])

	y_train = [0, 1, 0]
	preproc_time = time.clock() - start

	input("TIME TO LOAD AND PREPROCESS THE DATA: "+ str(preproc_time))

	start = time.clock()

	# fit and evaluate the model
	model = RecurrentModel()
	model.fit(X_train, y_train, X_val, y_val)

	fit_model = time.clock() - preproc_time
	input("TIME TO FIT THE MODEL: "+ str(fit_model))

	model.evaluate(X_test, y_test)

	return

main()