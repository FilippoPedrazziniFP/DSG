import numpy as np
import tensorflow as tf
import time

from dsg.models.model import RecurrentModel

def main():
	# Fixing the seed
	np.random.seed(0)
	tf.set_random_seed(0)
	
	start = time.clock()
	
	X_train = [
		[
			[0, 1, 0, 1, 0]
		], 
		[
			[0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0]
		],
		[
			[0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0]
		]
	]

	y_train = [0, 1, 0]
	preproc_time = time.clock() - start

	input("TIME TO LOAD AND PREPROCESS THE DATA: "+ str(preproc_time))

	start = time.clock()

	# fit and evaluate the model
	model = RecurrentModel()
	model.fit(X_train, y_train)

	fit_model = time.clock() - preproc_time
	input("TIME TO FIT THE MODEL: "+ str(fit_model))

	model.evaluate(X_test, y_test)

	return

main()