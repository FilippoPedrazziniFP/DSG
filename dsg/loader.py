import pandas as pd
import pickle

class Util():
	TRAIN_SESSION = "./data/train_session.csv"
	TRAIN_TRACKING = "./data/train_tracking.csv"
	RANDOM_SUBMISSION = "./data/random_submission.csv"
	TEST_TRACKING = "./data/test_tracking.csv"
	PRODUCT_CATEGORY = "./data/productid_category.csv"
	SUBMISSION = "./data/submission.csv"

	FEATURES = "./data/Xtrain.p"
	LABELS = "./data/Ytrain.p"

	TRAIN_LSTM = "./data/train_lstm.pickle"
	TEST_LSTM = "./data/test_lstm.pickle"

	AFTER_PREPROCESSING = "./data/after_preprocessing.pkl"
	BEFORE_PREPROCESSING = "./data/before_preprocessing.pkl"
	AFTER_PREPROCESSING_TEST = "./data/after_preprocessing_test.pkl"
	BEFORE_PREPROCESSING_TEST = "./data/before_preprocessing_test.pkl"

	AFTER_PREPROCESSING_LSTM = "./data/after_preprocessing_lstm.pkl"
	BEFORE_PREPROCESSING_LSTM = "./data/before_preprocessing_lstm.pkl"
	AFTER_PREPROCESSING_TEST_LSTM = "./data/after_preprocessing_test_lstm.pkl"
	BEFORE_PREPROCESSING_TEST_LSTM = "./data/before_preprocessing_test_lstm.pkl"

	@staticmethod
	def generate_submission_file(model, preprocessor):
		X_test = preprocessor.transform()
		y_pred = model.predict(X_test)
		input(y_pred)
		submission = pd.read_csv(Util.RANDOM_SUBMISSION)
		submission["target"] = y_pred
		submission.to_csv(Util.SUBMISSION, index=False)
		return

class DataLoader(object):	
	
	@staticmethod
	def load_created_data():
		with open(Util.FEATURES, "rb") as f:
			X_train = pickle.load(f)
		with open(Util.LABELS, "rb") as f:
			y_train = pickle.load(f)
		print("DATA LOADED")
		return [X_train, y_train]
	
	@staticmethod
	def save_into_pickle(file_path, file):
		with open(file_path, 'wb') as f:
			pickle.dump(file, f)
		return
	
	@staticmethod
	def load_from_pickle(file_path):
		with open(file_path, "rb") as f:
			file = pickle.load(f)
		return file

	
