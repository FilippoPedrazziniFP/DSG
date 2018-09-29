import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from sklearn.metrics import log_loss

class Model(object):
	def __init__(self):
		super(Model, self).__init__()
		self.model = None
	
	def fit(self, X_train, y_train):
		raise NotImplementedError
	
	def transform(self, X_test):
		raise NotImplementedError

class RecurrentModel(Model):
	def __init__(self, epochs=500, steps_per_epoch=10):
		super(Model, self).__init__()
		self.epochs = epochs
		self.steps_per_epoch = steps_per_epoch
		self.model = None
	
	def get_max_sequence_length(self, X_train):
		max_len = 0
		for sample in X_train:
			if len(sample) > max_len:
				max_len = len(sample)
		return max_len
	
	def build_model(self, features_len):
		model = Sequential()
		model.add(LSTM(32, input_shape=(None, features_len)))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(
			loss='binary_crossentropy',
			optimizer=Adam())
		return model
	
	def data_iterator(self, X, y, max_len):
		"""
			The method is defined as an iterator
			to generate the necessary data to
			fit the model.

			@args
				X : list[list[word] list[case], 
					list[numpy array(chars x embedding size)]]
				y : list[list[int]]
				max_len : int -> maximum len of sentences
					in the training examples.
		"""
		while True:
			for l in range(1, max_len):
				X_len = []
				y_len = []
				for features, label in zip(X, y):
					if len(features) == l:
						X_len.append(features)
						if l == 1:
							y_len.append([label])
						else:
							y_len.append(label)
				
				# Preprocess the Data to feed the model
				X_train, y_train = self.transform_into_model_data(X_len, y_len)

				# Check if there are examples with length == l
				if y_train.shape[0] == 0:
					continue
				else:
					yield X_train, y_train
	
	def transform_into_model_data(self, X, y=None):
		# Transform the data into trainable data (numpy arrays)
		X = np.asarray(X)
		if y is not None:
			y = np.asarray(y)
		return X, y
	
	def fit(self, X_train, y_train, X_val=None, y_val=None):
		# get max sequence length inside the training samples
		# self.max_sequence_length = self.get_max_sequence_length(X_train)
		self.max_sequence_length = 5

		print("MAX SEQUENCE LENGTH : "+str(self.max_sequence_length))
		
		# build the model architecture in case of None
		if self.model == None:
			print("FEATURES DIM : ", len(X_train[0][0]))
			self.model = self.build_model(len(X_train[0][0]))
				
		# Fit the model
		print("TRAINING MODEL..")
		if X_val is not None:
			# fit the model
			history = self.model.fit_generator(
				self.data_iterator(X_train, y_train, self.max_sequence_length),
				epochs=self.epochs,
				validation_data=self.data_iterator(X_val, y_val, self.max_sequence_length),
				validation_steps=self.steps_per_epoch,
				steps_per_epoch=self.steps_per_epoch,
				shuffle=True,
				verbose=1
				)
		else:
			history = self.model.fit_generator(
					self.data_iterator(X_train, y_train, self.max_sequence_length),
					epochs=self.epochs,
					steps_per_epoch=self.steps_per_epoch,
					shuffle=True,
					verbose=1
					)
		print("MODEL TRAINED ON {} SAMPLES. ".format(len(X_train)))
		return history
	
	def evaluate(self, X_test, y_test):
		y_pred = self.transform(X_test)
		score = log_loss(y_test, y_pred)
		print("LOG LOSS : ", score)
		return
	
	def predict(self, X):
		predictions = []
		for sample in tqdm(X):
			if len(sample) == 1:
				sample, _ = self.transform_into_model_data([[sample]])
			else:
				sample, _ = self.transform_into_model_data([sample])
			prediction = self.model.predict_proba(sample)
			prediction = np.squeeze(prediction)
			predictions.append(prediction)
		return predictions
	
	def transform(self, X):
		predictions = []
		for sample in X:
			if len(sample) == 1:
				sample, _ = self.transform_into_model_data([[sample]])
			else:
				sample, _ = self.transform_into_model_data([sample])
			prediction = self.model.predict_proba(sample)
			prediction = np.squeeze(prediction)
			predictions.append([1-prediction, prediction])
		return predictions
