import numpy as np

from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

class Model(object):
	def __init__(self):
		super(Model, self).__init__()
		self.model = None
	
	def fit(self, X_train, y_train):
		raise NotImplementedError
	
	def transform(self, X_test):
		raise NotImplementedError

class RecurrentModel(Model):
	def __init__(self, epochs=10, steps_per_epoch=1):
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
		model.add(LSTM(256, input_shape=(None, features_len)))
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
	
	def transform_into_model_data(self, X, y):
		# Transform the data into trainable data (numpy arrays)
		X = np.asarray(X)
		y = np.asarray(y)
		"""try:
			# Reshape the label to feed the model (Categorical crossentropy)
			# y = to_categorical(y, num_classes=self.label_dim)
			y = to_categorical(y, num_classes=self.label_dim)
			# y = np.reshape(y, (y.shape[0], y.shape[1], 1))
			# print("Sparse cross entropy")
		except IndexError as e:
			print("TRAINING SAMPLES MUST HAVE ALL THE ENITYT TYPES.")
			print("PROVIDE SAMPLES FOR EACH ENTITY IN ORDER TO 
			TRAIN THE DEEP LEARNING MODEL.")"""
		return X, y
	
	def fit(self, X_train, y_train, X_val=None, y_val=None):

		# get max sequence length inside the training samples
		self.max_sequence_length = self.get_max_sequence_length(X_train)
		
		# build the model architecture in case of None
		if self.model == None:
			print("FEATURES DIM : ", len(X_train[0][0]))
			self.model = self.build_model(len(X_train[0][0]))
		
		"""
			HERE PREPROCESSING
		"""
		
		# Fit the model
		print("TRAINING MODEL..")
		if X_val is not None:
			# preprocess

			"""
				HERE PREPROCESSING
			"""

			# fit the model
			history = self.model.fit_generator(
				self.data_iterator(X_train, y_train, self.max_sequence_length),
				epochs=self.epochs,
				validation_data=self.data_iterator(X_val, y_val, max_sequence_length),
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
		losses = self.model.evaluate_generator(
			self.data_iterator(X, y, self.max_sequence_length),
			workers=16,
			use_multiprocessing=True)
		print(losses)
		return losses
	
	def transform(self, X):
		prediction = self.model.predict_proba(X)
		return prediction
