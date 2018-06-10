from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from sklearn.metrics import mean_absolute_error

class LSTMModel(object):
	def __init__(self, epochs=100, batch_size=32, input_shape=(30, 1)):
		super(LSTMModel, self).__init__()
		self.epochs = epochs
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.model = self.build_model()

	@staticmethod
	def root_mean_squared_error(y_true, y_pred):
		return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
	
	def build_model(self, parameters=None):		
		model = Sequential()
		model.add(GRU(64, return_sequences=False, 
			input_shape=self.input_shape))
		model.add(Dense(1))

		model.compile(
			loss=LSTMModel.root_mean_squared_error, 
			optimizer=Adam(), 
			metrics=['mae']
			)
		return model

	def fit(self, X_train, y_train, X_val, y_val):
		history = self.model.fit(
			X_train, 
			y_train, 
			epochs=self.epochs,
			batch_size=self.batch_size,
			validation_data=(X_val, y_val),
			shuffle=True
			)
		return history

	def predict(self, X):
		y_pred = self.model.predict(X)
		return y_pred

	def evaluate(self, X_test, y_test):
		loss, score = self.model.evaluate(X_test, y_test)
		return loss, score