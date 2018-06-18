from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
import logging

class LSTMModel(object):
	def __init__(self, epochs=10, batch_size=32, input_shape=(8, 14)):
		super(LSTMModel, self).__init__()
		self.epochs = epochs
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.model = self.build_model()

	def define_callbacks(self, X_val, y_val):
		ival = IntervalEvaluation(validation_data=(X_val, y_val), interval=1)
		chk = ModelCheckpoint("./dsg/recurrent/weights/lstm.hdf5", 
			monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		return [ival, chk]

	@staticmethod
	def f1(y_true, y_pred):
	    def recall(y_true, y_pred):
	        """Recall metric.

	        Only computes a batch-wise average of recall.

	        Computes the recall, a metric for multi-label classification of
	        how many relevant items are selected.
	        """
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	        recall = true_positives / (possible_positives + K.epsilon())
	        return recall

	    def precision(y_true, y_pred):
	        """Precision metric.

	        Only computes a batch-wise average of precision.

	        Computes the precision, a metric for multi-label classification of
	        how many selected items are relevant.
	        """
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	        precision = true_positives / (predicted_positives + K.epsilon())
	        return precision
	    precision = precision(y_true, y_pred)
	    recall = recall(y_true, y_pred)
	    return 2*((precision*recall)/(precision+recall+K.epsilon()))
	
	def build_model(self, parameters=None):		
		model = Sequential()
		model.add(GRU(128, return_sequences=False, 
			input_shape=self.input_shape))
		model.add(Dense(1, activation="sigmoid"))

		model.compile(
			loss="binary_crossentropy", 
			optimizer=Adam(), 
			metrics=["accuracy"]
			)
		return model

	def fit(self, X_train, y_train, X_val, y_val):

		self.define_cb = self.define_callbacks(X_val, y_val)

		history = self.model.fit(
			X_train, 
			y_train, 
			epochs=self.epochs,
			batch_size=self.batch_size,
			validation_data=(X_val, y_val),
			shuffle=True, 
			callbacks=self.define_cb
			)
		return history

	def restore(self):
		try:
			self.model.load_weights("./dsg/recurrent/weights/lstm.hdf5")
		except:
			pass
		return

	def predict(self, X):
		y_pred = self.model.predict_proba(X)
		return y_pred

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, y_pred)
		return score

class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            print("ROC Score: ", score)