from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
import tflearn.objectives.roc_auc_score as roc_loss
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
import logging

class LSTMModel(object):
	def __init__(self, epochs=2, batch_size=32, input_shape=(8, 14)):
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
	def binary_crossentropy_with_ranking(y_true, y_pred):
		""" Trying to combine ranking loss with numeric precision"""
		# first get the log loss like normal
		logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
		# next, build a rank loss
		# clip the probabilities to keep stability
		y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
		# translate into the raw scores before the logit
		y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))
		# determine what the maximum score for a zero outcome is
		y_pred_score_zerooutcome_max = K.max(y_pred_score * (y_true <1))
		# determine how much each score is above or below it
		rankloss = y_pred_score - y_pred_score_zerooutcome_max
		# only keep losses for positive outcomes
		rankloss = rankloss * y_true
		# only keep losses where the score is below the max
		rankloss = K.square(K.clip(rankloss, -100, 0))
		# average the loss for just the positive outcomes
		rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_true > 0) + 1)
		# return (rankloss + 1) * logloss - an alternative to try
		return rankloss + logloss
		
	@staticmethod
	def soft_AUC_backend(y_true, y_pred):
	    # Extract 1s
	    pos_pred_vr = y_pred[y_true.nonzero()]
	    # Extract zeroes
	    neg_pred_vr = y_pred[K.eq(y_true, 0).nonzero()]
	    # Broadcast the subtraction to give a matrix of differences  between pairs of observations.
	    pred_diffs_vr = pos_pred_vr.dimshuffle(0, 'x') - neg_pred_vr.dimshuffle('x', 0)
	    # Get signmoid of each pair.
	    stats = K.sigmoid(pred_diffs_vr * 2)
	    # Take average and reverse sign
	    return 1-K.mean(stats) 
	
	def build_model(self, parameters=None):		
		model = Sequential()
		model.add(GRU(32, return_sequences=False, 
			input_shape=self.input_shape))
		model.add(Dense(1, activation="sigmoid"))

		model.compile(
			loss=roc_loss, 
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