import numpy as np
from dsg.recommenders.recommender import Recommender
from surprise import SVD, SVDpp, KNNBasic, KNNBaseline, NMF, SlopeOne, CoClustering
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import Dataset
from sklearn.metrics import roc_auc_score
from surprise.model_selection import GridSearchCV

class CollaborativeFiltering(Recommender):
	def __init__(self, max_rating=5):
		super(CollaborativeFiltering, self).__init__()
		self.reader = Reader(rating_scale=(1, max_rating))

	def fit(self, df):
		data = Dataset.load_from_df(df[['CustomerIdx', 
			'IsinIdx', 'CustomerInterest']], self.reader)
		trainset = data.build_full_trainset()
		self.model.fit(trainset)
		return

	def cross_validation(self, df):
		data = Dataset.load_from_df(df[['CustomerIdx', 'IsinIdx', 
			'CustomerInterest']], self.reader)
		scores = cross_validate(self.model, data, measures=['RMSE', 'MAE'], 
			cv=5, verbose=True, n_jobs=-1)
		return scores

	def tune(self, df):
		raise NotImplementedError

	def predict(self, df):
		predictions = []
		for sample in df:
			pred = self.model.predict(sample[0], sample[1])
			predictions.append(pred[3])
		predictions = np.asarray(predictions)
		predictions = predictions/predictions.max()
		return predictions

	def evaluate(self, test):
		X_test, y_test = self.features_labels_split_df(test)
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, list(y_pred))
		return score

class SimpleKNN(CollaborativeFiltering):
	def __init__(self):
		super(SimpleKNN, self).__init__()
		self.model = KNNBasic()

class NMFAlgo(object):
	def __init__(self):
		super(NMFAlgo, self).__init__()
		self.model = NMF()

class Slope(object):
		def __init__(self):
			super(Slope, self).__init__()
			self.model = SlopeOne()

class Clustering(object):
	def __init__(self):
		super(Clustering, self).__init__()
		self.model = CoClustering()

class BaselineKNN(object):
	def __init__(self):
		super(BaselineKNN, self).__init__()
		self.model = KNNBaseline()

class SVDRec(CollaborativeFiltering):
	def __init__(self):
		super(SVDRec, self).__init__()
		self.model = SVD()

	def tune(self, df):
		data = Dataset.load_from_df(df[['CustomerIdx', 
			'IsinIdx', 'CustomerInterest']], self.reader)
		param_grid = {
			'n_epochs': [10, 20, 30], 
			'lr_all': [0.005, 0.0005], 
			'reg_all': [0.4, 0.3, 0.2, 0.1], 
			'n_factors': [100, 200, 300]}
		gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
		gs.fit(data)
		print("BEST RMSE: ", gs.best_score['rmse'])
		print("BEST PARAMETERS: ", gs.best_params['rmse'])
		return

class AsynchSVDRec(CollaborativeFiltering):
	def __init__(self):
		super(AsynchSVDRec, self).__init__()
		self.model = SVDpp()

	def tune(self, df):
		data = Dataset.load_from_df(df[['CustomerIdx', 
			'IsinIdx', 'CustomerInterest']], self.reader)
		param_grid = {
			'n_epochs': [10, 20, 30], 
			'lr_all': [0.005, 0.0005], 
			'reg_all': [0.4, 0.3, 0.2, 0.1], 
			'n_factors': [100, 200, 300]}
		gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
		gs.fit(data)
		print("BEST RMSE: ", gs.best_score['rmse'])
		print("BEST PARAMETERS: ", gs.best_params['rmse'])
		return
		
		
		
	
