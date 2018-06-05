import numpy as np
from dsg.recommenders.recommender import Recommender
from surprise import SVD, SVDpp
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import Dataset
from sklearn.metrics import roc_auc_score
from surprise.model_selection import GridSearchCV

class MatrixFactorization(Recommender):
	def __init__(self):
		super(MatrixFactorization, self).__init__()
		self.reader = Reader(rating_scale=(1, 5))

	def fit(self, df):
		data = Dataset.load_from_df(df[['CustomerIdx', 'IsinIdx', 'CustomerInterest']], self.reader)
		trainset = data.build_full_trainset()
		self.model.fit(trainset)
		return

	def cross_validation(self, df):
		data = Dataset.load_from_df(df[['CustomerIdx', 'IsinIdx', 'CustomerInterest']], self.reader)
		scores = cross_validate(self.model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
		return scores

	def predict(self, df):
		predictions = []
		for sample in df:
			pred = self.model.predict(sample[0], sample[1])
			predictions.append(pred[3])
		predictions = np.asarray(predictions)
		predictions = predictions/predictions.max()
		return predictions

	def tune(self, df):
		data = Dataset.load_from_df(df[['CustomerIdx', 'IsinIdx', 'CustomerInterest']], self.reader)
		param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.4, 0.6]}
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(data)
        print(gs.best_score['rmse'])
        print(gs.best_params['rmse'])
        return

	def tune(self, df):
		raise NotImplementedError

	def evaluate(self, test):
		X_test, y_test = self.features_labels_split_df(test)
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, list(y_pred))
		return


class SVDRec(MatrixFactorization):
	def __init__(self, latent_factors=100):
		super(SVDRec, self).__init__()
		self.model = SVD(n_factors=latent_factors)

class AsynchSVDRec(MatrixFactorization):
	def __init__(self, latent_factors=100):
		super(AsynchSVDRec, self).__init__()
		self.model = SVDpp(n_factors=latent_factors)
		
		
		
	
