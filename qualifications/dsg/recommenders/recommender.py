
class Recommender(object):
	def __init__(self):
		super(Recommender, self).__init__()

	def fit(self, X):
		raise NotImplementedError

	def predict(self, customer_id, bond_id):
		raise NotImplementedError

	def features_labels_split_df(self, test_df):
		labels = test_df["CustomerInterest"]
		features = test_df.drop(["CustomerInterest"], axis=1)
		return features.values, labels.values
		