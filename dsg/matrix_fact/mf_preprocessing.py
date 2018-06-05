import dsg.util as util
from dsg.classifier.data_generator_class import DataGenerator

class MatrixFactorizationPreprocessor(object):
	def __init__(self):
		super(MatrixFactorizationPreprocessor, self).__init__()

	def fit(self, df):
		return

	def test_transform(self, df):
		df = df.drop(["PredictionIdx", "DateKey", "BuySell"], axis=1)
		X = df.values
		return X

	def transform(self, df):
		return

	def filter_data(self, df, date):
		df = df[df["TradeDateKey"] >= date]
		return df

	def fit_transform(self, df):
		"""
			The method drops the useless columns from
			the DataFrame and splits the data into train 
			test set based on the data

			@args
				df : DataFrame

			@return
				X_train, y_train, X_test, y_test : numpy array
		"""
		# Delete Holding Values
		df = df[df["TradeStatus"] != "Holding"]

		# Drop Useless Columns
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)

		# Filter Data 
		df = self.filter_data(df, self.from_date)

		# Train, test, val split
		data_generator = DataGenerator()
		X_train, y_train = data_generator.generate_train_set_regression(df, self.train_date, self.val_date)
		y_test = data_generator.generate_test_set(df, self.test_date)
		y_val = data_generator.generate_test_set(df, self.val_date, self.test_date)

		# Entire Train
		X, y = data_generator.generate_train_set_regression(df, self.test_date)

		
		

		return X_train, y_train, y_test, y_val, X, y

	def create_ur_matrix(self, df):
		"""
			The method creates a user rating matrix starting from
			a Data Frame object.
		"""





		return

	def df_to_csr(df, nrows, ncols, is_binary=False, user_key='CustomerIdx', item_key='IsinIdx', rating_key='CustomerInterest'):
	    """
	    Convert a pandas DataFrame to a scipy.sparse.csr_matrix
	    """
	    rows = df[user_key].values
	    columns = df[item_key].values
	    ratings = df[rating_key].values if not is_binary else np.ones(df.shape[0])
	    # use floats by default
	    ratings = ratings.astype(np.float32)
	    shape = (nrows, ncols)
	    # using the 4th constructor of csr_matrix
	    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
	    return sps.csr_matrix((ratings, (rows, columns)), shape=shape)

	def train_test_validation_split(self, features):
		raise NotImplementedError