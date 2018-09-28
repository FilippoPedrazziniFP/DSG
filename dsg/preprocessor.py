from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

class Preprocessor():
	
	def __init__(self):
		super(Preprocessor, self).__init__()
	
	def fit_transform(self, data):
		raise NotImplementedError

	def fit(self, data):
		raise NotImplementedError
	
	def transform(self, data):
		raise NotImplementedError
	
class FlatPreprocessor(Preprocessor):

	def __init__(self, train_samples=2000, 
		test_samples=300, val_samples=0):
		super(Preprocessor, self).__init__()
		self.train_samples = train_samples
		self.test_samples = test_samples
		self.val_samples = val_samples
            
    
    def pre_fit_transform(self, isTrain= True):
        sess = pd.read_csv("data/train_session.csv")
        if isTrain:    
            X = pd.read_csv("data/train_tracking.csv")
        else:
            X =  pd.read_csv("data/test_tracking.csv")
        
        tra_feat = gen_tra_features(X)
        dev_feat = gen_dev_features(X)
        train = sess.merge(tra_feat, on=['sid'], how='left').merge(dev_feat, on=['sid'], how='left')
        train.drop(["target"], inplace=True, axis=1)
        y = sess["target"]
        if isTrain:
            return train, y
        else:
            return train, None

	def fit_transform(self, data):
		# label split
		X = data[0]
		y = data[1]

		print(len(X))
		print(len(y))

		# normalize data
		X = self.standardize_features(X)

		# shuffle data
		X, y = shuffle(X, y)

		# train, test, validation
		X_train, y_train, X_test, y_test, X_val, y_val = \
		self.train_test_validation_split(X, y)

		return X_train, y_train, X_test, y_test, X_val, y_val, X, y
	
	def train_test_validation_split(self, features, labels):

		X_test = features[-self.test_samples:]
		y_test = labels[-self.test_samples:]

		X_val = features[-(self.test_samples+self.val_samples):-self.test_samples]
		y_val = labels[-(self.test_samples+self.val_samples):-self.test_samples]

		X_train = features[-(self.test_samples+self.val_samples+self.train_samples):-(self.test_samples+self.val_samples)]
		y_train = labels[-(self.test_samples+self.val_samples+self.train_samples):-(self.test_samples+self.val_samples)]

		return X_train, y_train, X_test, y_test, X_val, y_val
	
	def standardize_features(self, X_train):
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		return X_train

def get_type_feature(type_str):
	"""
	input : type field of df_train_tracking
	output : 2 one-hot-encoded vectors (page, event)

	"""
	page_type = ['PA', 'LP', 'LR', 'CAROUSEL', 'SHOW_CASE']
	event_type = ['ADD_TO_BASKET', 'PURCHASE_PRODUCT', 'PRODUCT']

	page_vec = [0] * len(page_type)
	event_vec = [0] * len(event_type)

	indeces_page = [i for i, elem in enumerate(page_type) if elem in type_str]
	indeces_event = [i for i, elem in enumerate(event_type) if elem in type_str]

	if indeces_page:
		page_vec[indeces_page[0]] = 1

	if indeces_event:
		event_vec[indeces_event[0]] = 1

	# return 2 hot encoded vectors
	return page_vec, event_vec


def process_string(s):
	s = s.replace("SEARCH", "LR")
	s = s.replace("LIST_PRODUCT", "LP")
	return s

def encode_train(df):

    """
    Dataframe grouped by id. Type feature is encoded in two vectors page_vec, event_vec
    Resulting dataframe saved in df_encoded.pkl
    """
    df_grouped = df.groupby('sid').apply(lambda x: x.sort_values(["duration"]))
    df_grouped['page_vec'] = df_grouped["type"].apply(lambda x: get_type_feature(process_string(x))[0])
    df_grouped['event_vec'] = df_grouped["type"].apply(lambda x: get_type_feature(process_string(x))[1])
    df_grouped.to_pickle("./df_encoded.pkl")

    return df_grouped


def columns_df_to_list(df):
    """
    Returns a dataframe sid, [list of actions features]
    """
    df["list"] = df.apply(lambda x: list(x[:]), axis=1)
    listed_final = df["list"]
    listed_final = listed_final.reset_index().groupby('sid')['list'].apply(list).reset_index()
    return listed_final

def matr_to_list(l, op = np.add):
    
    res_page = np.zeros(5)
    res_event = np.zeros(3)
    
    for oh_page, oh_event in l:
        res_page = op(oh_page, res_page)
        res_event = op(oh_event, res_event)
        
    return np.append(res_page, res_event)

def gen_tra_features(df):
    page_type = ['PA', 'LP', 'LR', 'CAROUSEL', 'SHOW_CASE']
    event_type = ['ADD_TO_BASKET', 'PURCHASE_PRODUCT', 'PRODUCT']
    
    tra_one_list = df.groupby('sid').agg({'type':lambda x: list(x)}).reset_index()
    tra_one_list['one_hot'] = tra_one_list["type"].apply(lambda x: [get_type_feature(s) for s in x])
    tra_one_list["feature"] = tra_one_list["one_hot"].apply(matr_to_list)
    
    tra_features = pd.DataFrame(tra_one_list["feature"].values.tolist(), columns=page_type+event_type)
    tra_features["sid"] = tra_one_list["sid"]
    return tra_features


def gen_dev_features(df):
    dummies = pd.get_dummies(df["device"])
    temp = pd.DataFrame()
    temp["sid"] = df["sid"]
    temp[["dev1", "dev2", "dev3"]] = dummies
    dev_features = temp.groupby("sid").max().reset_index()
    return dev_features


def transform_train_tracking(df):

    df = encode_train(df)
    df = columns_df_to_list(df)
    df.to_pickle("./df_transformed.pkl")

    return df

def get_dataset(transformed_df, label_df):

    merged_df = label_df.merge(transformed_df, on=['sid'], how='left')
    merged_df['label'] = merged_df["target"].apply(lambda x: 0 if x is False else 1)
    x = merged_df['list'].tolist()
    y = merged_df['label'].tolist()

    return x,y
