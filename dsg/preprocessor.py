class Preprocessor():
    
    def __init__(self, seq_length=10):
		super(Preprocessor, self).__init__()
		self.seq_length = seq_length
    
    def fit(self):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError


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