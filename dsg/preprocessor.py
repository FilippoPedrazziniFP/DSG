class Preprocessor():
    
    def __init__(self, seq_length=10):
		super(Preprocessor, self).__init__()
		self.seq_length = seq_length
    
    def fit(self):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError