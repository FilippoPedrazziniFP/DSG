
class FakeGeneratorFilo(object):
	def __init__(self):
		super(FakeGeneratorFilo, self).__init__()

	def generate_train_set(self, df):
		raise NotImplementedError

	def generate_test_set(self, df):
		raise NotImplementedError
		