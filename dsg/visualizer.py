import matplotlib.pyplot as plt
import seaborn as sns

class Explorer(object):
	"""docstring for Eplorer"""
	def __init__(self):
		super(Explorer, self).__init__()

	@staticmethod
	def plot_array(array):
		sns.distplot(array)
		plt.show()
		return


		