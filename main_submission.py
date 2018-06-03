import pandas as pd
import dsg.util as util

def main():
	df = pd.read_csv(util.SUBMISSION)
	print(df.describe())
	print(df.head())
main()