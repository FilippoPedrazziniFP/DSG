import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
from itertools import islice
import pickle
import operator
from itertools import chain
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, CatBoostRegressor

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False, 
	help='If True train and validate the model locally.')
parser.add_argument('--sub', type=bool, default=True, 
	help='If True train the model on the entire data and creates the submission.')
""" General Parameters """
args = parser.parse_args()

SAMPLE_SUBMISSION = "./sample_submission.csv"
SUBMISSION = "./submission.csv"

class Model(object):
	def __init__(self):
		super(Model, self).__init__()

	def load_data(self):
		"""
			The method drops the useless columns from
			the DataFrame and splits the data into train 
			test set based on the data

			@args
				df : DataFrame

			@return
				X_train, y_train, X_test, y_test : numpy array
		"""

		X_train = pickle.load(open("X_train.pkl", "rb"))
		print(X_train.shape)
		
		y_train = pickle.load(open("y_train.pkl", "rb"))
		print(y_train.shape)
		
		X_test = pickle.load(open("X_test.pkl", "rb"))
		print(X_test.shape)
		
		y_test = pickle.load(open("y_test.pkl", "rb"))
		print(y_test.shape)
		
		X_val = pickle.load(open("X_val.pkl", "rb"))
		print(X_val.shape)
		
		y_val = pickle.load(open("y_val.pkl", "rb"))
		print(y_val.shape)
		
		X = pickle.load(open("X.pkl", "rb"))
		print(X.shape)
		
		y = pickle.load(open("y.pkl", "rb"))
		print(y.shape)

		X_challenge = pickle.load(open("X_challenge.pkl", "rb"))
		print(X_challenge.shape)
		
		# Load Submission file
		submission = pd.read_csv(SAMPLE_SUBMISSION)

		return X_train, y_train, X_test, y_test, X_val, y_val, \
			X, y, X_challenge, submission 
		
class Classifier(Model):
	def __init__(self):
		super(Classifier, self).__init__()

	def fit(self, X_train, y_train):
		# Train Classifier
		self.classifier = self.train_classifier(X_train, y_train)
		return

	def train_classifier(self, X_train, y_train):
		"""
			Simple classifier to put a weight 
			to the frequency feature.
		"""
		self.scaler = StandardScaler()
		self.scaler.fit(X_train)
		X_train = self.scaler.transform(X_train)

		# Fit the model
		model = CatBoostClassifier(verbose=False)
		model.fit(X_train, y_train)
		return model

	def predict(self, X):
		X = self.scaler.transform(X)
		predictions = self.classifier.predict_proba(X)[:,1]
		return predictions

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, y_pred, average="samples")
		return score

class Regressor(Model):
	def __init__(self):
		super(Regressor, self).__init__()

	def fit(self, X_train, y_train):
		# Train Classifier
		self.classifier = self.train_regressor(X_train, y_train)
		return

	def train_regressor(self, X_train, y_train):
		"""
			Simple classifier to put a weight 
			to the frequency feature.
		"""
		self.scaler = StandardScaler()
		self.scaler.fit(X_train)
		X_train = self.scaler.transform(X_train)

		# Fit the model
		model = CatBoostRegressor(verbose=False)
		model.fit(X_train, y_train)
		return model

	def predict(self, X):
		X = self.scaler.transform(X)
		predictions = self.classifier.predict(X)
		predictions = predictions/predictions.max()
		print(predictions.max())
		return predictions

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, y_pred)
		return score

def main():

    np.random.seed(0)
    tf.set_random_seed(0)
    
    model = Regressor()
    X_train, y_train, X_test, y_test, X_val, y_val, X, y, X_challenge, submission \
		= model.load_data()
    
    #move new features of X at the end
    new_features_X = X[:,4:7]
    new_features_X = np.append(new_features_X, X[:,22:25], axis=1)
    new_features_X = np.append(new_features_X, X[:,51:54], axis=1)
    
    X = np.delete(X, [4,5,6,22,23,24,51,52,53],1)
    X = np.append(X, new_features_X, axis=1)
    
    #move new features of X_challenge at the end
    new_features_X_ch = X_challenge[:,4:7]
    new_features_X_ch = np.append(new_features_X_ch, X_challenge[:,22:25], axis=1)
    new_features_X_ch = np.append(new_features_X_ch, X_challenge[:,51:54], axis=1)
    
    X_challenge = np.delete(X_challenge, [4,5,6,22,23,24,51,52,53],1)
    X_challenge = np.append(X_challenge, new_features_X_ch, axis=1)
    
    #outliers
    up_bound_dict = {}
    up_bound_dict[0]=8500
    up_bound_dict[2]=2000
    up_bound_dict[3]=600
    up_bound_dict[4]=1100
    up_bound_dict[5]=4000
    up_bound_dict[6]=300
    up_bound_dict[7]=600
    up_bound_dict[8]=1000
    up_bound_dict[9]=6000
    up_bound_dict[10]=330
    up_bound_dict[11]=600
    up_bound_dict[12]=1000
    up_bound_dict[13]=6000
    up_bound_dict[14]=1200
    up_bound_dict[16]=600
    up_bound_dict[17]=100
    up_bound_dict[18]=300
    up_bound_dict[19]=750
    up_bound_dict[20]=100
    up_bound_dict[21]=110
    up_bound_dict[22]=250
    #up_bound_dict[23]=300
    up_bound_dict[24]=75
    up_bound_dict[25]=150
    up_bound_dict[26]=210
    up_bound_dict[27]=600
    up_bound_dict[28]=80000
    up_bound_dict[29]=80000
    up_bound_dict[30]=400000
    up_bound_dict[31]=5000000
    up_bound_dict[32]=20000
    up_bound_dict[33]=20000
    up_bound_dict[34]=10000
    up_bound_dict[35]=20000
    #up_bound_dict[36]=180
    #low_bound_dict[36]=35
    #up_bound_dict[37]=180
    #low_bound_dict[37]=35
    #up_bound_dict[38]=180
    #low_bound_dict[38]=35
    #up_bound_dict[39]=180
    #low_bound_dict[39]=35
    up_bound_dict[40]=260
    up_bound_dict[42]=125
    up_bound_dict[43]=30
    up_bound_dict[44]=55
    up_bound_dict[45]=200
    up_bound_dict[46]=15
    up_bound_dict[47]=25
    up_bound_dict[48]=60
    up_bound_dict[49]=130
    up_bound_dict[50]=20
    up_bound_dict[51]=30
    up_bound_dict[52]=60
    up_bound_dict[53]=120
    up_bound_dict[57]=300
    up_bound_dict[58]=500
    up_bound_dict[59]=800
    up_bound_dict[60]=80
    up_bound_dict[61]=150
    up_bound_dict[62]=220
    up_bound_dict[63]=10
    up_bound_dict[64]=30
    up_bound_dict[65]=50
    
    #drop outliers
#    for key, value in up_bound_dict.items():
#        mask=X[:, key] <= value
#        X = X[mask]
#        y= y[mask]
    
    #clip ouliers
    for key, value in up_bound_dict.items():
        X[:,key] = np.clip(X[:,key], -9999999, value)
        X_challenge[:,key] = np.clip(X_challenge[:,key], -9999999, value)
        
    
    #drop columns
#    X=np.delete(X, [28,29,30,31,32,33,34,35,23,27,55,56,14], 1)
#    X_challenge = np.delete(X_challenge, [28,29,30,31,32,33,34,35,23,27,55,56,14], 1)



    if args.train == True:
        t = time.clock()
		
        print(X_train.shape)
        print(y_train.shape)
        model.fit(X_train, y_train)
		
        print("TRAINED FINISHED, STARTING TEST..")

        # Evaluate the model
        score = model.evaluate(X_test, y_test)
        print("TEST SCORE: ", score)
		
        print("TIME TO FIT AND EVALUATE THE MODEL: ", time.clock() - t)



    if args.sub == True:
        t = time.clock()

        # Fit on the entire data 
        print(X.shape)
        print(y.shape)
        model.fit(X, y)
				
        # Create the submission file
        preds = model.predict(X_challenge)
        submission["CustomerInterest"] = preds
        submission.to_csv(SUBMISSION, index=False)
        print("TIME TO FIT THE ENTIRE DATA and CREATE SUBMISSION: ", time.clock() - t)

    return

main()