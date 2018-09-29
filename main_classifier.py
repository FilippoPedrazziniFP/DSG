import numpy as np
import tensorflow as tf
import time
from collections import Counter

from dsg.loader import DataLoader, Util
from dsg.models.classifier import CatBoost, LightGradientBoosting
from dsg.preprocessor import FlatPreprocessor

def main():
	
    # Fixing the seed
    np.random.seed(0)
    tf.set_random_seed(0)

    start = time.clock()

    preprocessor = FlatPreprocessor()
    
    """try:
        X_train, y_train, X_test, y_test, X_val, y_val, X, y = \
        DataLoader.load_from_pickle(Util.AFTER_PREPROCESSING)
        print("FILE FOUND")
    except FileNotFoundError:"""
    print("FILE NOT FOUND, GENERATING THE TRAINIG DATA")
    X_train, y_train, X_test, y_test, X_val, y_val, X, y = preprocessor.fit_transform()
    DataLoader.save_into_pickle(Util.AFTER_PREPROCESSING, 
    [X_train, y_train, X_test, y_test, X_val, y_val, X, y])

    preproc_time = time.clock() - start
    input("TIME TO LOAD AND PREPROCESS THE DATA: "+ str(preproc_time))

    start = time.clock()

    # fit and evaluate the model
    model = CatBoost()
    """from imblearn.combine import SMOTEENN
    sm = SMOTEENN()
    print('dataset shape {}'.format(Counter(y_train)))
    X_train, y_train = sm.fit_sample(X_train, y_train)
    print('Resampled dataset shape {}'.format(Counter(y_train)))"""
    # model = LightGradientBoosting()
    # model.fit(X_train, y_train)
    # model.tune(X_train, y_train, X_val, y_val)

    fit_model = time.clock() - preproc_time
    input("TIME TO FIT THE MODEL: "+ str(fit_model))

    # evaluating performances
    # model.evaluate(X_test, y_test)

    # fit on entire data
    model.fit(X, y)
    print("FITTED ON ENTIRE DATA..")

    # generate submission file
    Util.generate_submission_file(model, preprocessor)

    return

main()