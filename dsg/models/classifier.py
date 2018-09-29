import numpy as np
import lightgbm as lgb
from dsg.models.model import Model
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool, cv

class CatBoost(Model):
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Fit the model
        self.model = CatBoostClassifier(verbose=False)
        self.model.fit(X_train, y_train)
        return
    
    def tune_best(self, X_train, y_train, X_val, y_val):
        params = {
            'iterations': 500,
            'learning_rate': 0.001,
            'eval_metric': 'Logloss',
            'random_seed': 42,
            'logging_level': 'Silent',
            'use_best_model': False
        }
        train_pool = Pool(X_train, y_train)
        validate_pool = Pool(X_val, y_val)
        self.model = CatBoostClassifier(**params)
        self.model.fit(train_pool, eval_set=validate_pool)
        best_model_params = params.copy()
        best_model_params.update({
            'use_best_model': True
        })
        self.model = CatBoostClassifier(**best_model_params)
        self.model.fit(train_pool, eval_set=validate_pool, logging_level='Verbose')
        return
    
    def tune(self, X_train, y_train, X_val, y_val):
        self.model = CatBoostClassifier(
            custom_loss=['Logloss'],
            random_seed=42,
            logging_level='Silent'
        )
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            logging_level='Verbose',
            plot=True
        );
        cv_params = self.model.get_params()
        cv_params.update({
            'loss_function': 'Logloss'
        })
        cv_data = cv(
            Pool(X_train, y_train),
            cv_params,
            plot=True
        )
        print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
            np.max(cv_data['test-Logloss-mean']),
            cv_data['test-Logloss-std'][np.argmax(cv_data['test-Logloss-mean'])],
            np.argmax(cv_data['test-Logloss-mean'])
        ))
        print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Logloss-mean'])))

    def transform(self, X_test):
        predictions = self.model.predict_proba(X_test)
        return predictions
    
    def predict(self, X_test):
        predictions = self.model.predict_proba(X_test)[:,1]
        return predictions

    def evaluate(self, X_test, y_test):
        y_pred = self.transform(X_test)
        score = log_loss(y_test, y_pred)
        print("LOG LOSS : ", score)
        return

class LightGradientBoosting(object):
    
    def tune(self, X_train, y_train, X_val, y_val):
        self.model = lgb.LGBMClassifier(learning_rate=0.0001,  objective="binary", is_unbalance=True)
        self.model.fit(X_train, y_train)
        return

    def transform(self, X_test):
        predictions = self.model.predict_proba(X_test)
        return predictions
    
    def predict(self, X_test):
        predictions = self.model.predict_proba(X_test)[:,1]
        return predictions

    def evaluate(self, X_test, y_test):
        y_pred = self.transform(X_test)
        score = log_loss(y_test, y_pred)
        print("LOG LOSS : ", score)
        return