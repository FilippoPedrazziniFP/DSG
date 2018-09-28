from catboost import CatBoostClassifier
from dsg.models.model import Model
from sklearn.metrics import log_loss

class CatBoost(Model):
    
    def fit(self, X_train, y_train):
        # Fit the model
        self.model = CatBoostClassifier(verbose=False)
        self.model.fit(X_train, y_train)
        return

    def transform(self, X_test):
        predictions = self.model.predict_proba(X_test)[0]
        return predictions
    
    def evaluate(self, X_test, y_test):
        y_pred = self.transform(X_test)
        score = log_loss(y_test, y_pred)
        print("LOG LOSS : ", score)
        return