import pickle
from sklearn.ensemble import RandomForestClassifier


class QuinielaModel:

    def train(self, X_train, y_train):
        self.gb_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=15)
        self.gb_model.fit(X_train, y_train)

    def predict(self, X_predict, return_probabilities=False):
        if return_probabilities:
            probabilities = self.gb_model.predict_proba(X_predict)
            return probabilities
        else:
            predictions = self.gb_model.predict(X_predict)
            return predictions

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
