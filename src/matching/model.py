"""Matching model interface."""


class MatchingModel:
    def fit(self, features, labels=None):
        raise NotImplementedError

    def predict_scores(self, features):
        raise NotImplementedError
