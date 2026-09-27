"""Matching models (Member 1).

LGBMMatcher - LightGBM binary classifier on pair features (MIT licence,
a few MB, far below the 8B-parameter limit). RuleBasedMatcher is the Phase-1
baseline kept for src/pipeline_baseline.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


class LGBMMatcher:
    def __init__(self, params: dict | None = None, num_boost_round=600, early_stopping_rounds=40):
        self.params = dict(params or {"objective": "binary", "verbose": -1})
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None
        self.features: list[str] = []

    def fit(self, X: pd.DataFrame, y, X_valid=None, y_valid=None):
        import lightgbm as lgb
        self.features = list(X.columns)
        dtrain = lgb.Dataset(X, label=np.asarray(y), free_raw_data=True)
        valid_sets, callbacks = [dtrain], [lgb.log_evaluation(50)]
        if X_valid is not None:
            dvalid = lgb.Dataset(X_valid[self.features], label=np.asarray(y_valid), reference=dtrain)
            valid_sets = [dtrain, dvalid]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=True))
        params = {**self.params, "metric": ["binary_logloss", "auc"]}
        self.booster = lgb.train(params, dtrain, num_boost_round=self.num_boost_round,
                                 valid_sets=valid_sets, callbacks=callbacks)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.features], num_iteration=self.booster.best_iteration or None)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.features,
                             "gain": self.booster.feature_importance("gain")}).sort_values("gain", ascending=False)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.booster.save_model(os.path.join(directory, "lgbm_matcher.txt"))
        with open(os.path.join(directory, "lgbm_matcher.json"), "w") as f:
            json.dump({"features": self.features, "best_iteration": self.booster.best_iteration,
                       "params": self.params}, f, indent=1)

    @classmethod
    def load(cls, directory: str) -> "LGBMMatcher":
        import lightgbm as lgb
        with open(os.path.join(directory, "lgbm_matcher.json")) as f:
            meta = json.load(f)
        m = cls(meta["params"])
        m.booster = lgb.Booster(model_file=os.path.join(directory, "lgbm_matcher.txt"))
        m.features = meta["features"]
        return m


class RuleBasedMatcher:
    """Phase-1 conservative rules (baseline)."""

    def __init__(self, name_threshold: float = 0.65, addr_threshold: float = 0.45):
        self.name_threshold = name_threshold
        self.addr_threshold = addr_threshold

    def predict(self, s1_id, s2s3_id, features: dict, addr1: str = "", addr2: str = ""):
        name_tset = features.get("name_tset", 0.0)
        name_jaccard = features.get("name_jaccard", 0.0)
        addr_tset = features.get("addr_tset", 0.0)
        name_ratio = features.get("name_ratio", 0.0)
        score = 0.5 * name_tset + 0.2 * name_jaccard + 0.3 * addr_tset
        addr_missing = (not addr1 or len(addr1) < 3) or (not addr2 or len(addr2) < 3)
        is_match = False
        if name_tset >= 0.90 and name_jaccard >= 0.50:
            is_match = addr_missing or addr_tset >= 0.30
        elif name_tset >= self.name_threshold and name_ratio >= 0.55:
            is_match = addr_missing or addr_tset >= self.addr_threshold
        elif name_tset >= 0.55 and name_jaccard >= 0.25:
            is_match = addr_tset >= 0.75
        return is_match, score


class MatchingModel:
    """Config-driven factory (Decision D004)."""

    def __init__(self, method: str = "lightgbm", **kwargs):
        if method == "baseline_rules":
            self.model = RuleBasedMatcher(**kwargs)
        elif method == "lightgbm":
            self.model = LGBMMatcher(**kwargs)
        else:
            raise ValueError(f"Unknown matching method: {method}")
        self.method = method
