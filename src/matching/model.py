"""Conservative rule-based matching model.

Prioritizes precision over recall for F0.5 metric.
Uses simple feature thresholds — no ML model needed for baseline.
"""


class RuleBasedMatcher:
    """Conservative rule-based matcher.

    A candidate pair is matched if:
      - name_tset >= name_threshold AND
      - (addr_tset >= addr_threshold OR both addresses are empty/missing)

    Higher thresholds = fewer false positives = better F0.5.
    """

    def __init__(self, name_threshold: float = 0.65, addr_threshold: float = 0.45):
        self.name_threshold = name_threshold
        self.addr_threshold = addr_threshold

    def predict(self, s1_id: str, s2s3_id: str, features: dict,
                addr1: str = "", addr2: str = "") -> tuple[bool, float]:
        """Predict whether a candidate pair is a match.

        Returns (is_match, confidence_score).
        """
        name_tset = features.get("name_tset", 0.0)
        name_jaccard = features.get("name_jaccard", 0.0)
        addr_tset = features.get("addr_tset", 0.0)
        name_ratio = features.get("name_ratio", 0.0)

        # Confidence score: weighted combination
        score = 0.5 * name_tset + 0.2 * name_jaccard + 0.3 * addr_tset

        # Address handling: if either address is empty/short, relax addr requirement
        addr_missing = (not addr1 or len(addr1) < 3) or (not addr2 or len(addr2) < 3)

        # Match rules (conservative, precision-first):
        is_match = False

        # Rule 1: Very high name similarity (likely exact or near-exact match)
        if name_tset >= 0.90 and name_jaccard >= 0.50:
            if addr_missing or addr_tset >= 0.30:
                is_match = True

        # Rule 2: Good name + good address
        elif name_tset >= self.name_threshold and name_ratio >= 0.55:
            if addr_missing or addr_tset >= self.addr_threshold:
                is_match = True

        # Rule 3: Moderate name but excellent address (address confirms entity)
        elif name_tset >= 0.55 and name_jaccard >= 0.25:
            if addr_tset >= 0.75:
                is_match = True

        return is_match, score


class MatchingModel:
    """Wrapper interface for matching models. Decision D004: config-driven."""

    def __init__(self, method: str = "baseline_rules", **kwargs):
        if method == "baseline_rules":
            self.model = RuleBasedMatcher(**kwargs)
        else:
            raise ValueError(f"Unknown matching method: {method}")
        self.method = method

    def fit(self, features, labels=None):
        """No-op for rule-based model."""
        pass

    def predict_scores(self, features):
        """Not used in rule-based approach."""
        raise NotImplementedError("Use predict() directly for rule-based matching.")
