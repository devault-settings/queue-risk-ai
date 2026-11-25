"""The benchmark ladder, from rules of thumb to tree ensembles, in the order of the lectures.

    Benchmarks (no learning)        flat completion rate; historical rate by region x technology
    Lecture 3  logistic regression  the interpretable classifier
    Lecture 4  L1-regularized LR    shrinkage, penalty chosen by validation
    Lecture 5  CART                 a readable tree, pruned by ccp_alpha
    Lecture 6  random forest        averaging many deep trees
    Lecture 6  gradient boosting    "usually the strongest default for tabular data"

Every model is a scikit-learn Pipeline, so imputation and scaling are fit on the
training folds only (Lecture 4: standardize on the training set, never the test set).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import CATEGORICAL


def _l1_penalty() -> dict:
    """The LASSO (L1) penalty, spelled for the installed scikit-learn.

    From 1.8, `penalty` is deprecated and `l1_ratio=1.0` alone means L1. BEFORE 1.8,
    `l1_ratio` is IGNORED unless penalty="elasticnet", so `l1_ratio=1.0` silently fits an
    L2 (ridge) model. That happened on Colab (Nov 24): its "LASSO" scored 0.669 out of time
    and 0.566 on test, exactly an L2 model's numbers, against the real LASSO's 0.686 / 0.613.
    """
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    return {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}


L1_PENALTY = _l1_penalty()


# ------------------------------------------------------------------ benchmarks

class FlatRate(ClassifierMixin, BaseEstimator):
    """Business rule: every project gets the historical completion rate. Ranks nothing."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class HistoricalRate(ClassifierMixin, BaseEstimator):
    """Predictive baseline (Lecture 5's "historical baseline"): the training completion
    rate of the project's region x technology cell, falling back to technology, then overall.
    Cells with fewer than `min_count` resolved projects use the fallback."""

    def __init__(self, min_count=20):
        self.min_count = min_count

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        d = pd.DataFrame({"region": X["region"].astype(str), "tech": X["type_clean"].astype(str),
                          "y": np.asarray(y)})
        self.overall_ = d["y"].mean()
        self.by_tech_ = self._rates(d, ["tech"])
        self.by_cell_ = self._rates(d, ["region", "tech"])
        return self

    def _rates(self, d, keys):
        g = d.groupby(keys)["y"].agg(["mean", "size"])
        return g.loc[g["size"] >= self.min_count, "mean"].to_dict()

    def predict_proba(self, X):
        region, tech = X["region"].astype(str), X["type_clean"].astype(str)
        p = [self.by_cell_.get((r, t), self.by_tech_.get(t, self.overall_)) for r, t in zip(region, tech)]
        p = np.asarray(p, dtype=float)
        return np.column_stack([1 - p, p])


# ------------------------------------------------------------------ learned models

def _preprocess(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    # No missing-value indicators: blank fields leak the outcome (features._hide_missingness).
    num_steps = [SimpleImputer(strategy="median", add_indicator=False)]
    if scale:
        num_steps.append(StandardScaler())
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ("num", make_pipeline(*num_steps), numeric),
    ])


def model_zoo(X: pd.DataFrame) -> dict:
    """name -> (estimator, hyperparameter grid for rolling-origin validation)."""
    rs = config.SEED
    return {
        "Flat rate (rule of thumb)": (FlatRate(), {}),
        "Historical rate (region x tech)": (HistoricalRate(), {}),
        "Logistic regression": (
            Pipeline([("prep", _preprocess(X, scale=True)),
                      ("clf", LogisticRegression(max_iter=2000))]), {}),
        "LASSO logistic": (
            Pipeline([("prep", _preprocess(X, scale=True)),
                      ("clf", LogisticRegression(**L1_PENALTY, solver="saga", max_iter=5000))]),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]}),
        "CART": (
            Pipeline([("prep", _preprocess(X, scale=False)),
                      ("clf", DecisionTreeClassifier(min_samples_leaf=50, random_state=rs))]),
            {"clf__ccp_alpha": [1e-4, 3e-4, 1e-3, 3e-3]}),
        "Random forest": (
            Pipeline([("prep", _preprocess(X, scale=False)),
                      ("clf", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=rs))]),
            {"clf__max_depth": [6, 10, None], "clf__max_features": [0.25, 0.5]}),
        "Gradient boosting": (
            Pipeline([("prep", _preprocess(X, scale=False)),
                      ("clf", HistGradientBoostingClassifier(random_state=rs))]),
            {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 4, None],
             "clf__max_iter": [200, 500]}),
        **_tabpfn(X),
    }


def _tabpfn(X: pd.DataFrame) -> dict:
    """TabPFN: a transformer pre-trained on millions of synthetic tables, which predicts a
    new table "in context" with no training and no tuning (Hollmann et al., Nature 2025).

    Course framing: Lecture 12's pre-trained-transformer idea applied to tabular data, and
    a direct test of Lecture 8's finding that neural nets rarely beat tuned tree ensembles
    on tables. Optional: included only when `tabpfn` is installed. Needs a GPU for our
    ~4,000 training rows (run it in Colab: notebooks/tabpfn_colab.ipynb). The weights are
    gated and licensed for non-commercial use; accept the license and set TABPFN_TOKEN.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return {}
    from sklearn.preprocessing import OrdinalEncoder
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    # Impute BEFORE TabPFN, with no missing flag. TabPFN can read NaN natively, but blank
    # fields leak the outcome in this snapshot (features._hide_missingness), so it must not
    # see which values were blank, exactly like the other models.
    prep = ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
         CATEGORICAL),
        ("num", SimpleImputer(strategy="median", add_indicator=False), numeric),
    ])
    cat_idx = list(range(len(CATEGORICAL)))
    try:
        clf = TabPFNClassifier(categorical_features_indices=cat_idx, random_state=config.SEED)
    except TypeError:            # parameter name differs across tabpfn releases
        clf = TabPFNClassifier(random_state=config.SEED)
    return {"TabPFN (pre-trained transformer)": (Pipeline([("prep", prep), ("clf", clf)]), {})}


def selected(name: str, X: pd.DataFrame, outcome: str = "within"):
    """The model `name` at its FROZEN settings (frozen.py), for the 5-year outcome.

    Frozen settings are identical on every machine; re-selection is not (Colab picked LASSO
    C = 0.01 where the laptop picked 0.1). Models not in the frozen spec, or another outcome,
    fall back to the settings the last local pipeline run selected
    (reports/results_<outcome>_validation.csv), then to defaults."""
    import ast
    from .frozen import FROZEN
    est, _ = model_zoo(X)[name]
    if outcome == "within" and name in FROZEN["models"]:
        settings = FROZEN["models"][name]
        if settings:
            est.set_params(**{f"clf__{k}": v for k, v in settings.items()})
        return est
    path = config.ROOT / "reports" / f"results_{outcome}_validation.csv"
    if path.exists():
        res = pd.read_csv(path)
        row = res[res["model"] == name]
        if len(row):
            settings = ast.literal_eval(row["settings"].iloc[0])
            est.set_params(**{f"clf__{k}": v for k, v in settings.items()})
    return est
