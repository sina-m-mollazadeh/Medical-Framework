"""Pre-sampled, LHS-driven parameter grid for the medical pipeline search.

`build_param_grid(spw_base, total_n=200)` returns `(param_grid, family_index)`:
  * `param_grid` — list of dicts, each pinning one full configuration (every
    value wrapped in a length-1 list). Pass straight to ``GridSearchCV``.
  * `family_index` — list of model-family labels aligned to `param_grid`.

Coverage is enforced by construction: per-family quotas drive the count split,
and an independent Latin Hypercube sample fills each family's joint
preprocessing + classifier space. ``diversity_report`` summarises the assembled
grid before any fitting happens.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc

from imblearn import FunctionSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from clean_data import (
    MeanImputer, ClassMeanImputer, InterpolateImputer,
    KNNImputerWrapper, IterativeModelImputer,
)
from normalization import RobustScalerNorm, ZScoreNormalizationNorm
from feature_selection import SelectKBestFilter, TreeBasedSelection
from balance import SMOTESampler, BorderlineSMOTESampler
from model_training import (
    LogisticRegressionEstimator, RandomForestEstimator,
    XGBoostEstimator, LightGBMEstimator, CatBoostEstimator,
)


# ---------- shared sampler factory -----------------------------------------

def _identity(X, y):
    return X, y


def _make_identity_sampler():
    return FunctionSampler(func=_identity, validate=False)


# ---------- shared categorical sets (every family LHS-samples these) -------

IMPUTERS = [
    MeanImputer(),
    ClassMeanImputer(),
    InterpolateImputer(),
    KNNImputerWrapper(n_neighbors=5),
    IterativeModelImputer(),
]
NORMALIZERS = [ZScoreNormalizationNorm(), RobustScalerNorm()]
DETECTORS = ['iforest', 'cluster', 'statistical']
REMEDIATIONS = ['winsorize', 'percentile_clip', 'mad_replace',
                'cluster_local_median', 'shrink', 'none']
SELECTORS = [
    SelectKBestFilter(k=10),
    SelectKBestFilter(k=20),
    SelectKBestFilter(k=30),
    SelectKBestFilter(k=40),
    TreeBasedSelection(),
]
BALANCERS = [
    _make_identity_sampler(),
    SMOTESampler(k_neighbors=3),
    SMOTESampler(k_neighbors=5),
    BorderlineSMOTESampler(k_neighbors=3),
]


# ---------- unit-cube → param value ----------------------------------------

def _map(u, kind, *args):
    if kind == 'lin':
        lo, hi = args
        return float(lo + u * (hi - lo))
    if kind == 'log':
        lo, hi = args
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    if kind == 'int':
        lo, hi = args
        return int(lo + min(int(u * (hi - lo + 1)), hi - lo))
    if kind == 'logint':
        lo, hi = args
        return int(round(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))))
    if kind == 'cat':
        items, = args
        return items[min(int(u * len(items)), len(items) - 1)]
    if kind == 'mix_int_none':
        # Q4: mostly int[lo, hi], small `none_frac` chance of None.
        lo, hi, none_frac = args
        if u < none_frac:
            return None
        u2 = (u - none_frac) / (1.0 - none_frac)
        return int(lo + min(int(u2 * (hi - lo + 1)), hi - lo))
    raise ValueError(f'unknown kind: {kind!r}')


def _lhs(n, d, seed):
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


# ---------- per-family dimension lists -------------------------------------

PREPROC_DIMS = [
    ('imputer',              'cat', IMPUTERS),
    ('normalizer',           'cat', NORMALIZERS),
    ('tamer__detector',      'cat', DETECTORS),
    ('tamer__remediation',   'cat', REMEDIATIONS),
    ('tamer__contamination', 'log', 0.005, 0.15),
    ('tamer__mad_threshold', 'lin', 2.0, 4.0),
    ('selector',             'cat', SELECTORS),
    ('balancer',             'cat', BALANCERS),
]


def _xgb_dims(spw_cap):
    return [
        ('classifier',                   'cat', [XGBoostEstimator()]),
        ('classifier__learning_rate',    'log',    0.01, 0.3),
        ('classifier__max_depth',        'int',    2, 12),
        ('classifier__n_estimators',     'logint', 100, 1000),
        ('classifier__subsample',        'lin',    0.5, 1.0),
        ('classifier__colsample_bytree', 'lin',    0.5, 1.0),
        ('classifier__min_child_weight', 'logint', 1, 50),
        ('classifier__reg_alpha',        'log',    1e-3, 5.0),
        ('classifier__reg_lambda',       'log',    0.1, 10.0),
        ('classifier__scale_pos_weight', 'log',    1.0, spw_cap),
    ]


def _lgbm_dims(spw_cap):
    return [
        ('classifier',                    'cat', [LightGBMEstimator()]),
        ('classifier__learning_rate',     'log',    0.01, 0.3),
        ('classifier__num_leaves',        'logint', 8, 128),
        ('classifier__n_estimators',      'logint', 100, 1000),
        ('classifier__subsample',         'lin',    0.5, 1.0),
        ('classifier__colsample_bytree',  'lin',    0.5, 1.0),
        ('classifier__min_child_samples', 'logint', 5, 100),
        ('classifier__reg_alpha',         'log',    1e-3, 5.0),
        ('classifier__reg_lambda',        'log',    0.1, 10.0),
        ('classifier__scale_pos_weight',  'log',    1.0, spw_cap),
    ]


def _rf_dims():
    return [
        ('classifier',                   'cat',          [RandomForestEstimator()]),
        ('classifier__n_estimators',     'logint',       100, 1000),
        ('classifier__max_depth',        'mix_int_none', 2, 24, 0.10),
        ('classifier__min_samples_leaf', 'logint',       1, 50),
        ('classifier__max_features',     'cat',          ['sqrt', 'log2', 0.5]),
    ]


def _catboost_dims():
    return [
        ('classifier',                'cat',    [CatBoostEstimator()]),
        ('classifier__iterations',    'logint', 100, 1000),
        ('classifier__learning_rate', 'log',    0.01, 0.3),
        ('classifier__depth',         'int',    2, 10),   # CatBoost memory wall at 10
        ('classifier__l2_leaf_reg',   'log',    1.0, 10.0),
    ]


def _lr_dims():
    return [
        ('classifier',          'cat', [LogisticRegressionEstimator()]),
        ('classifier__C',       'log', 1e-3, 1e2),
        ('classifier__penalty', 'cat', ['l1', 'l2']),
    ]


# ---------- calibrated stacker ---------------------------------------------

def _build_stacker_instance():
    """Construct a fresh CalibratedClassifierCV(StackingClassifier(...)).
    A new instance per candidate avoids cross-candidate mutation when
    sklearn applies set_params on the inner XGB knobs."""
    inner_xgb = XGBoostEstimator(
        n_estimators=300, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    )
    inner_lgbm = LightGBMEstimator(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, reg_lambda=1.0,
    )
    inner_lr = LogisticRegressionEstimator(C=1.0, penalty='l2')
    stacking_clf = StackingClassifier(
        estimators=[('xgb', inner_xgb), ('lgbm', inner_lgbm), ('lr', inner_lr)],
        final_estimator=LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42,
        ),
        stack_method='predict_proba', n_jobs=1, passthrough=False,
    )
    return CalibratedClassifierCV(
        estimator=stacking_clf, method='isotonic', cv=3,
    )


def _stacker_dims(spw_base):
    # Q2: tune three XGB knobs inside the stacker, narrow ranges only.
    spw_low = max(1.0, spw_base / 2.0)
    spw_high = float(min(10.0, spw_base))
    return [
        # Placeholder; replaced by a fresh stacker per candidate in postprocess.
        ('classifier',                                   'cat', [None]),
        ('classifier__estimator__xgb__learning_rate',    'log', 0.03, 0.1),
        ('classifier__estimator__xgb__max_depth',        'int', 3, 6),
        ('classifier__estimator__xgb__scale_pos_weight', 'log', spw_low, spw_high),
    ]


# ---------- sampling -------------------------------------------------------

def _row_to_dict(u_row, dims):
    out = {}
    for (name, kind, *args), u in zip(dims, u_row):
        out[name] = [_map(u, kind, *args)]
    return out


def _sample_family(family_name, dims, n, seed):
    u = _lhs(n, len(dims), seed)
    samples = [_row_to_dict(u[i], dims) for i in range(n)]
    if family_name == 'stacker':
        for s in samples:
            s['classifier'] = [_build_stacker_instance()]
    return samples


# ---------- public API -----------------------------------------------------

DEFAULT_QUOTAS = {
    'xgb':      0.30,
    'lgbm':     0.30,
    'rf':       0.10,
    'catboost': 0.10,
    'lr':       0.10,
    'stacker':  0.10,
}


def build_param_grid(spw_base, total_n=200, quotas=None, seed=42):
    quotas = quotas or DEFAULT_QUOTAS
    spw_cap = float(min(10.0, 1.5 * spw_base))  # Q3

    family_dims = {
        'xgb':      PREPROC_DIMS + _xgb_dims(spw_cap),
        'lgbm':     PREPROC_DIMS + _lgbm_dims(spw_cap),
        'rf':       PREPROC_DIMS + _rf_dims(),
        'catboost': PREPROC_DIMS + _catboost_dims(),
        'lr':       PREPROC_DIMS + _lr_dims(),
        'stacker':  PREPROC_DIMS + _stacker_dims(spw_base),
    }
    counts = {fam: int(round(quotas[fam] * total_n)) for fam in family_dims}
    counts['xgb'] += total_n - sum(counts.values())  # absorb rounding drift

    grid, family_index = [], []
    for i, (fam, dims) in enumerate(family_dims.items()):
        samples = _sample_family(fam, dims, counts[fam], seed=seed + i * 7919)
        grid.extend(samples)
        family_index.extend([fam] * counts[fam])
    return grid, family_index


# ---------- diversity report -----------------------------------------------

_CAT_AXES = [
    '_family', 'imputer', 'normalizer', 'tamer__detector',
    'tamer__remediation', 'selector', 'balancer',
    'classifier__max_features', 'classifier__penalty',
]

_CONT_PREFIXES = (
    'tamer__contamination', 'tamer__mad_threshold',
    'classifier__learning_rate', 'classifier__max_depth',
    'classifier__n_estimators', 'classifier__subsample',
    'classifier__colsample_bytree', 'classifier__min_child_weight',
    'classifier__min_child_samples', 'classifier__reg_alpha',
    'classifier__reg_lambda', 'classifier__scale_pos_weight',
    'classifier__num_leaves', 'classifier__min_samples_leaf',
    'classifier__iterations', 'classifier__depth',
    'classifier__l2_leaf_reg', 'classifier__C',
    'classifier__estimator__xgb__learning_rate',
    'classifier__estimator__xgb__max_depth',
    'classifier__estimator__xgb__scale_pos_weight',
)


def _short(v):
    if v is None:
        return 'None'
    if isinstance(v, (int, float, str, bool)):
        return str(v)
    name = type(v).__name__
    for attr in ('k', 'k_neighbors', 'n_neighbors'):
        if hasattr(v, attr):
            return f'{name}({attr}={getattr(v, attr)})'
    return name


def diversity_report(param_grid, family_index):
    df = pd.DataFrame([{k: v[0] for k, v in cfg.items()} for cfg in param_grid])
    df.insert(0, '_family', family_index)

    print('=== Family distribution ===')
    for fam, n in df['_family'].value_counts().sort_index().items():
        print(f'  {fam:10s} {n:4d}   ({100 * n / len(df):.1f}%)')
    print()

    print('=== Categorical entropy (Hₙ ∈ [0,1]; 1.0 = uniform across categories) ===')
    for c in _CAT_AXES:
        if c not in df.columns:
            continue
        vals = df[c].dropna().apply(_short)
        if vals.empty:
            continue
        counts = vals.value_counts()
        p = counts / counts.sum()
        h = -(p * np.log(p)).sum()
        hn = float(h / np.log(len(counts))) if len(counts) > 1 else 0.0
        warn = '  ⚠ low' if hn < 0.80 else ''
        print(f'  {c:35s}  H={hn:.3f}   k={len(counts):2d}   n={vals.size:3d}{warn}')
    print()

    cont_cols = [c for c in df.columns if c.startswith(_CONT_PREFIXES)]
    if cont_cols:
        print('=== Continuous spread (per-axis; only over candidates that sample it) ===')
        print(f'  {"axis":45s}  {"min":>10s}  {"max":>10s}  {"std":>9s}  {"n":>4s}')
        for c in sorted(cont_cols):
            vals = pd.to_numeric(df[c], errors='coerce').dropna()
            if vals.empty:
                continue
            print(f'  {c:45s}  {vals.min():>10.4g}  {vals.max():>10.4g}  {vals.std():>9.4g}  {len(vals):>4d}')

    return df
