from collections import Counter
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

class IdentitySampler(BaseSampler):
    """Pass-through sampler handling balanced baseline targets seamlessly."""
    _sampling_type = "bypass"
    _parameter_constraints: dict = {}

    def __init__(self):
        super().__init__()
    def _fit_resample(self, X, y):
        return X, y


class SMOTESampler(BaseSampler):
    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, k_neighbors=3, random_state=42):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampler = None

    def _fit_resample(self, X, y):
        self.sampler = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
        return self.sampler.fit_resample(X, y)


class RandomOverSamplerSampler(BaseSampler):
    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.sampler = None

    def _fit_resample(self, X, y):
        self.sampler = RandomOverSampler(random_state=self.random_state)
        return self.sampler.fit_resample(X, y)


class ADASYNSampler(BaseSampler):
    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, n_neighbors=3, random_state=42):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.sampler = None

    def _fit_resample(self, X, y):
        self.sampler = ADASYN(n_neighbors=self.n_neighbors, random_state=self.random_state)
        return self.sampler.fit_resample(X, y)


class BorderlineSMOTESampler(BaseSampler):
    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, k_neighbors=3, random_state=42):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampler = None

    def _fit_resample(self, X, y):
        self.sampler = BorderlineSMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
        return self.sampler.fit_resample(X, y)


class SMOTETomekSampler(BaseSampler):
    _sampling_type = "clean-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.sampler = None

    def _fit_resample(self, X, y):
        self.sampler = SMOTETomek(random_state=self.random_state)
        return self.sampler.fit_resample(X, y)


class SMOTEENNSampler(BaseSampler):
    _sampling_type = "clean-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.sampler = None

    def _fit_resample(self, X, y):
        self.sampler = SMOTEENN(random_state=self.random_state)
        return self.sampler.fit_resample(X, y)


class RandomUnderSamplerSampler(BaseSampler):
    _sampling_type = "under-sampling"
    _parameter_constraints: dict = {}

    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.sampler = None

    def _fit_resample(self, X, y):
        self.sampler = RandomUnderSampler(random_state=self.random_state)
        return self.sampler.fit_resample(X, y)