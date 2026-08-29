import numpy as np

from common import GradientReversal as CommonGRL
from common import GradientReversalFn as CommonGRLFn
from common.grl import GradientReversal, GradientReversalFn
from lstm_dann.Model.GRL import GradientReversal as LSTM_GRL
from ops_dann.Model.GRL import GradientReversal as OPS_GRL
from utils import FeatureStats as UtilsFeatureStats
from utils.types import FeatureStats


def test_feature_stats():
    mins = np.array([0.0, 1.0, 2.0])
    maxs = np.array([10.0, 20.0, 30.0])
    stats = FeatureStats(min=mins, max=maxs)

    assert np.array_equal(stats.min, mins)
    assert np.array_equal(stats.max, maxs)
    assert stats.__class__ is UtilsFeatureStats


def test_grl_common_reexports():
    assert LSTM_GRL is GradientReversal
    assert OPS_GRL is GradientReversal
    assert CommonGRL is GradientReversal
    assert CommonGRLFn is GradientReversalFn
