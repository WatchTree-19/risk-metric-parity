"""Self-contained reproduction: four libraries, four risk estimators.

    pip install quantstats empyrical-reloaded ffn skfolio
    python repro.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import empyrical as ep
import ffn
import quantstats as qs
from skfolio import measures as skm
from skfolio.measures import get_drawdowns

ALPHA = 0.05


def dated(r):
    return pd.Series(np.asarray(r, float),
                     index=pd.bdate_range("2000-01-03", periods=len(r)))


def fat_tail(n=1512, seed=2):
    rng = np.random.default_rng(seed)
    return 0.0004 + 0.011 * rng.standard_t(3, n) / np.sqrt(3.0)


def tiny(n=7, seed=8):
    return np.random.default_rng(seed).normal(0.0004, 0.011, n)


r = fat_tail()
print("Student t(3), n=1512, 5 percent level")
print("  VaR   empyrical  %+.5f" % ep.value_at_risk(r, cutoff=ALPHA))
print("  VaR   quantstats %+.5f" % qs.stats.value_at_risk(dated(r), confidence=1 - ALPHA))
print("  VaR   skfolio    %+.5f" % -skm.value_at_risk(r, beta=1 - ALPHA))
print("  CVaR  empyrical  %+.5f" % ep.conditional_value_at_risk(r, cutoff=ALPHA))
print("  CVaR  quantstats %+.5f" % qs.stats.conditional_value_at_risk(dated(r), confidence=1 - ALPHA))
print("  CVaR  skfolio    %+.5f" % -skm.cvar(r, beta=1 - ALPHA))

t = tiny()
print()
print("Max drawdown, n=7, first return %+.5f" % t[0])
print("  empyrical  %+.6f" % ep.max_drawdown(t))
print("  quantstats %+.6f" % qs.stats.max_drawdown(dated(t)))
print("  ffn        %+.6f" % ffn.core.calc_max_drawdown((1 + pd.Series(t)).cumprod()))
print("  skfolio    %+.6f" % -skm.max_drawdown(get_drawdowns(t, compounded=True)))
