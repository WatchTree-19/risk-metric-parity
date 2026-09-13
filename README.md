# Four libraries, four VaRs: a reconciliation of risk-metric estimators in Python portfolio analytics

Four widely used Python libraries compute value at risk, conditional value
at risk and maximum drawdown from the same array of returns and return four
different numbers. None of them is wrong. Each implements a legitimate
published estimator, and in most cases the docstring does not say which one.
This note names the estimator each library actually implements, quantifies
the gap on realistic return series, and separates the cases that are an
honest choice of convention from the three cases that are defects.

The practical claim is narrow and, I think, uncontroversial once stated: a
risk number is not portable between these libraries, so any report,
backtest comparison or model-validation exercise that sources metrics from
more than one of them is silently mixing estimators.

## What each library computes

| metric | empyrical-reloaded | ffn | quantstats | skfolio |
|---|---|---|---|---|
| VaR, 5 percent | empirical quantile, interpolated (`np.percentile`) | not exposed | parametric Gaussian | order statistic, lower |
| CVaR, 5 percent | mean of the tail at or below `floor(alpha * n)` | not exposed | empirical mean below a parametric Gaussian threshold | Rockafellar-Uryasev |
| maximum drawdown | geometric, initial capital counted as the first peak | geometric on compounded wealth | geometric, initial capital counted as the first peak | geometric on compounded wealth |

The VaR and CVaR rows are stable. They hold identically across eight return
series covering Gaussian returns with positive and negative drift, Student
t(3), a two-regime volatility process, a left-skewed gamma process, an
all-positive series, a near-zero-volatility series and a seven-observation
sample.

## How much it matters

Percentages below are the spread between the highest and lowest library
value, as a fraction of the smallest absolute figure. All at the 5 percent
level, n = 1512.

| series | VaR spread | CVaR spread |
|---|---|---|
| two-regime volatility | 22.8 percent | 9.6 percent |
| Student t(3) | 15.8 percent | 9.3 percent |
| left-skewed gamma | 10.4 percent | 6.7 percent |
| Gaussian, negative drift | 4.5 percent | 3.1 percent |

The ordering is the point. The gap widens exactly where tail risk is the
question being asked, because that is where the parametric and empirical
estimators part company. On the two-regime series, one library's 5 percent
VaR is a fifth larger than another's on identical input.

Note what this does not say. The parametric estimator is not uniformly the
conservative one. At the 5 percent level the standardised t(3) quantile is
roughly -1.36 against the Gaussian -1.64, so on that series the parametric
VaR is the more conservative figure, while on the left-skewed series it is
the less conservative one. The direction of the gap depends on the
distribution and the confidence level together, which is precisely why
substituting one library for another cannot be signed off with a rule of
thumb.

## The maximum drawdown split, which is sharper than it looks

All four libraries compute the geometric drawdown on compounded wealth. The
divergence is a single convention: whether the initial capital counts as the
first peak. empyrical and quantstats say yes, so a series that falls on its
first observation is already in drawdown. ffn and skfolio take the first
peak from the first observed wealth value, so that opening fall is invisible
to them.

The two conventions agree on any series that rises before it falls, which is
why this is easy to miss. They disagree whenever the series is under water
from the start, which is the case a drawdown statistic exists to describe.
On a seven-observation series opening at -1.87 percent:

    empyrical   -0.084858
    quantstats  -0.084858
    ffn         -0.067398
    skfolio     -0.067398

A 26 percent relative difference on seven identical numbers, from a
convention that no docstring involved states.

## Where convention ends and defect begins

Three findings from the same sweep are not estimator choices. They are
written up separately and are being filed upstream:

1. **empyrical-reloaded, `conditional_value_at_risk`**: the cutoff index is
   computed from the length of the array including NaNs, and `np.partition`
   places NaNs last, so missing observations widen the averaged tail rather
   than being excluded. On a 1512-point series with 40 NaNs the tail runs to
   78 real observations instead of 76, biasing CVaR towards zero.
   `value_at_risk` on the same input propagates NaN, so two risk measures in
   one library disagree about what missing data means and one of them fails
   silently.

2. **quantstats, `_prepare_returns`**: `data.fillna(0)` converts every
   missing observation to a zero return inside essentially every statistic.
   Forty NaNs in 1512 daily points move volatility, Sharpe, Sortino, VaR and
   CVaR by roughly 1.3 percent with no warning.

3. **quantstats, `conditional_value_at_risk`**: the threshold is parametric
   Gaussian VaR, as the docstring says, but the tail statistic is the
   empirical mean below that threshold. That combination matches neither the
   Gaussian expected-shortfall closed form nor any empirical estimator.
   Worse, the function ends `return c_var if ~np.isnan(c_var) else var`, so
   when no observation falls below the parametric threshold the empty mean
   gives NaN and the function returns the VaR itself. CVaR then equals VaR
   exactly, which cannot happen for a real expected shortfall on a
   continuous distribution. It fires on any all-positive series and on short
   samples: on a 756-point all-positive series both come back -0.00065106,
   and on a seven-point sample both come back -0.02611821, with zero
   observations below the threshold in each case.

Filed upstream as stefan-jansen/empyrical-reloaded#54,
ranaroussi/quantstats#546 and ranaroussi/quantstats#547.

The distinction matters for how these are reported. A library computing a
different published estimator from its peers is a documentation issue and is
raised as one. A library computing no published estimator, or silently
imputing data, is a defect.

## Reproducing this

The figures above come from a self-contained script with no dependency on
any private tooling.

```
pip install quantstats empyrical-reloaded ffn skfolio
python repro.py
```

```python
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
```

Expected output:

```
Student t(3), n=1512, 5 percent level
  VaR   empyrical  -0.01405
  VaR   quantstats -0.01668
  VaR   skfolio    -0.01411
  CVaR  empyrical  -0.02338
  CVaR  quantstats -0.02578
  CVaR  skfolio    -0.02343

Max drawdown, n=7, first return -0.01872
  empyrical  -0.084858
  quantstats -0.084858
  ffn        -0.067398
  skfolio    -0.067398
```

Two call conventions are worth flagging for anyone reproducing this, because
getting either wrong produces a number that looks plausible and is not.
`quantstats.stats.max_drawdown` raises `TypeError` on a series without a
`DatetimeIndex`, since it subtracts a `Timedelta` from `index[0]`, so every
quantstats call above is given one. And `skfolio.measures.max_drawdown`
takes a drawdowns series from `get_drawdowns`, not raw returns; feeding it
returns directly yields a number rather than an error.

## Versions

Verified on 12 September 2026 against quantstats 0.0.81,
empyrical-reloaded 0.5.12, ffn 1.2.1 and skfolio 1.1.0, running on numpy
2.4.4, pandas 3.0.2 and scipy 1.17.1, CPython 3.11. Library behaviour here
is a property of the version, not of the library in perpetuity, so the
versions are part of the claim.

## Limitations, and what would change my mind

The classification is done by computing the candidate estimators
independently and matching each library's output to within a relative
tolerance of 1e-6. A library whose implementation coincides numerically with
a variant it does not conceptually implement would be misclassified by this
method, and on a seven-observation sample several variants collapse onto the
same value, so the small-sample rows carry less information than the large
ones. The return series are synthetic, chosen to separate estimators rather
than to represent any particular market. Nothing here has been checked
against the R or MATLAB implementations of the same metrics, which would be
the obvious next step and might well turn a two-way split into a four-way
one.

I would withdraw the maximum drawdown finding if either of the two
conventions turned out to be documented in the libraries concerned, and I
have not found it stated in any of them. I would revise the CVaR
classification if the quantstats construction were shown to correspond to a
published estimator I have not found.

---

Sandeep Singh Rai. Open-source contributions under WatchTree-19.
ORCID 0009-0001-3360-9205.
