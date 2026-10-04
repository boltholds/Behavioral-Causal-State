"""Small protocol metrics; callers must supply independent base-group samples."""
import math
from statistics import mean, variance


def total_variation(p,q):
    if len(p) != len(q) or not p:
        raise ValueError('matching nonempty outcome spaces required')
    for law in (p,q):
        if any(not math.isfinite(x) or x < 0 or x > 1 for x in law) or not math.isclose(sum(law),1,rel_tol=0,abs_tol=1e-12):
            raise ValueError('normalized probability laws required')
    return sum(abs(a-b) for a,b in zip(p,q))/2


def empirical_bernstein_upper(samples,lower,upper,alpha=.05):
    if len(samples) < 2 or not 0 < alpha < 1 or not lower < upper or any(not math.isfinite(x) or not lower <= x <= upper for x in samples):
        raise ValueError('invalid bounded independent samples')
    n = len(samples)
    log = math.log(2/alpha)
    return min(upper,mean(samples)+math.sqrt(2*variance(samples)*log/n)+7*(upper-lower)*log/(3*(n-1)))
