"""G2: exact feasibility in the declared XOR family, separate from sampling."""
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction as F
import math
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING
import numpy as np
from scipy.stats import beta
from .simulator import Regime, WorldSpec, DeviceState, Variable as V, Action, advance_device

@dataclass(frozen=True)
class Interval:
    lower: F
    upper: F
    def __post_init__(self):
        if not isinstance(self.lower,F) or not isinstance(self.upper,F) or not 0 <= self.lower <= self.upper <= 1:
            raise ValueError('exact rational probability interval required')

class NoiseClass(StrEnum):
    INDEPENDENT = 'H_ind'
    JOINT = 'H_joint'

@dataclass(frozen=True)
class Compatible:
    weights: tuple[F,F,F,F]

@dataclass(frozen=True)
class Incompatible:
    obstruction: str

@dataclass(frozen=True)
class NotEstablished:
    reason: str


def _endpoints(panel):
    if not isinstance(panel,tuple) or len(panel) != 4 or not all(isinstance(i,Interval) for i in panel):
        raise ValueError('four intervals in 00,01,10,11 order required')
    i00,i01,i10,i11 = panel
    return max(i00.lower,1-i10.upper),min(i00.upper,1-i10.lower),max(i01.lower,i11.lower),min(i01.upper,i11.upper)


def certify(panel,model: NoiseClass,budget=1):
    if not isinstance(model,NoiseClass) or type(budget) is not int or budget < 0:
        raise ValueError('invalid model/budget')
    xlo,xhi,blo,bhi = _endpoints(panel)
    if budget == 0:
        return NotEstablished('BudgetExhausted')
    if xlo > xhi:
        return Incompatible('complementary_R_rows_disjoint')
    if blo > bhi:
        return Incompatible('C_one_rows_disjoint')
    if model == NoiseClass.JOINT:
        # Choose D=NM xor NY independently of NY, then NM=NY xor D.
        x,b = xlo,blo
        return Compatible(((1-b)*(1-x),b*x,(1-b)*x,b*(1-x)))
    m = min(blo,1-bhi)
    x = max(xlo,m)
    if x > min(xhi,1-m):
        return Incompatible('independent_xor_range_disjoint')
    b = blo if blo <= 1-bhi else bhi
    a = F(0) if b == F(1,2) else (x-b)/(1-2*b)
    return Compatible(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b))


def verify(panel,model,result):
    """Independent witness substitution / obstruction inequality check, exact arithmetic."""
    xlo,xhi,blo,bhi = _endpoints(panel)
    if not isinstance(model,NoiseClass):
        return False
    if isinstance(result,Compatible):
        w = result.weights
        if len(w) != 4 or any(not isinstance(p,F) or p < 0 for p in w) or sum(w) != 1:
            return False
        p00,p01,p10,p11 = w
        if model == NoiseClass.INDEPENDENT and p00*p11 != p01*p10:
            return False
        predictions = (p01+p10,p01+p11,p00+p11,p01+p11)
        return all(interval.lower <= p <= interval.upper for p,interval in zip(predictions,panel))
    if isinstance(result,Incompatible):
        if result.obstruction == 'complementary_R_rows_disjoint':
            return xlo > xhi
        if result.obstruction == 'C_one_rows_disjoint':
            return blo > bhi
        if result.obstruction == 'independent_xor_range_disjoint' and model == NoiseClass.INDEPENDENT:
            m = min(blo,1-bhi)
            return xlo <= xhi and blo <= bhi and (xhi < m or xlo > 1-m)
    return False


def binomial_cdf_enclosure(k,n,p):
    """Directed Decimal interval sum enclosing P(Bin(n,p)<=k).

    All operations are positive +, *, / or interval subtraction. Integer powers
    use repeated multiplication (no transcendental/power implementation). The
    returned decimals therefore enclose the exact rational binomial sum.
    """
    if type(n) is not int or n < 0 or type(k) is not int or not isinstance(p,F) or not 0 <= p <= 1:
        raise ValueError('invalid exact binomial arguments')
    if k < 0:
        return Decimal(0),Decimal(0)
    if k >= n or p == 0:
        return Decimal(1),Decimal(1)
    if p == 1:
        return Decimal(0),Decimal(0)
    down = Context(prec=60,rounding=ROUND_FLOOR,Emin=-999999999,Emax=999999999)
    up = Context(prec=60,rounding=ROUND_CEILING,Emin=-999999999,Emax=999999999)
    one = Decimal(1)
    pl = down.divide(Decimal(p.numerator),Decimal(p.denominator))
    pu = up.divide(Decimal(p.numerator),Decimal(p.denominator))
    ql,qu = down.subtract(one,pu),up.subtract(one,pl)
    def power(ctx,value,exponent):
        result = one
        while exponent:
            if exponent & 1:
                result = ctx.multiply(result,value)
            exponent //= 2
            if exponent:
                value = ctx.multiply(value,value)
        return result
    tl,tu = power(down,ql,n),power(up,qu,n)
    sl,su = tl,tu
    rl,ru = down.divide(pl,qu),up.divide(pu,ql)
    for i in range(k):
        tl = down.multiply(down.divide(down.multiply(tl,Decimal(n-i)),Decimal(i+1)),rl)
        tu = up.multiply(up.divide(up.multiply(tu,Decimal(n-i)),Decimal(i+1)),ru)
        sl,su = down.add(sl,tl),up.add(su,tu)
    return max(Decimal(0),sl),min(one,su)


def clopper_pearson(successes,trials,alpha):
    if type(successes) is not int or type(trials) is not int or trials <= 0 or not 0 <= successes <= trials or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('invalid binomial interval arguments')
    tail = F(alpha)/2
    lo = 0.0 if successes == 0 else float(beta.ppf(float(tail),successes,trials-successes+1))
    hi = 1.0 if successes == trials else float(beta.ppf(float(1-tail),successes+1,trials-successes))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ArithmeticError('nonfinite beta quantile')
    # SciPy proposes endpoints; verified binomial inequalities certify enclosure.
    # A bounded verification failure never becomes an infeasibility certificate.
    for steps in (1,4,16,64,256,1024):
        lower = max(0.0,math.nextafter(lo,-math.inf,steps=steps)) if successes else 0.0
        upper = min(1.0,math.nextafter(hi,math.inf,steps=steps)) if successes < trials else 1.0
        low_ok = successes == 0 or F(binomial_cdf_enclosure(successes-1,trials,F(lower))[0]) >= 1-tail
        high_ok = successes == trials or F(binomial_cdf_enclosure(successes,trials,F(upper))[1]) <= tail
        if low_ok and high_ok:
            return Interval(F(lower),F(upper))
    raise ArithmeticError('NumericalFailure: outward CP endpoints could not be certified')


def certificate_wire(result):
    if isinstance(result,Compatible):
        return {'kind':'Compatible','weights':[str(x) for x in result.weights],'computation':'Complete','verification':'exact rational substitution'}
    if isinstance(result,Incompatible):
        return {'kind':'Incompatible','obstruction':result.obstruction,'computation':'Complete','verification':'exact rational obstruction inequalities'}
    return {'kind':'NotEstablished','computation':result.reason,'certificate':'NoCertificate'}

REGIMES = {'independent_xor':Regime.STOCHASTIC_PARTIAL,'shared_xor':Regime.CORRELATED_PARTIAL,'wrong_gate':Regime.WRONG_GATE_PARTIAL}
EXPECTED = {'independent_xor':(True,True),'shared_xor':(False,True),'wrong_gate':(False,False)}


def population_panel(regime):
    spec = WorldSpec.for_regime(REGIMES[regime])
    return tuple(sum((p for nm,ny,p in spec.noise.support() if advance_device(DeviceState(0,0,0,0),Action.HOLD,((V.R,r),(V.C,c)),nm,ny,spec.wrong_gate).y),F(0)) for r,c in ((0,0),(0,1),(1,0),(1,1)))


def run_panel(seeds=tuple(range(100)),episodes=10000):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds) or type(episodes) is not int or episodes <= 0:
        raise ValueError('distinct nonnegative seeds and positive episode count required')
    exact, exact_certificates, rows = {}, {}, []
    rejected = {regime:{m.value:0 for m in NoiseClass} for regime in REGIMES}
    for regime in REGIMES:
        panel = tuple(Interval(p,p) for p in population_panel(regime))
        exact[regime] = {}
        exact_certificates[regime] = {}
        for model in NoiseClass:
            result = certify(panel,model)
            if not verify(panel,model,result):
                raise ArithmeticError('exact certificate failed verification')
            exact[regime][model.value] = isinstance(result,Compatible)
            exact_certificates[regime][model.value] = certificate_wire(result)
    for seed in seeds:
        for regime_index,regime in enumerate(REGIMES):
            counts, intervals = [], []
            interval_failures = []
            for cell,(r,c) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                rng = np.random.default_rng(np.random.SeedSequence([seed,regime_index,cell]))
                nm = rng.random(episodes) < .1
                ny = nm if regime == 'shared_xor' else rng.random(episodes) < .1
                m = np.logical_xor(bool(r),nm)
                y = np.logical_xor(m & bool(c if regime == 'wrong_gate' else 1-c),ny)
                count = int(np.count_nonzero(y))
                counts.append(count)
                try:
                    intervals.append(clopper_pearson(count,episodes,F(1,240)))
                except ArithmeticError as error:
                    interval_failures.append(str(error))
            statuses, certificates = {}, {}
            if interval_failures:
                rows.append({'seed':seed,'regime':regime,'successes':counts,'scope':'FiniteDataConstraints','statuses':{m.value:'NotEstablished:NumericalFailure' for m in NoiseClass},'computation':'NumericalFailure','certificate':'NoCertificate','errors':interval_failures})
                continue
            for model in NoiseClass:
                result = certify(tuple(intervals),model)
                if not verify(tuple(intervals),model,result):
                    statuses[model.value] = 'NotEstablished:NumericalFailure'
                    certificates[model.value] = certificate_wire(NotEstablished('NumericalFailure'))
                else:
                    statuses[model.value] = 'FeasibleWitnesses' if isinstance(result,Compatible) else 'CertifiedInfeasible'
                    rejected[regime][model.value] += isinstance(result,Incompatible)
                    certificates[model.value] = certificate_wire(result)
            rows.append({'seed':seed,'regime':regime,'successes':counts,'intervals':[[float(i.lower),float(i.upper)] for i in intervals], 'scope':'FiniteDataConstraints','statuses':statuses,'certificates':certificates})
    rates = {}
    for regime,counts in rejected.items():
        rates[regime] = {}
        for model,k in counts.items():
            try:
                interval = clopper_pearson(k,len(seeds),F(1,20))
                rates[regime][model] = [float(interval.lower),float(interval.upper)]
            except ArithmeticError:
                rates[regime][model] = {'status':'NotEstablished:NumericalFailure'}
    return {'episodes':len(seeds)*12*episodes,'seeds':list(seeds),'episodes_per_cell':episodes,'exact_compatibility':exact,'exact_certificates':exact_certificates,'rejections':rejected,'rejection_rate_95pct_intervals':rates,'rows':rows,'quantiles':'SciPy beta.ppf proposals; exact-alpha binomial tail inequalities verified by directed 60-digit Decimal interval arithmetic'}
