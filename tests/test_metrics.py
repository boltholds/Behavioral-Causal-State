import pytest
from bcs.metrics import total_variation, empirical_bernstein_upper


def test_tv_is_joint_distribution_distance_not_marginal_average():
    assert total_variation((.5,0,0,.5),(0,.5,.5,0)) == 1
    with pytest.raises(ValueError):
        total_variation((.5,.4),(.5,.5))


def test_paired_excess_uses_full_minus_one_to_one_range():
    tight = empirical_bernstein_upper([0]*100,0,1,.05)
    paired = empirical_bernstein_upper([0]*100,-1,1,.05)
    assert paired == pytest.approx(2*tight)
    with pytest.raises(ValueError):
        empirical_bernstein_upper([0],0,1,.05)
