from hypothesis import given
import hypothesis.extra.numpy as np_st
import hypothesis.strategies as st
import itertools
import numpy as np

from utils.arrays import nd_cartesian

@given(
    arrays=st.lists(np_st.arrays(np_st.integer_dtypes(), np_st.array_shapes(1, 1, max_side=4)), min_size=1, max_size=4)
)
def test_nd_cartesian(arrays):
    nd_result = [tuple(x) for x in nd_cartesian(*arrays).tolist()]
    itertools_result = list(itertools.product(*arrays))
    assert nd_result == itertools_result
