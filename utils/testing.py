import hypothesis.strategies as st

def dimensions_strategy(min_dim=1, max_dim=4):
    """A testing strategy for Hypothesis that generates reasonable cellular
    automaton dimension numbers.

    This is really just st.integers() with different defaults.
    """
    return st.integers(min_dim, max_dim)
