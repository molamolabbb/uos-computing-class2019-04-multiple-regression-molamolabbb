#!/usr/bin/env python3

from pytest import approx
from math import sqrt
from multiple import stochastic_minimize

def test_sgd():
    # Test a no data version, so just minimize's the theta function
    o = stochastic_minimize(lambda x, y, t: t[0]**2, lambda x, y, t: [2*t[0]], [[0]], [0], [2], .01, 100)
    assert o[0] == approx(0) # We get really really close to 0
    o = stochastic_minimize(lambda x, y, t: t[0]**4 - t[0]**2, lambda x, y, t: [4*t[0]**3-2*t[0]], [[0]], [0], [2], .01, 100)
    assert o[0] == approx(sqrt(1./2.))
    # Test an example with data
    o = stochastic_minimize(lambda x, y, t: (x[0]*t[0] - y)**2, lambda x, y, t: [x[0]], [[1.]], [1.], [2.], .01, 100)
    assert o[0] == approx(1.)
    o = stochastic_minimize(lambda x, y, t: (x[0]*t[0] - y)**2, lambda x, y, t: [x[0]], [[1.], [2.]], [1., 2.], [2.], .0001, 100)
    assert abs(o[0] - 1.) < 1e-3
