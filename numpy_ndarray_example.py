import numpy as np
from chain import Chain

chainer = Chain()

a = np.random.randn(3, 2)
handle = chainer.lazy_obj(a).round()
np.testing.assert_array_equal(a.round(), handle.execute())

a = np.random.rand(3, 2)
b = np.random.randn(3, 2)
handle = chainer.lazy_obj(a).__add__(1).__mul__(b)
np.testing.assert_array_equal((a).__add__(1).__mul__(b), handle.execute())
