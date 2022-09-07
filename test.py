from chain import Chain
import unittest
import numpy as np

chainer = Chain()


class ObjectB:
    @chainer.lazy
    def bar(self, x):
        return ObjectA("", 26)


class ObjectA:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Only wrapped method calls are tracked.
    @chainer.lazy
    def foo(self, x):
        self.x = x
        return ObjectB()

    @chainer.lazy
    def bar(self, y):
        self.y = 10
        return y + 2


def return_handle():
    # Enable tracking of calls
    with chainer:
        a = ObjectA("Random Name", 25).foo(42)
        a = a.bar(25).bar(26)

    return a


class Tests(unittest.TestCase):
    def test_handle_execute(self):
        # `execute` actually executes the method chain
        # and clears the tracker
        assert 28 == return_handle().execute()

    def test_context_manager(self):
        with chainer:
            ObjectA("", 25).foo("").bar(26).bar(26)
        # Can directly call the `execute` on the Chain object
        assert chainer.execute() == 28

    def test_eager_eval(self):
        # Eager evaluation as usual.
        chainer.lazy_eval = False
        assert 28 == ObjectA("", 25).foo("").bar(27).bar(26)

    def test_np_basic(self):
        chainer = Chain()

        a = np.random.randn(3, 2)
        handle = chainer.lazy_obj(a).round()
        np.testing.assert_array_equal(a.round(), handle.execute())

    def test_np_longer_chain(self):
        chainer = Chain()

        a = np.random.rand(3, 2)
        b = np.random.randn(3, 2)
        proxy_a = chainer.lazy_obj(a)
        handle = proxy_a.__add__(1).__mul__(b)
        np.testing.assert_array_equal((a).__add__(1).__mul__(b), handle.execute())


if __name__ == "__main__":
    unittest.main()
