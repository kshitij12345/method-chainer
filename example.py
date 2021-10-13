from chain import Chain

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

# `execute` actually executes the method chain
# and clears the tracker
assert 28 == return_handle().execute()

with chainer:
    ObjectA("", 25).foo("").bar(26).bar(26)
# Can directly call the `execute` on the Chain object
assert chainer.execute() == 28

# Eager evaluation as usual.
chainer.lazy_eval = False
assert 28 == ObjectA("", 25).foo("").bar(27).bar(26)
