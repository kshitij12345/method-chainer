class Chain:
    def __init__(self):
        self.call_chain = []
        self.lazy_eval = True

    def lazy(self, method):
        def fn(obj, *args, **kwargs):
            if self.lazy_eval:
                self.call_chain.append((method, method.__name__, obj, args, kwargs))
                return self

            return method(obj, *args, **kwargs)
        return fn

    def execute(self):
        org_self = self
        # Hold the generated object to call methods on.
        obj = None
        # Set for execution mode
        org_self.lazy_eval = False

        for calls in self.call_chain:

            method, name, self, args, kwargs = calls
            # For the first call, we capture the orignal object!
            if obj is None:
                obj = self

            if not isinstance(obj, Chain):
                if method is not None:
                    output = method(obj, *args, **kwargs)
                else:
                    output = getattr(obj, name)(*args, **kwargs)
            else:
                # Bad Case we probably got an instance of Chain!
                # We can't evaluate that
                raise RuntimeError("ERROR :(")
            
            # Update object for next iteration
            obj = output
        
        org_self.call_chain = []
        org_self.lazy_eval = True
        return output

    def __getattr__(self, name: str):
        def fn(*args, **kwargs):
            obj = None
            if self.obj is not None:
                obj = self.obj
            self.call_chain.append((None, name, obj, args, kwargs))
            self.obj = None
            return self
        return fn
    
    def __enter__(self):
        self.lazy_eval = True
    
    def __exit__(self, type, value, traceback):
        self.lazy_eval = False

    def lazy_obj(self, obj):
        # Return self
        self.obj = obj
        return self
