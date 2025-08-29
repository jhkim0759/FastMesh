MODELS = {}

def register(name):
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator

def make(name, **kwargs):
    dataset = MODELS[name](**kwargs)
    return dataset