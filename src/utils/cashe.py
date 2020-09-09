from functools import wraps

def cached(func):
    return_value = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal return_value
        if kwargs.get("cache") is None:
            return_value = func(*args, **kwargs)
        return return_value

    return wrapper
