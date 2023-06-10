std_lib_print = print

def print(*args, **kwargs):
    flush = True
    if kwargs and "flush" in kwargs:
        flush = kwargs.pop("flush")
    
    std_lib_print(*args, **kwargs, flush=flush)