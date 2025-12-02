import inspect

def filter_valid_args(arg_dict, class_type):
    valid_keys = inspect.signature(class_type).parameters.keys()
    return {k: v for k, v in arg_dict.items() if k in valid_keys}
