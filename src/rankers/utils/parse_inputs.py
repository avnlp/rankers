from ast import literal_eval


def dict_type(input_value):
    """Convert a string to a dictionary."""
    return {} if not input_value.strip() or input_value.strip() in ("{}", "''", '""') else literal_eval(input_value)
