
def to_int(value: str):
    """
    Safely turns the string into an integer
    :rtype : int or None
    http://stackoverflow.com/questions/379906/python-parse-string-to-float-or
    -int
    """
    try:
        return int(float(value))
    except ValueError:
        return None


def to_float(value: str):
    """
    Safely turns the string into a float
    :rtype : int or None
    http://stackoverflow.com/questions/379906/python-parse-string-to-float-or
    -int
    """
    try:
        return float(value)
    except ValueError:
        return None
