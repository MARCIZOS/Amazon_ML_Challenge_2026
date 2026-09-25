"""General validation helpers."""


def assert_nonempty(data, name="data"):
    if data is None or len(data) == 0:
        raise ValueError(f"{name} must not be empty")
    return True
