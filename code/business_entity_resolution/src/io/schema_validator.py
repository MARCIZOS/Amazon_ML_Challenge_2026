"""Input and output schema validation."""


def validate_schema(data, required_columns):
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
