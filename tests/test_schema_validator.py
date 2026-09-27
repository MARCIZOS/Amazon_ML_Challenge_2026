import pandas as pd
import pytest

from src.io.schema_validator import validate_schema


def test_validate_schema_success():
    data = pd.DataFrame({"id": [1], "name": ["x"]})
    assert validate_schema(data, ["id", "name"]) is True


def test_validate_schema_missing_column():
    data = pd.DataFrame({"id": [1]})
    with pytest.raises(ValueError):
        validate_schema(data, ["id", "name"])
