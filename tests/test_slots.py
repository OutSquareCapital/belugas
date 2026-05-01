import pytest
from pyochain import Err, Ok, Result

import pql

_OBJS = [
    pql.col(""),
    pql.LazyFrame({"a": [1]}),
    pql.when(1),
    pql.when(1).then(2),
    pql.when(1).then(2).when(3).then(4),
    pql.selectors.all(),
    *pql.datatypes.NON_NESTED_MAP.values(),
]


@pytest.mark.parametrize("obj", _OBJS)
def test_slots(obj: object) -> None:
    assert _check_slots(obj).is_ok()


def _check_slots(obj: object) -> Result[None, str]:
    try:
        _ = obj.__dict__
        msg = f"{obj.__class__.__name__} has __dict__, but should have __slots__"
        return Err(msg)
    except AttributeError:
        return Ok(None)
