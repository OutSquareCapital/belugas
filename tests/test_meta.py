from collections.abc import Callable

import pytest
from pyochain import Iter, Seq

import pql
from pql import meta


def _get_fn(name: str) -> Callable[..., pql.LazyFrame]:
    return getattr(meta, name)  # pyright: ignore[reportAny]


_META_FNS: Seq[Callable[[], pql.LazyFrame]] = (
    Iter(dir(meta))
    .map(_get_fn)
    .filter(lambda fn: callable(fn) and fn.__name__ != "LazyFrame")
    .collect()
)


@pytest.mark.parametrize("fns", _META_FNS)
def test_meta_fns(fns: Callable[..., pql.LazyFrame]) -> None:
    assert isinstance(fns(), pql.LazyFrame)
