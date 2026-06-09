from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, overload

from pyochain import Err, Iter, Ok, Result, Seq
from pyochain.abc import PyoIterator

from ..._expr import Expr
from ...utils import try_iter

if TYPE_CHECKING:
    from sqlglot import exp

    from ...typing import DescConds, IntoExpr, TryIter, TrySeq


def sort(
    by: TryIter[IntoExpr],
    more_by: Iterable[IntoExpr],
    *,
    descending: DescConds,
    nulls_last: TrySeq[bool],
) -> PyoIterator[exp.Expr]:

    return (
        try_iter(by)
        .chain(more_by)
        .map(lambda v: Expr.new(v, as_col=True))
        .collect(Seq)
        .into(
            lambda sort_exprs: sort_exprs.iter().zip(
                check_by_arg(
                    sort_exprs, "descending", arg=descending, broadcast_nones=False
                ).unwrap(),
                check_by_arg(
                    sort_exprs, "nulls_last", arg=nulls_last, broadcast_nones=True
                ).unwrap(),
            )
        )
        .map_star(
            lambda expr, desc, nls: expr.order_by(descending=desc, nulls_last=nls).inner
        )
    )


class SortArgsError(ValueError):
    def __init__(
        self, name: str, len_arg: int, provided: Seq[Expr], expected_length: int
    ) -> None:
        msg = f"""
The length of `{name}` ({len_arg}) does not match the length of `by` ({expected_length}).
Current expr:
{provided!r}
"""
        super().__init__(msg)


type CheckRes[T] = Result[PyoIterator[T], SortArgsError]
"""Ouptut after checking the sorting arguments (`descending` and `nulls_last`) against the length of the `by` expressions,
and handling broadcasting of single `None` or `bool` values if necessary."""


@overload
def check_by_arg(
    compared: Seq[Expr], name: str, *, arg: None, broadcast_nones: Literal[True]
) -> CheckRes[None]: ...
@overload
def check_by_arg(
    compared: Seq[Expr], name: str, *, arg: DescConds, broadcast_nones: Literal[False]
) -> CheckRes[bool]: ...
@overload
def check_by_arg(
    compared: Seq[Expr], name: str, *, arg: TrySeq[bool], broadcast_nones: bool
) -> CheckRes[bool] | CheckRes[None]: ...
def check_by_arg(
    compared: Seq[Expr], name: str, *, arg: TrySeq[bool], broadcast_nones: bool
) -> CheckRes[bool] | CheckRes[None]:
    length = compared.len()
    match arg:
        case Sequence():
            len_arg = len(arg)
            if len_arg == length:
                return Ok(Iter(arg))
            return Err(SortArgsError(name, len_arg, compared, length))
        case None if broadcast_nones:
            return Ok(Iter.once(arg).cycle().take(length))

        case _:
            return Ok(try_iter(arg).cycle().take(length))
