# ml/clean.py
from __future__ import annotations
from typing import Iterable, Literal
import pandas as pd

NumericKind = Literal["int", "float"]


def ensure_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    default: str | float | int = ""
) -> pd.DataFrame:
    """
    DataFrame에 특정 컬럼이 없으면 기본값으로 생성합니다.
    - 모델/전처리 파이프라인이 '컬럼 존재'를 전제로 동작하도록 보장.
    - 기존 df는 변경하지 않고, 복사본을 반환합니다.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = default
    return out


def _coerce_numeric_series(
    s: pd.Series,
    kind: NumericKind,
    fillna: float | int | None
) -> pd.Series:
    """
    개별 Series에 대해 안전하게 숫자형으로 변환합니다.
    - "1,200", " 42 ", "" 같은 문자열을 숫자로 변환
    - 변환 실패/결측은 NaN → fillna로 대체
    - kind="int" → int64, kind="float" → float64
    """
    # 문자열로 캐스팅 후 쉼표/공백 제거
    s2 = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.strip()
    )

    # 숫자 변환 (실패 시 NaN)
    s2 = pd.to_numeric(s2, errors="coerce")

    # 결측 대체
    if fillna is not None:
        s2 = s2.fillna(fillna)

    # 최종 dtype 지정
    if kind == "int":
        # 정수 변환 시 소수점이 있으면 pandas가 그대로 두기도 함 → astype에서 오류 회피
        try:
            s2 = s2.astype("int64")
        except Exception:
            # 정수 변환이 곧바로 안 되는 데이터는 반올림 후 변환(보수적)
            s2 = s2.round().astype("int64", errors="ignore")
    else:
        s2 = s2.astype("float64", errors="ignore")

    return s2


def coerce_numeric_cols(
    df: pd.DataFrame,
    cols: Iterable[str],
    kind: NumericKind = "float",
    fillna: float | int | None = 0
) -> pd.DataFrame:
    """
    지정한 여러 컬럼을 안전하게 숫자형으로 변환합니다.
    - 예: coerce_numeric_cols(df, ["lprice","hprice"], kind="float", fillna=0)
    - 예: coerce_numeric_cols(df, ["rank"], kind="int", fillna=0)
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = _coerce_numeric_series(out[c], kind=kind, fillna=fillna)
    return out


def strip_string_cols(
    df: pd.DataFrame,
    cols: Iterable[str]
) -> pd.DataFrame:
    """
    문자열 컬럼의 좌우 공백을 제거합니다.
    - 강한 정규화(소문자화, 특수문자 제거 등)는 별도 전처리 단계에서 수행하세요.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out
