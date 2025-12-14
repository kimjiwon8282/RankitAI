# ml/final_prep.py
from __future__ import annotations
import re
import html
from typing import Iterable, Tuple, Optional, List
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1) HTML 태그/엔티티 제거
# -----------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(text: str | float | None) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_text_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(strip_html)
    return out

# -----------------------------------------------------------------------------
# 2) 가격/랭크 정합성 보정
# -----------------------------------------------------------------------------
def fix_price_and_rank(df: pd.DataFrame,
                       drop_zero_price: bool = True) -> pd.DataFrame:
    out = df.copy()

    for col in ["lprice", "hprice"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "rank" not in out.columns:
        out["rank"] = np.nan
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")

    mask = (out["hprice"] <= 0) & (out["lprice"] > 0)
    out.loc[mask, "hprice"] = out.loc[mask, "lprice"]

    mask = (out["lprice"] <= 0) & (out["hprice"] > 0)
    out.loc[mask, "lprice"] = out.loc[mask, "hprice"]

    mask = (out["hprice"] > 0) & (out["lprice"] > 0) & (out["hprice"] < out["lprice"])
    l_tmp = out.loc[mask, "lprice"].copy()
    out.loc[mask, "lprice"] = out.loc[mask, "hprice"]
    out.loc[mask, "hprice"] = l_tmp

    if drop_zero_price:
        out = out[~((out["lprice"] <= 0) & (out["hprice"] <= 0))]

    out = out[(out["rank"].notna()) & (out["rank"] >= 1)]
    out["rank"] = out["rank"].astype(int, errors="ignore")

    return out.reset_index(drop=True)

# -----------------------------------------------------------------------------
# 3) 쿼리 내 중복 제거
# -----------------------------------------------------------------------------
def dedupe_within_query(df: pd.DataFrame,
                        query_col: str = "query",
                        product_col: str = "productId",
                        tie_breakers: Optional[List[Tuple[str, bool]]] = None) -> pd.DataFrame:
    out = df.copy()

    if tie_breakers is None:
        out["title_len"] = out.get("title", "").astype(str).str.len()
        tie_breakers = [("lprice", True), ("title_len", False)]

    sort_cols = [query_col, "rank"] + [c for (c, _) in tie_breakers]
    ascending = [True, True] + [asc for (_, asc) in tie_breakers]

    for c in sort_cols:
        if c not in out.columns:
            out[c] = np.inf if c in ("lprice",) else ""

    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    out = out.drop_duplicates(subset=[query_col, product_col], keep="first")

    if "title_len" in out.columns:
        out = out.drop(columns=["title_len"])

    return out.reset_index(drop=True)

# -----------------------------------------------------------------------------
# 4) 최종 전처리 (허용 컬럼 필터링 포함)  ← 카테고리 보존 추가
# -----------------------------------------------------------------------------
def apply_final_prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    학습 직전 최종 전처리:
      1) HTML 태그/엔티티 제거 (title)
      2) 가격/랭크 정합성 보정
      3) 쿼리 내 중복 제거
      4) 불필요 컬럼 삭제 → 허용 컬럼만 남김
         (category1~4 컬럼을 보존하여 이후 피처에서 활용)
    """
    out = df.copy()

    # 1) 텍스트 정리
    out = clean_text_cols(out, cols=["title"])

    # 1-b) 문자열 가벼운 트림(강한 정규화는 하지 않음)
    for c in ["mallName", "brand", "maker", "category1", "category2", "category3", "category4"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    # 2) 가격/랭크 정합성
    out = fix_price_and_rank(out, drop_zero_price=True)

    # 3) 중복 제거
    out = dedupe_within_query(out, query_col="query", product_col="productId")

    # 4) 최종 허용 컬럼만 유지  ← 카테고리 컬럼 추가
    allow_cols = [
        "query", "title",
        "lprice", "hprice",
        "mallName", "brand", "maker",
        "category1", "category2", "category3", "category4",
        "rank",
    ]
    out = out[[c for c in allow_cols if c in out.columns]]

    return out.reset_index(drop=True)
