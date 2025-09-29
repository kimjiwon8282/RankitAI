# ml/loaders.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
from pymongo import MongoClient
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.clean import coerce_numeric_cols, ensure_columns


# -----------------------------------------------------------------------------
# 환경설정 (.env)
# -----------------------------------------------------------------------------
class Settings(BaseSettings):
    MONGODB_URI: str
    MONGODB_DB: str = "rankit"
    MONGODB_COLL_SHOP_TREND: str = "shop_search_trend"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()


def _get_collection():
    """설정(.env)을 기반으로 Mongo 컬렉션 핸들을 반환"""
    client = MongoClient(settings.MONGODB_URI)
    return client[settings.MONGODB_DB][settings.MONGODB_COLL_SHOP_TREND]


# -----------------------------------------------------------------------------
# 로더: shop_search_trend 문서의 items(Top200)를 롱 형태 DataFrame으로 변환
#  - 현재 스키마: 최상위에 _id(=query 문자열), items[...], callAt, _class
#  - query_filter가 주어지면 _id로 필터링
# -----------------------------------------------------------------------------
def fetch_shop_search_trend_long(
    query_filter: Optional[Iterable[str]] = None,
    limit_docs: Optional[int] = None,
) -> pd.DataFrame:
    """
    MongoDB에서 문서를 읽고, 각 문서의 items 배열을 행으로 펼쳐 DataFrame 반환.

    Returns
    -------
    pd.DataFrame
      예시 컬럼:
      ['query','doc_id','callAt','callAt_dt','title','link','image',
       'lprice','hprice','rank','mallName','productId','productType',
       'brand','maker','category1','category2','category3','category4', ...]
    """
    coll = _get_collection()

    # _id(=query 문자열)로 필터링
    mongo_filter: Dict[str, Any] = {}
    if query_filter:
        mongo_filter["_id"] = {"$in": list(query_filter)}

    cursor = coll.find(mongo_filter, projection=None)
    if limit_docs is not None:
        cursor = cursor.limit(int(limit_docs))

    docs = list(cursor)
    if not docs:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for d in docs:
        # _id가 곧 query 문자열
        try:
            q = str(d.get("_id")) if d.get("_id") is not None else None
        except Exception:
            q = None

        call_at = d.get("callAt")
        try:
            doc_id = str(d.get("_id")) if d.get("_id") is not None else None
        except Exception:
            doc_id = None

        # items: 최상위 리스트(필수). 래핑 구조 대비(예: payload.items)
        items = d.get("items", [])
        if not isinstance(items, list):
            for wrap_key in ("payload", "result", "data"):
                w = d.get(wrap_key)
                if isinstance(w, dict) and isinstance(w.get("items"), list):
                    items = w["items"]
                    break
        if not isinstance(items, list):
            continue  # 방어

        for it in items:
            row = {"query": q, "doc_id": doc_id, "callAt": call_at}
            if isinstance(it, dict):
                row.update(it)
            rows.append(row)

    df = pd.DataFrame(rows)

    # (안전) 필수 컬럼 보장
    required_cols = [
        "query", "doc_id", "callAt",
        "title", "link", "image",
        "lprice", "hprice", "rank",
        "mallName", "productId", "productType",
        "brand", "maker",
        "category1", "category2", "category3", "category4",
    ]
    df = ensure_columns(df, required_cols, default="")

    # 숫자형 안전 변환
    df = coerce_numeric_cols(df, ["lprice", "hprice"], kind="float", fillna=0)
    df = coerce_numeric_cols(df, ["rank"], kind="int", fillna=0)

    # 문자열 가벼운 트림
    for c in [
        "query","title","link","image","mallName","brand","maker","productType",
        "category1","category2","category3","category4","productId"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # callAt → datetime(ms, UTC)
    if "callAt" in df.columns:
        try:
            df["callAt_dt"] = pd.to_datetime(df["callAt"], unit="ms", utc=True)
        except Exception:
            pass

    return df.reset_index(drop=True)
