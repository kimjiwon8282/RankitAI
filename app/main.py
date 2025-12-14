# app/main.py
from __future__ import annotations
import os
import uvicorn
from pathlib import Path
from typing import List, Optional, Literal, Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import joblib

# 당신의 전처리/피처 모듈
from fastapi.responses import JSONResponse
from ml.feature_set_v1 import build_features_v1
from ml.clean import ensure_columns, coerce_numeric_cols, strip_string_cols
from ml.loaders import fetch_shop_search_trend_long

# -----------------------
# 설정
# -----------------------
ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트

MODEL_DIR = ROOT / "artifacts" / "lgbm_ranker_final"  # <- 변경!

MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_list.json"

# -----------------------
# 스키마
# -----------------------
class Item(BaseModel):
    # 학습시 사용한 원본 입력 컬럼들 (rank 제외)
    query: str
    title: str
    lprice: Optional[float | int | str] = 0
    hprice: Optional[float | int | str] = 0
    mallName: Optional[str] = ""
    brand: Optional[str] = ""
    maker: Optional[str] = ""
    productId: Optional[str] = ""
    productType: Optional[str] = ""
    category1: Optional[str] = ""
    category2: Optional[str] = ""
    category3: Optional[str] = ""
    category4: Optional[str] = ""

class PredictRequest(BaseModel):
    items: List[Item] = Field(..., description="예측할 아이템 리스트(최소 1개)")
    clip_to_range: bool = Field(default=True, description="예측값을 [1,200]로 클리핑할지")

class PredictResponseItem(BaseModel):
    pred_rank: float
    pred_rank_clipped: float
    # 디버그/트레이싱에 유용
    query: str
    title: str
    exp_id: str = "lgbm_ranker_final"  # ← 새 모델 버전으로

class PredictResponse(BaseModel):
    results: List[PredictResponseItem]
    n: int

# -----------------------
# 앱 생성 & 모델 로딩
# -----------------------
app = FastAPI(title="RankIt Inference API", version="0.1.0")

PIPE = None
X_COLUMNS: List[str] = []

def _load_model_and_features():
    global PIPE, X_COLUMNS
    if PIPE is None:
        PIPE = joblib.load(MODEL_PATH.as_posix())
    if not X_COLUMNS:
        import json
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # feature_list.json이 리스트 또는 {"X_columns": [...]} 형태 둘 다 대응
        if isinstance(data, dict) and "X_columns" in data:
            X_COLUMNS = data["X_columns"]
        elif isinstance(data, list):
            X_COLUMNS = data
        else:
            raise ValueError("feature_list.json 형식을 알 수 없습니다.")
    return PIPE, X_COLUMNS

# -----------------------
# 작은 유틸: 입력 → 피처 DF
# -----------------------
RAW_REQUIRED = [
    "query","title","lprice","hprice","mallName","brand","maker",
    "productId","productType","category1","category2","category3","category4"
]

def _items_to_features(items: List[Dict[str, Any]]) -> pd.DataFrame:
    # 1) DataFrame 구성 & 필수 컬럼 보장
    df = pd.DataFrame(items)
    df = ensure_columns(df, RAW_REQUIRED, default="")

    # 2) 타입 정리(숫자/문자)
    df = coerce_numeric_cols(df, ["lprice","hprice"], kind="float", fillna=0)
    df = strip_string_cols(df, ["query","title","mallName","brand","maker",
                                "productId","productType",
                                "category1","category2","category3","category4"])

    # 3) 피처 생성(학습과 동일 함수)
    feats = build_features_v1(df)

    return feats

# -----------------------
# 엔드포인트
# -----------------------
@app.get("/health")
def health():
    ok = MODEL_PATH.exists() and FEATURES_PATH.exists()
    return {"ok": ok, "model_dir": MODEL_DIR.as_posix()}

@app.get("/version")
def version():
    return {"exp_id": "lgbm_ranker_final", "model_path": MODEL_PATH.as_posix()}  # <- 변경!

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipe, X_cols = _load_model_and_features()
    user_item = req.items[0]  # 사용자 1개 상품만 받는 구조
    query = user_item.query

    # 1. 경쟁상품 200개 불러오기 (query 기준)
    df_competitors = fetch_shop_search_trend_long(query_filter=[query], limit_docs=1)
    df_competitors = df_competitors.sort_values("rank").head(200)

    competitors = df_competitors[RAW_REQUIRED].to_dict(orient="records")
    
    # 2. 예외처리: 경쟁상품 너무 적으면 에러 반환
    if len(competitors) < 30:
        return JSONResponse(
            status_code=400,
            content={"detail": f"해당 검색어 '{query}'의 경쟁상품이 {len(competitors)}개로 예측이 어렵습니다."}
        )
    # 3. 사용자 입력 상품 dict로 변환 (누락 컬럼 채우기)
    user_dict = user_item.model_dump()
    for k in RAW_REQUIRED:
        if k not in user_dict:
            user_dict[k] = ""
    # 4. 합치기
    all_items = competitors + [user_dict]

    # 5. 피처 생성
    feats = _items_to_features(all_items)
    X = feats[X_cols].copy()

    # 6. 예측
    y_score = pipe.predict(X)

    # 7. 점수 내림차순 정렬 → 랭킹 계산
    order = np.argsort(-y_score)
    user_index = len(all_items) - 1
    rank = int(np.where(order == user_index)[0][0]) + 1  # 1-based rank

    # 8. 결과 생성 (요청 상품만)
    results = [PredictResponseItem(
        pred_rank=float(rank),
        pred_rank_clipped=float(rank),  # 클리핑 의미없음
        query=user_item.query,
        title=user_item.title
    )]

    return PredictResponse(results=results, n=1)

# === 실행 엔트리포인트 추가 ===
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)