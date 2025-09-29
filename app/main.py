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
from ml.feature_set_v1 import build_features_v1
from ml.clean import ensure_columns, coerce_numeric_cols, strip_string_cols

# -----------------------
# 설정
# -----------------------
ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트

MODEL_DIR =  ROOT / "artifacts" / "final_model_ccc2987d"

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
    exp_id: str = "ccc2987d"

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
    return {"exp_id": "ccc2987d", "model_path": MODEL_PATH.as_posix()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipe, X_cols = _load_model_and_features()

    # 입력 -> 피처 DF
    feats = _items_to_features([i.model_dump() for i in req.items])

    # 학습 때의 X 컬럼만 사용 (순서 유지)
    missing = [c for c in X_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing[:10]}...")

    X = feats[X_cols].copy()

    # 예측
    y_hat = pipe.predict(X)
    if req.clip_to_range:
        y_clip = np.clip(y_hat, 1, 200)
    else:
        y_clip = y_hat

    results = []
    for i, row in enumerate(req.items):
        results.append(PredictResponseItem(
            pred_rank=float(y_hat[i]),
            pred_rank_clipped=float(y_clip[i]),
            query=row.query,
            title=row.title
        ))

    return PredictResponse(results=results, n=len(results))

# === 실행 엔트리포인트 추가 ===
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)