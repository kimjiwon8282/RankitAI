# train_lgbm_ranker_search.py
from __future__ import annotations
import os, json, time, random
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, ParameterSampler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from lightgbm import LGBMRanker

# --- 프로젝트 모듈 ---
from ml.loaders import fetch_shop_search_trend_long
from ml.final_prep import apply_final_prep
from ml.feature_set_v1 import build_features_v1

ARTIFACT_DIR = "artifacts/lgbm_ranker_search"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(ARTIFACT_DIR, "lgbm_ranker_search_results.csv")

# ---------------------------
# 평가 함수들
# ---------------------------
def spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    coef, _ = spearmanr(y_true, y_pred)
    return float(coef) if np.isfinite(coef) else -1.0

def _dcg_at_k(rels: np.ndarray, k: int = 10) -> float:
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))

def ndcg_at_k(y_true_rank: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_score)
    rel = 1.0 / (1.0 + np.asarray(y_true_rank, dtype=float))
    dcg = _dcg_at_k(rel[order], k=k)
    ideal_order = np.argsort(-rel)
    idcg = _dcg_at_k(rel[ideal_order], k=k)
    return float(dcg / idcg) if idcg > 0 else 0.0

def mean_ndcg_by_query(df: pd.DataFrame, query_col: str, true_rank_col: str, score_col: str, k: int = 10) -> float:
    vals = []
    for q, g in df.groupby(query_col):
        vals.append(ndcg_at_k(g[true_rank_col].values, g[score_col].values, k=k))
    return float(np.mean(vals)) if vals else 0.0

# ---------------------------
# 데이터/피처 준비 함수 (변경 없음)
# ---------------------------
def prepare_data() -> pd.DataFrame:
    raw = fetch_shop_search_trend_long(limit_docs=None)
    prepped = apply_final_prep(raw)
    feats = build_features_v1(prepped)
    drop_leaky = [c for c in feats.columns if c in ("cat_avg_rank_by_query", "cat_share_by_query")]
    feats = feats.drop(columns=drop_leaky, errors="ignore")
    return feats

def split_train_test(feats: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = feats["query"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(feats, groups=groups, y=feats["rank"].values))
    return feats.iloc[tr_idx].copy(), feats.iloc[te_idx].copy()

def build_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], int]:
    target_col = "rank"
    candidate_cat = [c for c in ["mallName", "brand", "maker", "category1", "category2", "category3", "category4"] if c in df.columns]
    drop_cols = set([target_col, "query", "title"])
    if "cat_path" in df.columns:
        drop_cols.add("cat_path")
    X_cols = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in X_cols if (c not in candidate_cat) and np.issubdtype(df[c].dtype, np.number)]
    cat_cols = [c for c in candidate_cat if c in X_cols]
    X = df[X_cols].copy()
    Rmax = int(df[target_col].max())
    y = (Rmax + 1 - df[target_col].astype(int)).values
    return X, y, num_cols, cat_cols, Rmax

def sort_by_group_and_get_group_sizes(df: pd.DataFrame, group_name: str = "query") -> Tuple[pd.DataFrame, List[int]]:
    df_sorted = df.sort_values(group_name).reset_index(drop=True)
    sizes = df_sorted.groupby(group_name).size().tolist()
    return df_sorted, sizes

# ---------------------------
# 랜덤 그리드 서치 메인
# ---------------------------
def main(n_iter=200, random_seed=42):
    feats = prepare_data()
    train_df, test_df = split_train_test(feats, test_size=0.2, seed=random_seed)
    train_df_sorted, train_group = sort_by_group_and_get_group_sizes(train_df, "query")
    test_df_sorted, _ = sort_by_group_and_get_group_sizes(test_df, "query")
    X_tr, y_tr, num_cols, cat_cols, Rmax = build_matrix(train_df_sorted)
    X_te, y_te, _, _, _ = build_matrix(test_df_sorted)

    # 파라미터 공간 정의
    param_grid = {
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [10, 20, 40],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "n_estimators": [300, 400, 500, 600],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    }
    sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_seed))

    results = []
    for idx, params in enumerate(sampler, 1):
        ranker = LGBMRanker(
            random_state=42,
            importance_type="gain",
            objective="lambdarank",
            metric="ndcg",
            label_gain=list(range(Rmax + 2)),
            **params
        )
        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        pipe = Pipeline([
            ("prep", pre),
            ("ranker", ranker),
        ])
        pipe.fit(X_tr, y_tr, ranker__group=train_group)
        y_score = pipe.predict(X_te)
        sp = spearman_scorer(y_te, -y_score)
        r2 = float(r2_score(y_te, -y_score))
        eval_df = test_df_sorted.copy()
        eval_df["score"] = y_score
        ndcg10 = mean_ndcg_by_query(eval_df, "query", "rank", "score", k=10)
        ndcg20 = mean_ndcg_by_query(eval_df, "query", "rank", "score", k=20)
        rec = dict(params)
        rec.update({
            "spearman_using_neg_score": sp,
            "r2_using_neg_score": r2,
            "ndcg@10": ndcg10,
            "ndcg@20": ndcg20,
        })
        results.append(rec)
        print(f"[{idx}/{n_iter}] ndcg@10={ndcg10:.3f} ndcg@20={ndcg20:.3f} spearman={sp:.3f}")

        # 실험마다 결과를 누적 저장(중간 저장)
        pd.DataFrame(results).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    print("\n✅ ALL DONE!")
    print(f"결과: {RESULT_CSV}")

if __name__ == "__main__":
    main(n_iter=200)
