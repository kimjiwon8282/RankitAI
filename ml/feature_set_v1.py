# ml/feature_set_v1.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Optional
import re
from collections import Counter
import numpy as np
import pandas as pd

# =========================
# 0) 기본 유틸 (토큰화 등)
# =========================
_TOKEN_RE = re.compile(r"[^\w]+", flags=re.UNICODE)

def norm_text(s: str | float | None) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return _TOKEN_RE.sub(" ", str(s).strip().lower()).strip()

def toks(s: str | float | None) -> List[str]:
    return [t for t in norm_text(s).split() if t]

# =======================================
# 1) 문자 n-gram(3~5) 코사인 유사도
# =======================================
def char_ngrams(s: str, n_min: int = 3, n_max: int = 5) -> Counter:
    s2 = norm_text(s)
    if not s2:
        return Counter()
    grams = Counter()
    for n in range(n_min, n_max + 1):
        if len(s2) < n:
            continue
        for i in range(len(s2) - n + 1):
            grams[s2[i:i+n]] += 1
    return grams

def cosine_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in keys)
    if dot <= 0:
        return 0.0
    na = np.sqrt(sum(v*v for v in a.values()))
    nb = np.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))

def char_cosine_similarity(query: str, title: str) -> float:
    return cosine_from_counters(char_ngrams(query), char_ngrams(title))

# ==============================
# 2) 토큰 기반 커버리지 / 자카드
# ==============================
def token_overlap_and_jaccard(q: str, t: str) -> Tuple[float, float]:
    qs, ts = set(toks(q)), set(toks(t))
    if not qs and not ts:
        return 0.0, 0.0
    inter = len(qs & ts)
    union = len(qs | ts) if (qs or ts) else 1
    overlap = inter / max(len(qs), 1)  # |∩| / |Q|
    jacc = inter / union
    return float(overlap), float(jacc)

# ========================
# 3) LCS (토큰 단위) 비율
# ========================
def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        ai = a[i]
        for j in range(m):
            if ai == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[n][m]

def token_lcs_ratio(q: str, t: str) -> float:
    tq, tt = toks(q), toks(t)
    if not tq:
        return 0.0
    return float(lcs_len(tq, tt) / max(len(tq), 1))

# =======================================
# 4) 숫자/단위 정규화 후 매칭 (간단 룰)
# =======================================
_UNIT_PAT = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>kg|g|mg|l|ml|cm|mm|inch|in|\"|gk|kgm)",
    flags=re.IGNORECASE
)

def _normalize_num(x: str) -> float:
    x = x.replace(",", "")
    try:
        return float(x)
    except Exception:
        return np.nan

def normalize_number_unit_pairs(s: str) -> List[Tuple[str, float]]:
    res: List[Tuple[str, float]] = []
    if not s:
        return res
    for m in _UNIT_PAT.finditer(s):
        num = _normalize_num(m.group("num"))
        unit = m.group("unit").lower()
        if np.isnan(num):
            continue
        if unit in ("kg",):
            res.append(("g", num * 1000.0))
        elif unit in ("g",):
            res.append(("g", num))
        elif unit in ("mg",):
            res.append(("g", num / 1000.0))
        elif unit in ("l",):
            res.append(("ml", num * 1000.0))
        elif unit in ("ml",):
            res.append(("ml", num))
        elif unit in ("cm",):
            res.append(("mm", num * 10.0))
        elif unit in ("mm",):
            res.append(("mm", num))
        elif unit in ("inch", "in", "\""):
            res.append(("mm", num * 25.4))
    return res

def number_unit_match_ratio(query: str, title: str, tol_ratio: float = 0.05) -> float:
    q_pairs = normalize_number_unit_pairs(norm_text(query))
    t_pairs = normalize_number_unit_pairs(norm_text(title))
    if not q_pairs or not t_pairs:
        return 0.0
    matched = 0
    used = [False] * len(t_pairs)
    for uq, vq in q_pairs:
        for j, (ut, vt) in enumerate(t_pairs):
            if used[j] or uq != ut:
                continue
            denom = max(abs(vq), 1e-9)
            if abs(vq - vt) / denom <= tol_ratio:
                matched += 1
                used[j] = True
                break
    return float(matched / max(len(q_pairs), 1))

# ==================================
# 5) 가격 파생 + 쿼리별 상대화 피처
# ==================================
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lprice = pd.to_numeric(out.get("lprice", 0), errors="coerce").fillna(0.0)
    hprice = pd.to_numeric(out.get("hprice", 0), errors="coerce").fillna(0.0)
    spread = (hprice - lprice).clip(lower=0)
    avgp   = (hprice + lprice) / 2.0
    spread_ratio = np.where(lprice > 0, spread / lprice, 0.0)

    out["log_lprice"] = np.log1p(lprice)
    out["log_hprice"] = np.log1p(hprice)
    out["price_spread"] = spread
    out["price_spread_ratio"] = spread_ratio
    out["price_avg"] = avgp

    # 쿼리별 상대화
    if "query" in out.columns:
        med_by_q = out.groupby("query")["price_avg"].transform("median")
        std_by_q = out.groupby("query")["price_avg"].transform("std").fillna(0.0)
        out["price_ratio_by_query"] = np.where(med_by_q > 0, out["price_avg"] / med_by_q, 1.0)
        out["price_z_by_query"] = np.where(std_by_q > 1e-12, (out["price_avg"] - med_by_q) / std_by_q, 0.0)
    else:
        out["price_ratio_by_query"] = 1.0
        out["price_z_by_query"] = 0.0

    return out

# ===========================================================
# 5-b) 카테고리 확장 (누설 방지 옵션 포함)
# ===========================================================
def _build_cat_path_row(row: pd.Series) -> str:
    parts = []
    for c in ("category1", "category2", "category3", "category4"):
        v = str(row.get(c, "") or "").strip()
        if v:
            parts.append(v)
    return ">".join(parts) if parts else ""

def add_category_features(
    df: pd.DataFrame,
    *,
    with_query_stats: bool = False,      # True면 (query, cat_path) 통계 생성(누설 주의)
    use_rank_in_stats: bool = False,     # True면 cat_avg_rank_by_query 생성(훈련 전용 권장)
    keep_cat_path: bool = False,         # True면 cat_path 컬럼을 유지(인코딩 직접 처리 전제)
    cat_path_as_category: bool = False   # True면 cat_path를 pandas category로 캐스팅
) -> pd.DataFrame:
    """
    서빙 기본값: with_query_stats=False, use_rank_in_stats=False, keep_cat_path=False
    훈련 시 폴드 내부에서만 with_query_stats=True를 켜서 사용(누설 금지).
    """
    out = df.copy()

    # 안전하게 결측 대비
    for c in ["category1", "category2", "category3", "category4", "brand", "maker", "title", "query"]:
        if c not in out.columns:
            out[c] = ""

    # A) 카테고리 경로/일치도
    out["cat_path"] = out.apply(_build_cat_path_row, axis=1)
    cat_tokens = out["cat_path"].astype(str).apply(toks)
    q_tokens = out["query"].astype(str).apply(toks)
    t_tokens = out["title"].astype(str).apply(toks)

    def _q_in_cat_ratio(qs: List[str], cs: List[str]) -> float:
        if not qs:
            return 0.0
        inter = len(set(qs) & set(cs))
        return float(inter / len(set(qs)))

    out["q_in_cat_path"] = [ _q_in_cat_ratio(q, c) for q, c in zip(q_tokens, cat_tokens) ]

    # exact_hit_l{1..4}
    for i, cname in enumerate(["category1", "category2", "category3", "category4"], start=1):
        lev = out[cname].astype(str).apply(norm_text)
        out[f"exact_hit_l{i}"] = [ 1 if (lv and lv in set(qt)) else 0 for lv, qt in zip(lev, q_tokens) ]

    # D) 상호작용/품질 보강
    def _any_hit(a: List[str], b: List[str]) -> int:
        return 1 if set(a) & set(b) else 0

    brand_tokens = out["brand"].astype(str).apply(toks)
    maker_tokens = out["maker"].astype(str).apply(toks)

    out["brand_in_title"] = [ _any_hit(bt, tt) for bt, tt in zip(brand_tokens, t_tokens) ]
    out["maker_in_title"] = [ _any_hit(mt, tt) for mt, tt in zip(maker_tokens, t_tokens) ]
    out["brand_in_cat"]   = [ _any_hit(bt, ct) for bt, ct in zip(brand_tokens, cat_tokens) ]
    out["maker_in_cat"]   = [ _any_hit(mt, ct) for mt, ct in zip(maker_tokens, cat_tokens) ]
    out["query_title_cat_consistency"] = [
        1 if (set(qt) & set(tt) & set(ct)) else 0
        for qt, tt, ct in zip(q_tokens, t_tokens, cat_tokens)
    ]

    # B) 카테고리 기반 가격 상대화 (cat_path 기준)
    if "price_avg" not in out.columns:
        lprice = pd.to_numeric(out.get("lprice", 0), errors="coerce").fillna(0.0)
        hprice = pd.to_numeric(out.get("hprice", 0), errors="coerce").fillna(0.0)
        out["price_avg"] = (lprice + hprice) / 2.0

    med_by_cat = out.groupby("cat_path")["price_avg"].transform("median")
    std_by_cat = out.groupby("cat_path")["price_avg"].transform("std").fillna(0.0)
    out["price_ratio_by_cat"] = np.where(med_by_cat > 0, out["price_avg"] / med_by_cat, 1.0)
    out["price_z_by_cat"] = np.where(std_by_cat > 1e-12, (out["price_avg"] - med_by_cat) / std_by_cat, 0.0)

    # C) 쿼리 내 카테고리 통계 (옵션: 훈련 폴드 내부에서만 사용)
    if with_query_stats:
        counts_q = out.groupby("query")["cat_path"].transform("count").clip(lower=1)
        counts_qc = out.groupby(["query", "cat_path"])["cat_path"].transform("count")
        out["cat_share_by_query"] = (counts_qc / counts_q).astype(float)
        if use_rank_in_stats and "rank" in out.columns:
            mean_rank_qc = out.groupby(["query", "cat_path"])["rank"].transform("mean")
            out["cat_avg_rank_by_query"] = mean_rank_qc.fillna(mean_rank_qc.median())
    # 비활성 시 컬럼 제거(일관성)
    else:
        for c in ("cat_share_by_query", "cat_avg_rank_by_query"):
            if c in out.columns:
                out.drop(columns=[c], inplace=True)

    # cat_path 유지 여부/형태
    if not keep_cat_path:
        out.drop(columns=["cat_path"], inplace=True)
    elif cat_path_as_category:
        out["cat_path"] = out["cat_path"].astype("category")

    return out

# ======================
# 6) 메인 피처 빌드 함수
# ======================
def build_features_v1(
    df: pd.DataFrame,
    *,
    with_query_stats: bool = False,      # 기본 False: 서빙/일반 평가 안전
    use_rank_in_stats: bool = False,     # True면 cat_avg_rank_by_query 추가(훈련 폴드 전용 권장)
    keep_cat_path: bool = False,         # 문자열 피처 cat_path 유지 여부 (기본 드롭)
    cat_path_as_category: bool = False   # 유지 시 카테고리형으로 캐스팅
) -> pd.DataFrame:
    """
    입력: final_prep.apply_final_prep() 결과
      (columns: query, title, lprice, hprice, mallName, brand, maker, category1..4, rank)
    출력: 원본 + 다음 피처들
      - 텍스트: sim_char_cosine, tok_overlap, tok_jaccard, tok_lcs_ratio,
                numunit_match_ratio, title_len_tok
      - 가격:   log_lprice, log_hprice, price_spread, price_spread_ratio,
                price_avg, price_ratio_by_query, price_z_by_query
      - 카테고리: q_in_cat_path, exact_hit_l1..4,
                  price_ratio_by_cat, price_z_by_cat,
                  (옵션) cat_share_by_query, cat_avg_rank_by_query,
                  brand_in_title, maker_in_title, brand_in_cat, maker_in_cat,
                  query_title_cat_consistency
      - (옵션) cat_path (문자열/카테고리) — 기본은 드롭
    """
    out = df.copy()

    # --- 텍스트 기반 피처 ---
    overlaps, jaccs, lcss = [], [], []
    charcos, numunit = [], []
    title_len_tok = []
    for q, t in zip(out["query"].astype(str), out["title"].astype(str)):
        ov, jc = token_overlap_and_jaccard(q, t)
        overlaps.append(ov)
        jaccs.append(jc)
        lcss.append(token_lcs_ratio(q, t))
        charcos.append(char_cosine_similarity(q, t))
        numunit.append(number_unit_match_ratio(q, t, tol_ratio=0.05))
        title_len_tok.append(len(toks(t)))

    out["tok_overlap"] = overlaps
    out["tok_jaccard"] = jaccs
    out["tok_lcs_ratio"] = lcss
    out["sim_char_cosine"] = charcos
    out["numunit_match_ratio"] = numunit
    out["title_len_tok"] = title_len_tok

    # --- 가격 파생/상대화 (쿼리 기준) ---
    out = add_price_features(out)

    # --- 카테고리 기반 확장 (옵션 포함) ---
    out = add_category_features(
        out,
        with_query_stats=with_query_stats,
        use_rank_in_stats=use_rank_in_stats,
        keep_cat_path=keep_cat_path,
        cat_path_as_category=cat_path_as_category,
    )

    return out

# ==========================================
# 7) (옵션) 제목 중복 축소/밸런싱 도우미
# ==========================================
def dedupe_titles_within_query(df: pd.DataFrame,
                               tie_breakers: Optional[List[Tuple[str, bool]]] = None) -> pd.DataFrame:
    out = df.copy()
    if "title_len_tok" not in out.columns:
        out["title_len_tok"] = out["title"].astype(str).apply(lambda s: len(toks(s)))
    if tie_breakers is None:
        tie_breakers = [("rank", True), ("lprice", True), ("title_len_tok", False)]
    sort_cols = ["query"] + [c for (c, _) in tie_breakers] + ["title"]
    ascending = [True] + [asc for (_, asc) in tie_breakers] + [True]
    for c in sort_cols:
        if c not in out.columns:
            out[c] = np.inf if c in ("lprice", "rank") else ""
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    out = out.drop_duplicates(subset=["query", "title"], keep="first")
    return out.reset_index(drop=True)
