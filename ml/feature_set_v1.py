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
# 1) 문자 n-gram(3~5) 코사인 유사도 (pure python)
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
    # dot
    keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in keys)
    if dot <= 0:
        return 0.0
    # norms
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
    # "1,200" -> 1200.0
    x = x.replace(",", "")
    try:
        return float(x)
    except Exception:
        return np.nan

def normalize_number_unit_pairs(s: str) -> List[Tuple[str, float]]:
    """
    텍스트에서 (단위종류, 표준수치) 리스트 추출
      - 질량: g 기준 (kg->g, mg->g)
      - 부피: ml 기준 (l->ml)
      - 길이: mm 기준 (cm->mm, inch/in/"->mm)
    """
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
        else:
            # 미지 단위는 스킵
            pass
    return res

def number_unit_match_ratio(query: str, title: str, tol_ratio: float = 0.05) -> float:
    """
    쿼리/타이틀의 숫자-단위 페어가 '같은 종류(unit)'이면서
    수치가 tol_ratio(5% 기본) 이내면 매칭으로 간주.
    반환: 매칭된 쿼리 페어 비율 (matched_pairs / max(total_query_pairs,1))
    """
    q_pairs = normalize_number_unit_pairs(norm_text(query))
    t_pairs = normalize_number_unit_pairs(norm_text(title))
    if not q_pairs or not t_pairs:
        return 0.0

    matched = 0
    used = [False] * len(t_pairs)
    for uq, vq in q_pairs:
        # 같은 unit 중 5% 이내 매칭 존재?
        found = False
        for j, (ut, vt) in enumerate(t_pairs):
            if used[j]:
                continue
            if uq != ut:
                continue
            # 상대 오차 기준
            denom = max(abs(vq), 1e-9)
            if abs(vq - vt) / denom <= tol_ratio:
                matched += 1
                used[j] = True
                found = True
                break
        # 쿼리의 한 페어당 최대 1매칭만 인정
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

    # 쿼리별 중앙가/표준편차 기반 상대화
    if "query" in out.columns:
        med_by_q = out.groupby("query")["price_avg"].transform("median")
        std_by_q = out.groupby("query")["price_avg"].transform("std").fillna(0.0)
        out["price_ratio_by_query"] = np.where(med_by_q > 0, out["price_avg"] / med_by_q, 1.0)
        out["price_z_by_query"] = np.where(std_by_q > 1e-12, (out["price_avg"] - med_by_q) / std_by_q, 0.0)
    else:
        out["price_ratio_by_query"] = 1.0
        out["price_z_by_query"] = 0.0

    return out


# ======================
# 6) 메인 피처 빌드 함수
# ======================
def build_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력: final_prep.apply_final_prep() 를 통과한 DataFrame
          (columns: query, title, lprice, hprice, mallName, brand, maker, rank)
    출력: 위 컬럼 + 다음 피처들이 추가된 DataFrame
      - 텍스트: sim_char_cosine, tok_overlap, tok_jaccard, tok_lcs_ratio,
                numunit_match_ratio, title_len_tok
      - 가격:   log_lprice, log_hprice, price_spread, price_spread_ratio,
                price_avg, price_ratio_by_query, price_z_by_query
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

    # --- 가격 파생/상대화 ---
    out = add_price_features(out)

    # 학습 타깃/보조카테고리 유지(필요시)
    # out에는 최종적으로 rank, mallName/brand/maker도 남아있음

    return out


# ==========================================
# 7) (옵션) 제목 중복 축소/밸런싱 도우미
# ==========================================
def dedupe_titles_within_query(df: pd.DataFrame,
                               tie_breakers: Optional[List[Tuple[str, bool]]] = None) -> pd.DataFrame:
    """
    (query, title) 기준 완전 중복을 1건으로 축소 (기본 우선순위: rank↑, lprice↓, title_len_tok↓)
    final_prep에서 productId 중복을 제거한 뒤에도 제목이 중복될 때 사용.
    """
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
