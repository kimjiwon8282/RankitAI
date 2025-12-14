# 🤖 Rankit – Product Ranking Prediction AI Model

SmartStore 상품 데이터를 기반으로  
**검색어(Query) 단위의 상대적 상품 순위를 예측하는 AI 모델 서버**입니다.

- Backend API와 분리된 구조
- Learning to Rank(LTR) 모델 기반
- FastAPI를 통한 실시간 추론 제공

---

## 1. Problem & Approach

상품 랭킹 문제는  
- 단순한 점수 예측(Regression)이 아닌
- **동일 검색어 내에서의 상대적 순위**가 핵심입니다.

👉 이에 따라 본 프로젝트는  
**Learning to Rank (LambdaRank)** 접근 방식을 채택하여  
NDCG 지표 최적화를 목표로 설계되었습니다.

---

## 2. Data Engineering

- **Data Source**  
  - MongoDB (네이버 API 기반 수집 데이터)
  - 검색어별 Top-N 상품 데이터

- **Preprocessing**
  - HTML 태그 제거
  - 가격 정합성 보정 (`lprice`, `hprice`)
  - 결측치 처리 및 데이터 정제

---

## 3. Feature Engineering

- **Text Similarity**
  - 문자 N-gram Cosine Similarity
  - Token 기반 Jaccard / Overlap
  - Token LCS(Longest Common Subsequence)

- **Price Relativization**
  - 검색어 그룹 내 중앙값 대비 가격 비율
  - Z-score 기반 상대적 가격 위치

- **Structured Matching**
  - 정규식을 활용한 수치·단위 정규화
  - 예: `1.5kg` ↔ `1500g`

---

## 4. Model Training & Evaluation

- **Framework**
  - Scikit-learn Pipeline
  - ColumnTransformer 기반 전처리 통합
  - Training / Serving Skew 방지

- **Data Split**
  - `GroupShuffleSplit`
  - 검색어(query) 단위 분리로 Data Leakage 방지

- **Model**
  - LightGBM `LGBMRanker`
  - Learning to Rank (LambdaRank, Listwise)

- **Evaluation Metrics**
  - NDCG@k
  - Spearman Correlation
  - R² Score

- **Model Persistence**
  - Joblib 기반 Pipeline 전체 직렬화 (`.pkl`)

---

## 5. Model Serving

- **API Framework**
  - FastAPI

- **Validation**
  - Pydantic 기반 Request / Response 스키마 정의

- **Inference**
  - `/predict` 엔드포인트를 통한 실시간 순위 예측

- **Server**
  - Uvicorn (ASGI)

---

## 6. Tech Stack

- **Language**: Python  
- **ML**: Scikit-learn, LightGBM  
- **Data**: Pandas, NumPy, MongoDB  
- **Serving**: FastAPI, Uvicorn  
- **MLOps**: Joblib  

---

## 7. Related Repository

- 🔗 Backend API (Spring Boot)  
  https://github.com/kimjiwon8282/RankApiClient

---

## 8. Notes

- 본 모델은 졸업작품(Rankit)의 AI 예측 모듈입니다.
- 실제 서비스 환경을 고려하여 Backend와 분리된 구조로 설계되었습니다.
