# RankitAI (FastAPI + ML)

상품 검색 순위 예측 모델을 FastAPI 서버로 제공합니다.

## 🚀 Quickstart

### 1. 저장소 클론
```bash
git clone https://github.com/kimjiwon8282/RankitAI.git
cd RankitAI
```

### 2. 가상환경 생성 & 활성화
```bash
Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
```bash
#.env 파일을 루트에 생성하고 MongoDB 정보를 입력하세요.
#예시 (.env.example 참고):
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority
MONGODB_DB=****
MONGODB_COLL_SHOP_TREND=*****
```

### 5. 서버 실행
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
