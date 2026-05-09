# 🎭 Korean Celebrity Face Recognition + Smile Detection
**Multi-Task Learning | ResNet18 vs MobileNetV2 vs OpenCV**

한국 연예인 10인의 얼굴 인식과 미소 감지를 단일 모델로 동시에 학습하는 Multi-Task Learning 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 태스크 | 연예인 분류 (10-class) + 미소 감지 (smile / neutral) 동시 학습 |
| 비교 모델 | ResNet18 · MobileNetV2 · OpenCV Haar Cascade |
| 데이터셋 | 한국 연예인 10인, Test set 166장 기준 최종 평가 |
| 그리드 서치 | 총 82회 (ResNet 41회 + MobileNetV2 41회) |
| 학습 전략 | 2-Phase (Backbone Freeze → Full Fine-tuning) |

**대상 연예인:** 로제 · 박보검 · 박보영 · 아이유 · 임영웅 · 장원영 · 제니 · 차은우 · 최시원 · 카리나

---

## 🏆 최종 성능 (Test set 166장)

| 지표 | ResNet18 | MobileNetV2 | OpenCV Haar |
|------|----------|-------------|-------------|
| Celeb Accuracy | **72.3%** (120/166) | 65.7% (109/166) | N/A |
| Smile Accuracy | **69.9%** (116/166) | 66.9% (111/166) | 51.8% (86/166) |
| Smile Precision | 83.9% | 84.9% | 87.5% |
| Smile Recall | **56.5%** | 48.9% | 15.2% |
| Total Score | **1.422** ★ | 1.326 | — |

> OpenCV는 Precision 87.5%이지만 Recall 15.2% → 실제 미소의 84.8%를 놓침 → 딥러닝의 필요성 입증

---

## 🗂️ 프로젝트 구조

```
code/
├── data/                           # 수집 및 전처리된 이미지 데이터
├── models/                         # 학습된 모델 체크포인트 (.pt)
├── result/                         # 학습 결과 및 그래프
├── classification_report/          # ResNet18 분류 리포트
├── classification_report_mobile/   # MobileNetV2 분류 리포트
│
├── crawling_naver.py               # 네이버 이미지 크롤링
├── crawling_daum.py                # 다음 이미지 크롤링
├── initialize_data.py              # 데이터 분류 및 Stratified Split (8:1:1)
├── preprocessing.py                # MTCNN 얼굴 추출 및 전처리
├── dataset.py                      # Custom Dataset & On-the-fly 증강
├── model.py                        # Multi-Task ResNet18 아키텍처
├── model_mobile.py                 # Multi-Task MobileNetV2 아키텍처
├── train.py                        # ResNet18 2-Phase 학습
├── train_mobile.py                 # MobileNetV2 2-Phase 학습
├── grid_search_resnet.py           # ResNet18 α/β 그리드 서치
├── grid_search_mobile.py           # MobileNetV2 α/β 그리드 서치
├── evaluate.py                     # ResNet18 최종 평가 + OpenCV 벤치마크
└── evaluate_mobile.py              # MobileNetV2 최종 평가
```

---

## ⚙️ 파이프라인 (8단계)

```
① 크롤링          → crawling_naver.py / crawling_daum.py
                     (Selenium + BeautifulSoup, MD5 중복 제거)

② 분류 & 분할     → initialize_data.py
                     (Smile/Neutral 수동 라벨링, Stratified Split 8:1:1)

③ 전처리          → preprocessing.py
                     (MTCNN 얼굴 추출, Padding 20~30%, 224×224 resize)

④ 데이터셋        → dataset.py
                     (Custom Dataset, On-the-fly 증강)

⑤ 모델            → model.py / model_mobile.py
                     (Multi-Task ResNet18 / MobileNetV2)

⑥ 학습            → train.py / train_mobile.py
                     (2-Phase 학습, Early Stopping)

⑦ 그리드 서치     → grid_search_resnet.py / grid_search_mobile.py
                     (α/β 격자 탐색, Phase1 체크포인트 재사용)

⑧ 평가            → evaluate.py / evaluate_mobile.py
                     (4단계 오류 분류, OpenCV 벤치마크)
```

---

## 🧠 모델 아키텍처

```
Input (224×224)
    ↓
Backbone (ImageNet pretrained)
ResNet18 [512-dim] / MobileNetV2 [1280-dim]
    ↓
Shared Layer
Linear → LayerNorm → ReLU → Dropout(0.5)
    ↓              ↓
Celeb Head      Smile Head
Linear(512→10)  Linear(512→2)

Loss = α × CE(Celeb) + β × CE(Smile)
  ResNet18   : α=1.6, β=0.7
  MobileNetV2: α=1.5, β=1.2
```

| 구성 요소 | ResNet18 | MobileNetV2 |
|-----------|----------|-------------|
| Backbone 출력 | 512-dim | 1280-dim |
| 파라미터 수 | 11.4M | 2.9M |
| 파일 크기 | 43.7MB | 11.3MB |

---

## 📋 2-Phase 학습 전략

**Phase 1 — Backbone Frozen (15 Epoch)**
- Backbone 고정, Shared Layer + Head만 학습
- lr: 3e-4 (Adam) + ReduceLROnPlateau
- 완료 후 `phase1_resnet.pt` 저장 → 그리드 서치 재사용으로 비용 절감

**Phase 2 — Full Fine-tuning (최대 30 Epoch)**
- Backbone 해제, 전체 학습
- Backbone lr=1e-5 / Head lr=1e-4 (차등 학습률)
- Early Stopping (patience=8)
- 저장 조건: Score 갱신 **AND** smile_acc ≥ 0.70 (Smile 붕괴 방지)

---

## 🔍 하이퍼파라미터 그리드 서치

| 모델 | 1차 탐색 | 2차 세분화 | 총 실험 | 최적 조합 |
|------|---------|-----------|---------|----------|
| ResNet18 | α/β [0.5~2.0] (4×4=16회) | α [1.3~1.7] β [0.7~1.1] (5×5=25회) | 41회 | α=1.6, β=0.7 |
| MobileNetV2 | α/β [0.5~2.0] (4×4=16회) | α [1.2~1.7] β [0.8~1.5] (5×5=25회) | 41회 | α=1.5, β=1.2 |

- ResNet18: 하이퍼파라미터에 민감 (0.0 셀 다수 발생)
- MobileNetV2: 전 구간 안정적 수렴 (하이퍼파라미터 로버스트)

---

## ⚡ 추론 속도 비교

| 모델 | CPU | GPU |
|------|-----|-----|
| ResNet18 | 34.8 FPS / 28.72ms | **223.7 FPS / 4.47ms** ★ |
| MobileNetV2 | **52.5 FPS / 19.07ms** ★ | 176.8 FPS / 5.66ms |
| OpenCV Haar | 87.3 FPS / 11.45ms | 16.4 FPS / 61.04ms |

> CPU↔GPU 역전 현상: MobileNetV2의 Depthwise Conv는 CPU에서 유리하지만, GPU에서는 병렬화 효율 저하로 ResNet18에 역전됨

---

## 🛠️ 기술 스택

| 분야 | 기술 |
|------|------|
| 딥러닝 | PyTorch · ResNet18 · MobileNetV2 · MTCNN |
| 컴퓨터 비전 | OpenCV · PIL · torchvision |
| 데이터 수집 | Selenium · BeautifulSoup |
| 분석 & 시각화 | pandas · seaborn · matplotlib · numpy |

---

## 💡 핵심 인사이트

1. **성능 vs 안정성**: ResNet18 정확도 우세, MobileNetV2 하이퍼파라미터 로버스트
2. **Beta 방향성 차이**: ResNet β=0.7 vs MobileNet β=1.2 → 아키텍처별 손실 스케일 차이
3. **CPU↔GPU 속도 역전**: Depthwise Conv 구조가 CPU/GPU 환경에 따라 반대 결과
4. **OpenCV 한계 입증**: Precision 87.5%이지만 Recall 15.2% → 딥러닝 필요성 명확히 입증
5. **공통 약점**: Wrong Smile 36장은 모델과 무관한 데이터 자체의 난이도 → Focal Loss 도입 필요

---

## 🚀 실행 방법

```bash
# 1. 환경 설치
pip install -r requirements.txt

# 2. 데이터 수집
python crawling_naver.py
python crawling_daum.py

# 3. 데이터 분류 및 분할
python initialize_data.py

# 4. 전처리 (MTCNN 얼굴 추출)
python preprocessing.py

# 5. 학습
python train.py           # ResNet18
python train_mobile.py    # MobileNetV2

# 6. 그리드 서치 (선택)
python grid_search_resnet.py
python grid_search_mobile.py

# 7. 평가
python evaluate.py
python evaluate_mobile.py
```

---

## 🌐 API 서버 (FastAPI)

학습된 모델을 REST API로 서빙합니다. 이미지를 업로드하면 연예인 분류 + 미소 감지 결과를 JSON으로 반환합니다.

**서버 실행**
```bash
python api.py
```

실행 후 `http://127.0.0.1:8000/docs` 에서 브라우저로 바로 테스트 가능합니다.

**엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서버 상태 확인 |
| POST | `/predict` | 이미지 업로드 → 예측 결과 반환 |

**요청 파라미터**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `file` | 이미지 파일 | 필수 | 예측할 이미지 |
| `model_type` | string | `resnet` | `resnet` 또는 `mobilenet` 선택 |

**응답 예시**
```json
{
  "model_used": "resnet",
  "celeb": {
    "name": "아이유",
    "confidence": 91.3
  },
  "smile": {
    "label": "smile",
    "confidence": 84.7
  },
  "top3_candidates": [
    { "name": "아이유",  "confidence": 91.3 },
    { "name": "박보영", "confidence":  5.2 },
    { "name": "로제",   "confidence":  2.1 }
  ]
}
