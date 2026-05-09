import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models
from facenet_pytorch import MTCNN
from PIL import Image
import io
import uvicorn

# ── 설정 ──────────────────────────────────────────────
CELEB_NAMES     = ["로제", "박보검", "박보영", "아이유", "임영웅",
                   "장원영", "제니", "차은우", "최시원", "카리나"]
RESNET_PATH     = "models/best_model_a1.6_b0.7.pt"
MOBILENET_PATH  = "models/best_model_mobile_a1.5_b1.2.pt"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 모델 정의 ──────────────────────────────────────────
class MultiTaskResNet(nn.Module):
    def __init__(self, num_celebs, pretrained=False, dropout_rate=0.3, freeze_backbone=False):
        super(MultiTaskResNet, self).__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        self.celeb_head = nn.Linear(512, num_celebs)
        self.smile_head = nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)
        shared_feat = self.shared(features)
        return self.celeb_head(shared_feat), self.smile_head(shared_feat)


class MultiTaskMobileNet(nn.Module):
    def __init__(self, num_celebs, pretrained=False, dropout_rate=0.3, freeze_backbone=False):
        super(MultiTaskMobileNet, self).__init__()

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        self.celeb_head = nn.Linear(512, num_celebs)
        self.smile_head = nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)
        shared_feat = self.shared(features)
        return self.celeb_head(shared_feat), self.smile_head(shared_feat)

# ── 모델 & MTCNN 로드 ──────────────────────────────────
print(f"현재 실행 장치: {DEVICE}")
print(f"현재 작업 경로: {os.getcwd()}")

if not os.path.exists(RESNET_PATH):
    print(f"[ERROR] ResNet 모델 파일 없음: {RESNET_PATH}")
if not os.path.exists(MOBILENET_PATH):
    print(f"[ERROR] MobileNet 모델 파일 없음: {MOBILENET_PATH}")

try:
    print("MTCNN 로딩 중... (처음 실행 시 가중치 다운로드로 시간이 걸릴 수 있음)")
    mtcnn = MTCNN(image_size=224, margin=40, device=DEVICE, keep_all=False)
    print("MTCNN 로딩 완료")

    print("ResNet 모델 로딩 중...")
    resnet_model = MultiTaskResNet(num_celebs=10).to(DEVICE)
    resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    resnet_model.eval()
    print("ResNet 모델 로딩 완료")

    print("MobileNet 모델 로딩 중...")
    mobilenet_model = MultiTaskMobileNet(num_celebs=10).to(DEVICE)
    mobilenet_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
    mobilenet_model.eval()
    print("MobileNet 모델 로딩 완료")

    print("모든 모델 로딩 완료!")
except Exception as e:
    print(f"[ERROR] 모델 로딩 실패: {e}")

# ── FastAPI 앱 ─────────────────────────────────────────
app = FastAPI(title="연예인 얼굴인식 + 미소감지 API")

@app.get("/")
def root():
    return {"message": "연예인 얼굴인식 + 미소감지 API 서버 실행 중"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query(default="resnet", enum=["resnet", "mobilenet"])
):
    try:
        # 1. 모델 선택
        model = resnet_model if model_type == "resnet" else mobilenet_model

        # 2. 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 3. MTCNN 얼굴 감지
        face = mtcnn(image)

        if face is None:
            return JSONResponse(
                status_code=400,
                content={"error": "얼굴을 감지하지 못했습니다. 정면 사진을 사용해주세요."}
            )

        # 4. 모델 추론
        face_tensor = face.unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

        with torch.no_grad():
            celeb_logits, smile_logits = model(face_tensor)

        # 5. 결과 계산
        celeb_probs  = torch.softmax(celeb_logits, dim=1)[0]
        smile_probs  = torch.softmax(smile_logits, dim=1)[0]

        celeb_idx    = celeb_probs.argmax().item()
        smile_idx    = smile_probs.argmax().item()

        celeb_name   = CELEB_NAMES[celeb_idx]
        celeb_conf   = round(celeb_probs[celeb_idx].item() * 100, 1)
        smile_label  = "smile" if smile_idx == 1 else "neutral"
        smile_conf   = round(smile_probs[smile_idx].item() * 100, 1)

        # 6. Top-3 후보
        top3_idx     = celeb_probs.topk(3).indices.tolist()
        top3         = [
            {"name": CELEB_NAMES[i], "confidence": round(celeb_probs[i].item() * 100, 1)}
            for i in top3_idx
        ]

        return {
            "model_used": model_type,
            "celeb": {"name": celeb_name, "confidence": celeb_conf},
            "smile": {"label": smile_label, "confidence": smile_conf},
            "top3_candidates": top3
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── 직접 실행 ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
