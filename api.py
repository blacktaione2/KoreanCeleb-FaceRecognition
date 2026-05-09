from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from PIL import Image
import io

# ── 설정 ──────────────────────────────────────────────
CELEB_NAMES     = ["로제", "박보검", "박보영", "아이유", "임영웅",
                   "장원영", "제니", "차은우", "최시원", "카리나"]
RESNET_PATH     = "models/best_model_a1.6_b0.7.pt"
MOBILENET_PATH  = "models/best_model_mobile_a1.5_b1.2.pt"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 모델 정의 ──────────────────────────────────────────
class MultiTaskResNet(nn.Module):
    def __init__(self, num_celebs=10):
        super().__init__()
        base          = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # FC 제외

        self.shared = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.celeb_head = nn.Linear(512, num_celebs)
        self.smile_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.shared(x)
        return self.celeb_head(x), self.smile_head(x)


class MultiTaskMobileNet(nn.Module):
    def __init__(self, num_celebs=10):
        super().__init__()
        base          = models.mobilenet_v2(pretrained=False)
        self.backbone = base.features                                # FC 제외

        self.shared = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.celeb_head = nn.Linear(512, num_celebs)
        self.smile_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)   # Flatten (1280)
        x = self.shared(x)
        return self.celeb_head(x), self.smile_head(x)

# ── 모델 & MTCNN 로드 ──────────────────────────────────
print("모델 로딩 중...")
mtcnn = MTCNN(image_size=224, margin=40, device=DEVICE)

resnet_model = MultiTaskResNet(num_celebs=10).to(DEVICE)
resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
resnet_model.eval()

mobilenet_model = MultiTaskMobileNet(num_celebs=10).to(DEVICE)
mobilenet_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
mobilenet_model.eval()

print(f"모델 로딩 완료 (device: {DEVICE})")

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
        "celeb": {
            "name":       celeb_name,
            "confidence": celeb_conf
        },
        "smile": {
            "label":      smile_label,
            "confidence": smile_conf
        },
        "top3_candidates": top3
    }
