import os
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageFont, ImageDraw, Image          # PIL 한글 렌더링용
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from dataset import CelebSmileDataset, val_transform
from model import MultiTaskResNet
import numpy as np

# ── 한글 폰트 설정 (matplotlib) ──────────────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'        # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False           # 마이너스 기호 깨짐 방지

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"            # PIL용 한글 폰트 경로

def imread_korean(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

def imwrite_korean(path, img):
    ext = os.path.splitext(path)[1]
    result, encoded = cv2.imencode(ext, img)
    if result:
        encoded.tofile(path)

# cv2.putText 대체 - PIL로 한글 텍스트 렌더링
def put_text_korean(img, text, pos, font_size=18, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

TEST_DIR = 'data/processed/test'
OUTPUT_DIR = 'classification_report'
MODEL_PATH = 'models/best_model_a1.6_b0.7.pt' # 실험 후 최적 가중치 파일명으로 변경

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = CelebSmileDataset(TEST_DIR, transform=val_transform)
    model = MultiTaskResNet(num_celebs=len(test_dataset.celeb_classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    categories = ['1_All_Correct', '2_Wrong_Name', '3_Wrong_Smile', '4_All_Wrong']
    for cat in categories:
        os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)
        
    y_true_celeb, y_pred_celeb = [], []
    y_true_smile, y_pred_smile = [], []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_path, true_celeb_idx, true_smile_idx = test_dataset.files[idx]
            img_tensor, _, _ = test_dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            c_out, s_out = model(img_tensor)
            pred_celeb_idx = torch.argmax(c_out, 1).item()
            pred_smile_idx = torch.argmax(s_out, 1).item()
            
            y_true_celeb.append(true_celeb_idx)
            y_pred_celeb.append(pred_celeb_idx)
            y_true_smile.append(true_smile_idx)
            y_pred_smile.append(pred_smile_idx)
            
            # 4단계 분류 저장
            name_match = (pred_celeb_idx == true_celeb_idx)
            smile_match = (pred_smile_idx == true_smile_idx)
            
            if name_match and smile_match: target_cat = '1_All_Correct'
            elif not name_match and smile_match: target_cat = '2_Wrong_Name'
            elif name_match and not smile_match: target_cat = '3_Wrong_Smile'
            else: target_cat = '4_All_Wrong'
            
            orig_img = imread_korean(img_path)
            true_celeb_name = test_dataset.celeb_classes[true_celeb_idx]
            pred_celeb_name = test_dataset.celeb_classes[pred_celeb_idx]
            true_smile = "Smile" if true_smile_idx == 1 else "Neutral"
            pred_smile = "Smile" if pred_smile_idx == 1 else "Neutral"
            
            # 이미지 상단에 텍스트 표기
            orig_img = put_text_korean(orig_img, f"GT: {true_celeb_name}_{true_smile}", (10, 5),  color=(0, 255, 0))
            orig_img = put_text_korean(orig_img, f"PR: {pred_celeb_name}_{pred_smile}", (10, 25), color=(0, 0, 255))
            
            save_name = f"{os.path.basename(img_path)}"
            imwrite_korean(os.path.join(OUTPUT_DIR, target_cat, save_name), orig_img)

    # Confusion Matrix 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm_celeb = confusion_matrix(y_true_celeb, y_pred_celeb)
    sns.heatmap(cm_celeb, annot=True, fmt='d', xticklabels=test_dataset.celeb_classes, yticklabels=test_dataset.celeb_classes)
    plt.title('Celeb Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    cm_smile = confusion_matrix(y_true_smile, y_pred_smile)
    sns.heatmap(cm_smile, annot=True, fmt='d', xticklabels=['Neutral', 'Smile'], yticklabels=['Neutral', 'Smile'])
    plt.title('Smile Confusion Matrix')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))
    print("Confusion Matrix 저장 완료.")

def evaluate_opencv():
    print("\n--- OpenCV Haar Cascade 비교 검증 ---")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    test_dataset = CelebSmileDataset(TEST_DIR, transform=val_transform)
    y_true, y_pred = [], []
    
    for idx in range(len(test_dataset)):
        img_path, _, true_smile_idx = test_dataset.files[idx]
        img = imread_korean(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        y_true.append(true_smile_idx)
        
        # OpenCV 감지 로직
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        smile_detected = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            if len(smiles) > 0:
                smile_detected = 1
                break
        y_pred.append(smile_detected)
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    print(f"OpenCV Smile Detection -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

if __name__ == "__main__":
    evaluate_model()
    evaluate_opencv()
    
