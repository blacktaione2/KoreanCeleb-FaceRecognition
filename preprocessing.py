import os
import cv2
import numpy as np
import shutil
import torch
from facenet_pytorch import MTCNN
from PIL import Image

SPLIT_DIR = "data/split"
PROCESSED_DIR = "data/processed"
MANUAL_CHECK_DIR = "data/manual_check"

MIN_FACE_PX = 80        # 얼굴 최소 픽셀
MIN_FACE_RATIO = 0.1    # 얼굴/이미지 면적 비율 최소값
CONFIDENCE_THRESH = 0.90
PADDING = 0.25


def imread_korean(path):
    img_array = np.fromfile(path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def imwrite_korean(path, img):
    _, ext = os.path.splitext(path)
    result, encoded_img = cv2.imencode(ext, img)
    if result:
        with open(path, mode='w+b') as f:
            encoded_img.tofile(f)


def save_to_manual_check(fail_dir, img_name, img):
    os.makedirs(fail_dir, exist_ok=True)
    imwrite_korean(os.path.join(fail_dir, img_name), img)


def process_images():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    if os.path.exists(MANUAL_CHECK_DIR):
        shutil.rmtree(MANUAL_CHECK_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("MTCNN device:", device)

    mtcnn = MTCNN(keep_all=True, device=device)

    stats = {}

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(SPLIT_DIR, split)
        if not os.path.exists(split_path):
            continue

        # ↓ listdir → scandir (is_dir() 추가 시스템 콜 없음)
        for celeb_entry in os.scandir(split_path):
            if not celeb_entry.is_dir():
                continue

            celeb = celeb_entry.name
            celeb_path = celeb_entry.path
            key = f"{split}/{celeb}"
            stats[key] = {'smile': 0, 'neutral': 0, 'manual_check': 0}

            for status in ['smile', 'neutral']:
                status_path = os.path.join(celeb_path, status)
                if not os.path.exists(status_path):
                    continue

                target_dir = os.path.join(PROCESSED_DIR, split, celeb, status)
                fail_dir = os.path.join(MANUAL_CHECK_DIR, split, celeb, status)

                # ↓ listdir → scandir (is_file() 추가 시스템 콜 없음, entry.path로 join 불필요)
                for img_entry in os.scandir(status_path):
                    if not img_entry.is_file():
                        continue

                    img_name = img_entry.name
                    img_path = img_entry.path
                    img = imread_korean(img_path)
                    if img is None:
                        continue

                    img_h, img_w = img.shape[:2]
                    img_area = img_w * img_h

                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    boxes, probs = mtcnn.detect(pil_img)

                    # 얼굴 미감지
                    if boxes is None or probs is None:
                        save_to_manual_check(fail_dir, img_name, img)
                        stats[key]['manual_check'] += 1
                        continue

                    # 가장 큰 얼굴 선택
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    idx = int(np.argmax(areas))

                    if probs[idx] < CONFIDENCE_THRESH:
                        save_to_manual_check(fail_dir, img_name, img)
                        stats[key]['manual_check'] += 1
                        continue

                    x1, y1, x2, y2 = [int(b) for b in boxes[idx]]
                    w, h = x2 - x1, y2 - y1

                    # 얼굴 너무 작음
                    if w < MIN_FACE_PX or h < MIN_FACE_PX:
                        save_to_manual_check(fail_dir, img_name, img)
                        stats[key]['manual_check'] += 1
                        continue

                    # 얼굴 비율 미달
                    if (w * h) / img_area < MIN_FACE_RATIO:
                        save_to_manual_check(fail_dir, img_name, img)
                        stats[key]['manual_check'] += 1
                        continue

                    # 패딩 적용 크롭
                    pad_x = int(w * PADDING)
                    pad_y = int(h * PADDING)
                    nx1 = max(0, x1 - pad_x)
                    ny1 = max(0, y1 - pad_y)
                    nx2 = min(img_w, x2 + pad_x)
                    ny2 = min(img_h, y2 + pad_y)

                    face_crop = img[ny1:ny2, nx1:nx2]
                    if face_crop.size == 0:
                        save_to_manual_check(fail_dir, img_name, img)
                        stats[key]['manual_check'] += 1
                        continue

                    os.makedirs(target_dir, exist_ok=True)
                    imwrite_korean(os.path.join(target_dir, img_name), face_crop)
                    stats[key][status] += 1

    print("\n[전처리 후 데이터 통계]")
    for key, counts in stats.items():
        total = counts['smile'] + counts['neutral']
        print(
            f"{key}: "
            f"Smile({counts['smile']}), "
            f"Neutral({counts['neutral']}), "
            f"Total({total}), "
            f"ManualCheck({counts['manual_check']})"
        )


if __name__ == "__main__":
    process_images()