import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict, Counter

# 지원하는 이미지 확장자 정의
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

class CelebSmileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        if transform is None:
            raise ValueError("transform이 지정되지 않았습니다. train_transform 또는 val_transform을 전달하세요.")

        self.root_dir = root_dir
        self.transform = transform
        self.files = []

        self.celeb_classes = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]

        if not self.celeb_classes:
            raise FileNotFoundError(f"root_dir에 연예인 폴더가 없습니다: {root_dir}")

        self.celeb_to_idx = {cls_name: i for i, cls_name in enumerate(self.celeb_classes)}
        self.smile_to_idx = {'neutral': 0, 'smile': 1}

        stats = defaultdict(lambda: {'smile': 0, 'neutral': 0})

        for celeb in self.celeb_classes:
            celeb_path = os.path.join(root_dir, celeb)
            for status in ['smile', 'neutral']:
                status_path = os.path.join(celeb_path, status)

                if not os.path.isdir(status_path):
                    continue

                for entry in os.scandir(status_path):
                    if entry.is_file() and entry.name.lower().endswith(VALID_EXT):
                        self.files.append((
                            entry.path,
                            self.celeb_to_idx[celeb],
                            self.smile_to_idx[status]
                        ))
                        stats[celeb][status] += 1

        if not self.files:
            raise RuntimeError("로드된 이미지가 0장입니다. 경로 및 폴더 구조를 확인하세요.")

        self.num_celeb_classes = len(self.celeb_classes)
        self.num_smile_classes = 2

        self.sample_weights = self._calculate_weights()

        self._print_summary(stats)

    def _calculate_weights(self):
        label_pairs = [(f[1], f[2]) for f in self.files]
        count = Counter(label_pairs)
        weights = [1.0 / count[(f[1], f[2])] for f in self.files]
        return torch.tensor(weights, dtype=torch.float)

    def get_sample_weights(self):
        return self.sample_weights

    def _print_summary(self, stats):
        print(f"\n[Dataset Summary] root: {self.root_dir}")
        total = 0
        for celeb in self.celeb_classes:
            s, n = stats[celeb]['smile'], stats[celeb]['neutral']
            print(f"  {celeb}: Smile({s}), Neutral({n}), Total({s + n})")
            total += (s + n)
        print(f"\n  Total images   : {total}")
        print(f"  Celeb classes : {self.num_celeb_classes}")
        print(f"  Smile classes : {self.num_smile_classes}\n")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        max_retries = len(self.files)

        for attempt in range(max_retries):
            current_idx = (idx + attempt) % len(self.files)
            img_path, celeb_label, smile_label = self.files[current_idx]

            try:
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                return image, celeb_label, smile_label

            except Exception as e:
                print(f"[ERROR] 이미지 로드 실패 (파일: {img_path}) | 사유: {e}")

        raise RuntimeError(f"데이터셋 내 유효한 이미지가 없습니다. ({max_retries}회 시도)")

    def __repr__(self):
        return f"CelebSmileDataset(images={len(self.files)}, celebs={self.num_celeb_classes})"


# [수정] Augmentation 강화 - 과적합 방지
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# 검증/테스트용: 랜덤 요소 제거
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])