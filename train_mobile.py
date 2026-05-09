import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CelebSmileDataset, train_transform, val_transform
from model_mobile import MultiTaskMobileNet
import os

PHASE1_CKPT = 'models/phase1_mobile.pt'  # [수정] ResNet 체크포인트와 분리

def train_model(alpha=1.5, beta=1.2, num_epochs=30, patience=8, min_smile_acc=0.70):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = CelebSmileDataset('data/processed/train', transform=train_transform)
    val_dataset   = CelebSmileDataset('data/processed/val',   transform=val_transform)

    nw = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=nw, pin_memory=True)

    model = MultiTaskMobileNet(
        num_celebs=len(train_dataset.celeb_classes),
        dropout_rate=0.5,
        freeze_backbone=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    best_score      = 0.0
    best_celeb_acc  = 0.0  # [버그수정] best 시점의 celeb_acc 별도 추적
    best_smile_acc  = 0.0  # [버그수정] best 시점의 smile_acc 별도 추적
    patience_counter = 0

    # ── Phase 1: Backbone 고정, Head만 학습 ──────────────────────────
    PHASE1_EPOCHS = 15

    if os.path.exists(PHASE1_CKPT):
        print(f"\nPhase 1 체크포인트 발견 ({PHASE1_CKPT}) → Phase 1 스킵")
        model.load_state_dict(torch.load(PHASE1_CKPT, map_location=device))
    else:
        print("\n===== Phase 1: Head Only (backbone frozen) =====")
        optimizer = optim.Adam([
            {"params": model.shared.parameters(),     "lr": 3e-4},
            {"params": model.celeb_head.parameters(), "lr": 3e-4},
            {"params": model.smile_head.parameters(), "lr": 3e-4},
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        for epoch in range(PHASE1_EPOCHS):
            model.train()
            total_train_loss = 0.0

            for images, celeb_labels, smile_labels in train_loader:
                images       = images.to(device)
                celeb_labels = celeb_labels.to(device)
                smile_labels = smile_labels.to(device)

                optimizer.zero_grad()
                celeb_preds, smile_preds = model(images)

                loss_celeb = criterion(celeb_preds, celeb_labels)
                loss_smile = criterion(smile_preds, smile_labels)
                loss = (alpha * loss_celeb) + (beta * loss_smile)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            celeb_correct, smile_correct, total_val_loss = 0, 0, 0.0
            total_samples = len(val_dataset)

            with torch.no_grad():
                for images, celeb_labels, smile_labels in val_loader:
                    images       = images.to(device)
                    celeb_labels = celeb_labels.to(device)
                    smile_labels = smile_labels.to(device)

                    c_preds, s_preds = model(images)
                    loss = (alpha * criterion(c_preds, celeb_labels)) + \
                           (beta  * criterion(s_preds, smile_labels))
                    total_val_loss += loss.item()

                    celeb_correct += (torch.argmax(c_preds, 1) == celeb_labels).sum().item()
                    smile_correct += (torch.argmax(s_preds, 1) == smile_labels).sum().item()

            val_celeb_acc = celeb_correct / total_samples
            val_smile_acc = smile_correct / total_samples
            avg_val_loss  = total_val_loss / len(val_loader)

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"[Phase1] Epoch [{epoch+1}/{PHASE1_EPOCHS}] - "  # [버그수정] 실제 에폭 수 변수로 통일
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Celeb Acc: {val_celeb_acc:.4f} | Smile Acc: {val_smile_acc:.4f} | LR: {current_lr:.6f}")

        torch.save(model.state_dict(), PHASE1_CKPT)
        print(f"Phase 1 체크포인트 저장 완료: {PHASE1_CKPT}")

    # ── Phase 2: Backbone 해제, 전체 Fine-tuning ─────────────────────
    print("\n===== Phase 2: Full Fine-tuning (backbone unfrozen) =====")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {"params": model.backbone.parameters(),   "lr": 1e-5},
        {"params": model.shared.parameters(),     "lr": 1e-4},
        {"params": model.celeb_head.parameters(), "lr": 1e-4},
        {"params": model.smile_head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for images, celeb_labels, smile_labels in train_loader:
            images       = images.to(device)
            celeb_labels = celeb_labels.to(device)
            smile_labels = smile_labels.to(device)

            optimizer.zero_grad()
            celeb_preds, smile_preds = model(images)

            loss_celeb = criterion(celeb_preds, celeb_labels)
            loss_smile = criterion(smile_preds, smile_labels)
            loss = (alpha * loss_celeb) + (beta * loss_smile)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        celeb_correct, smile_correct, total_val_loss = 0, 0, 0.0
        total_samples = len(val_dataset)

        with torch.no_grad():
            for images, celeb_labels, smile_labels in val_loader:
                images       = images.to(device)
                celeb_labels = celeb_labels.to(device)
                smile_labels = smile_labels.to(device)

                c_preds, s_preds = model(images)
                loss = (alpha * criterion(c_preds, celeb_labels)) + \
                       (beta  * criterion(s_preds, smile_labels))
                total_val_loss += loss.item()

                celeb_correct += (torch.argmax(c_preds, 1) == celeb_labels).sum().item()
                smile_correct += (torch.argmax(s_preds, 1) == smile_labels).sum().item()

        val_celeb_acc = celeb_correct / total_samples
        val_smile_acc = smile_correct / total_samples
        avg_val_loss  = total_val_loss / len(val_loader)
        current_score = val_celeb_acc + val_smile_acc

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[1]['lr']

        print(f"[Phase2] Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Celeb Acc: {val_celeb_acc:.4f} | Smile Acc: {val_smile_acc:.4f} | LR: {current_lr:.6f}")

        if current_score > best_score and val_smile_acc >= min_smile_acc:
            best_score     = current_score
            best_celeb_acc = val_celeb_acc  # [버그수정] best 시점 값 캡처
            best_smile_acc = val_smile_acc  # [버그수정] best 시점 값 캡처

            model_filename = f'best_model_mobile_a{alpha}_b{beta}.pt'  # [수정] ResNet 파일과 분리
            save_path = os.path.join(save_dir, model_filename)

            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"-> Best Model 저장됨 (경로: {save_path} | Score: {best_score:.4f})")

        else:
            patience_counter += 1
            if val_smile_acc < min_smile_acc:
                print(f"-> 저장 안 함: Smile Acc 미달 ({val_smile_acc:.4f} < {min_smile_acc}) | patience {patience_counter}/{patience}")
            else:
                print(f"-> 저장 안 함: Score 미갱신 ({current_score:.4f} <= {best_score:.4f}) | patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early Stopping triggered.")
            break

    # [버그수정] return 추가 - grid search에서 결과값 수신에 필요
    return {
        'best_score':     best_score,
        'best_celeb_acc': best_celeb_acc,
        'best_smile_acc': best_smile_acc,
    }

if __name__ == "__main__":
    train_model(alpha=1.5, beta=1.2)