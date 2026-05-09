import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

RAW_DIR = "data/raw"
SPLIT_DIR = "data/split"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def stratified_split():
    # 0. RAW_DIR 존재 확인
    if not os.path.exists(RAW_DIR):
        print(f"[오류] 소스 디렉토리가 없습니다: {RAW_DIR}")
        return

    if os.path.exists(SPLIT_DIR):
        print("기존 split 데이터 삭제 중...")
        shutil.rmtree(SPLIT_DIR)

    files_info = []

    # ↓ listdir + isdir 중복 체크 → scandir (is_dir() 추가 시스템 콜 없음)
    for celeb_entry in os.scandir(RAW_DIR):
        if not celeb_entry.is_dir():
            continue

        celeb = celeb_entry.name
        celeb_path = celeb_entry.path

        for status in ['smile', 'neutral']:
            status_path = os.path.join(celeb_path, status)
            if not os.path.exists(status_path) or not os.path.isdir(status_path):
                continue

            # ↓ listdir → scandir (is_file() 추가 시스템 콜 없음, entry.path로 join 불필요)
            img_entries = [e for e in os.scandir(status_path) if e.is_file()]
            if not img_entries:
                print(f"[정보] 빈 폴더 스킵: {status_path}")
                continue

            for img_entry in img_entries:
                if not img_entry.name.lower().endswith(VALID_EXT):
                    continue

                if img_entry.name.startswith("aug_"):
                    # print(f"[경고] 증강 파일 제외: {img_entry.name}") # 너무 많을 수 있어 필요시 해제
                    continue

                img_path = img_entry.path
                # (파일경로, 계층용라벨, 연예인명, 상태)
                files_info.append((img_path, f"{celeb}_{status}", celeb, status))

    if not files_info:
        print("데이터가 없습니다. 수동 분류(smile/neutral)가 완료되었는지 확인하세요.")
        return

    X = [info[0] for info in files_info]
    y_stratify = [info[1] for info in files_info]

    # 클래스 분포 확인 및 필터링
    counter = Counter(y_stratify)
    print("\n--- 클래스별 데이터 분포 ---")

    # 8:1:1 분할을 위해 최소 데이터 개수 체크 (최소 10장은 있어야 안정적 분할 가능)
    insufficient_classes = [k for k, v in counter.items() if v < 10]
    if insufficient_classes:
        print(f"[오류] 데이터가 부족한 클래스가 있습니다: {insufficient_classes}")
        print("최소 10장 이상의 사진이 필요합니다.")
        return

    for k, v in sorted(counter.items()):
        print(f"  {k}: {v}장")

    # 8:1:1 분할 (전체에서 20%를 먼저 떼어내고, 그걸 다시 반으로 나눔)
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_stratify,
            test_size=0.1,
            stratify=y_stratify,
            random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=1 / 9,
            stratify=y_temp,
            random_state=42
        )
    except ValueError as e:
        print(f"[분할 실패] 데이터 분포 오류: {e}")
        return

    def copy_files(file_list, split_name):
        print(f"\n{split_name} 복사 중... ({len(file_list)}장)")
        for file_path in file_list:
            # os.path를 이용한 안전한 파싱
            status = os.path.basename(os.path.dirname(file_path))
            celeb = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

            target_dir = os.path.join(SPLIT_DIR, split_name, celeb, status)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(target_dir, os.path.basename(file_path)))

    copy_files(X_train, 'train')
    copy_files(X_val, 'val')
    copy_files(X_test, 'test')

    print("\n" + "="*30)
    print("최종 분할 결과:")
    print(f"  Train: {len(X_train)}장")
    print(f"  Val:   {len(X_val)}장")
    print(f"  Test:  {len(X_test)}장")
    print("="*30)

if __name__ == "__main__":
    stratified_split()