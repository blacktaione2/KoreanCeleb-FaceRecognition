import os
import time
import hashlib
import cv2
import numpy as np
import requests
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

CELEBS = ["박보검"]
SAVE_DIR = "data/raw"
MIN_SIZE = 100
MAX_SCROLL = 20
TARGET_COUNT = 1000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    return webdriver.Chrome(options=chrome_options)

def get_image_hash(img_bytes):
    return hashlib.md5(img_bytes).hexdigest()

def is_valid_image(img_bytes):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        return h >= MIN_SIZE and w >= MIN_SIZE
    except:
        return False

def load_existing_hashes(celeb_dir):
    """기존 저장된 파일로 해시셋 생성 (네이버 수집본 중복 방지용)"""
    existing_hashes = set()
    if not os.path.exists(celeb_dir):
        return existing_hashes
    
    files = [f for f in os.listdir(celeb_dir) if os.path.isfile(os.path.join(celeb_dir, f))]
    print(f"기존 파일 {len(files)}장 해시 로딩 중...")
    
    for fname in files:
        fpath = os.path.join(celeb_dir, fname)
        try:
            with open(fpath, "rb") as f:
                existing_hashes.add(hashlib.md5(f.read()).hexdigest())
        except:
            continue
    
    print(f"해시 로딩 완료: {len(existing_hashes)}개")
    return existing_hashes

def crawl_daum_images(celeb_name, driver):
    celeb_dir = os.path.join(SAVE_DIR, celeb_name)
    os.makedirs(celeb_dir, exist_ok=True)

    # 기존 네이버 수집본 해시 로드 및 파일 개수 파악
    saved_hashes = load_existing_hashes(celeb_dir)
    existing_count = len([f for f in os.listdir(celeb_dir) if os.path.isfile(os.path.join(celeb_dir, f))])
    start_index = existing_count

    encoded_name = quote(celeb_name)
    url = f"https://search.daum.net/search?w=img&q={encoded_name}"
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".wrap_thumb"))
        )
    except:
        print(f"[{celeb_name}] 다음 페이지 로딩 실패")
        return

    for i in range(MAX_SCROLL):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        print(f"[{celeb_name}] 스크롤 {i+1}/{MAX_SCROLL} 완료")

    images = driver.find_elements(By.CSS_SELECTOR, ".wrap_thumb img[src]")
    print(f"[{celeb_name}] 감지된 이미지: {len(images)}장")

    saved_urls = set()
    count = 0

    for img in images:
        save_path = None
        try:
            src = img.get_attribute("src")
            if not src or src in saved_urls or not src.startswith("http"):
                continue

            saved_urls.add(src)

            res = requests.get(src, headers=HEADERS, timeout=10)
            if res.status_code != 200:
                continue

            img_bytes = res.content

            if not is_valid_image(img_bytes):
                continue

            img_hash = get_image_hash(img_bytes)
            if img_hash in saved_hashes:
                continue
            saved_hashes.add(img_hash)

            ext = "jpg"
            if "png" in src.lower(): ext = "png"
            elif "webp" in src.lower(): ext = "webp"

            save_path = os.path.join(celeb_dir, f"{celeb_name}_{start_index + count:04d}.{ext}")
            with open(save_path, "wb") as f:
                f.write(img_bytes)

            count += 1
            if count % 50 == 0:
                print(f"[{celeb_name}] {count}장 진행 중...")

        except Exception as e:
            print(f"[오류 발생] {e}")
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
            continue

    print(f"[{celeb_name}] 다음 수집 완료: {count}장 추가 (총 {start_index + count}장)")

if __name__ == "__main__":
    driver = init_driver()
    driver.maximize_window()
    try:
        for celeb in CELEBS:
            print(f"\n===== {celeb} 다음 크롤링 시작 =====")
            crawl_daum_images(celeb, driver)
    finally:
        driver.quit()
    print("\n전체 크롤링 작업이 완료되었습니다.")