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
MAX_SCROLL = 10
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

def crawl_images(celeb_name, driver):
    celeb_dir = os.path.join(SAVE_DIR, celeb_name)
    os.makedirs(celeb_dir, exist_ok=True)

    encoded_name = quote(celeb_name)
    url = f"https://search.naver.com/search.naver?where=image&sm=tab_jum&query={encoded_name}"
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".image_wrap")))
    except:
        print(f"[{celeb_name}] 페이지 로딩 실패")
        return

    # 스크롤을 전부 마친 뒤 한 번에 파싱
    for i in range(MAX_SCROLL):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        print(f"[{celeb_name}] 스크롤 {i+1}/{MAX_SCROLL} 완료")

    # 스크롤 완료 후 전체 이미지 한 번에 수집
    images = driver.find_elements(By.CSS_SELECTOR, ".image_wrap img[src]")
    print(f"[{celeb_name}] 총 감지된 이미지: {len(images)}장")

    saved_urls = set()
    saved_hashes = set()
    count = 0

    for img in images:
        if count >= TARGET_COUNT:
            break

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

            save_path = os.path.join(celeb_dir, f"{celeb_name}_{count:04d}.{ext}")
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

    print(f"[{celeb_name}] 수집 완료: 최종 {count}장")

if __name__ == "__main__":
    driver = init_driver()
    driver.maximize_window()
    try:
        for celeb in CELEBS:
            print(f"\n--- {celeb} 크롤링 시작 ---")
            crawl_images(celeb, driver)
    finally:
        driver.quit()
    print("\n전체 크롤링 작업이 완료되었습니다.")