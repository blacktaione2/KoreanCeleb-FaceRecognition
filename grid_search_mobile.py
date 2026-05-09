import gc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from train_mobile import train_model, PHASE1_CKPT  # MobileNet용 train 모듈로 교체
import os

# ── 실험 설정 ────────────────────────────────────────────────────────
# alphas = [0.5, 1.0, 1.5, 2.0]  # ResNet 초기 탐색과 동일한 범위로 시작
# betas  = [0.5, 1.0, 1.5, 2.0]  # ResNet 초기 탐색과 동일한 범위로 시작
alphas = [1.2, 1.3, 1.5, 1.6, 1.7]  # Alpha 1.5 중심 ±0.2~0.3
betas  = [0.8, 0.9, 1.0, 1.2, 1.5]  # Beta 1.0 중심, 0.5 실패 구간 제외
OUTPUT_CSV = 'grid_search_results_mobile.csv'   # ResNet 결과와 구분
OUTPUT_IMG = 'grid_search_heatmap_mobile.png'   # ResNet 결과와 구분

# ── Phase 1 체크포인트 사전 안내 ────────────────────────────────────
if os.path.exists(PHASE1_CKPT):
    print(f"Phase 1 체크포인트 발견 ({PHASE1_CKPT}) → 모든 실험에서 Phase 1 스킵")
else:
    print("Phase 1 체크포인트 없음 → 첫 실험에서만 Phase 1 수행 후 저장")

# ── 그리드 서치 ─────────────────────────────────────────────────────
results = []
total_experiments = len(alphas) * len(betas)
print(f"\n그리드 서치 시작... (총 {total_experiments}회 실험)")

for exp_idx, a in enumerate(alphas):
    for beta_idx, b in enumerate(betas):  # betas.index(b) 제거
        current = exp_idx * len(betas) + beta_idx + 1
        print(f"\n[{current}/{total_experiments}] Alpha: {a}, Beta: {b}")

        result = train_model(
            alpha=a,
            beta=b,
            num_epochs=30,
            patience=8,
            min_smile_acc=0.70
        )

        results.append({
            'Alpha':       a,
            'Beta':        b,
            'Total_Score': result['best_score'],
            'Celeb_Acc':   result['best_celeb_acc'],
            'Smile_Acc':   result['best_smile_acc'],
        })

        print(f"-> 결과: Total={result['best_score']:.4f} | "
              f"Celeb={result['best_celeb_acc']:.4f} | Smile={result['best_smile_acc']:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

# ── DataFrame 생성 및 CSV 저장 ───────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n결과 저장 완료: {OUTPUT_CSV}")

# ── 시각화: 3분할 Heatmap ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

plot_configs = [
    ('Total_Score', 'Total Score (Celeb + Smile Acc)'),
    ('Celeb_Acc',   'Celeb Accuracy'),
    ('Smile_Acc',   'Smile Accuracy'),
]

for ax, (col, title) in zip(axes, plot_configs):
    pivot = df.pivot(index='Alpha', columns='Beta', values=col)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap='YlGnBu',
        vmin=0.0,
        vmax=None,
        ax=ax
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Beta (Smile Loss Weight)')
    ax.set_ylabel('Alpha (Celeb Loss Weight)')

plt.suptitle('Grid Search (MobileNet): Alpha vs Beta', fontsize=15, y=1.02)  # MobileNet 명시
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
plt.close()
print(f"Heatmap 저장 완료: {OUTPUT_IMG}")

# ── 최적 조합 출력 ───────────────────────────────────────────────────
best_row = df.loc[df['Total_Score'].idxmax()]
print("\n[최적 조합]")
print(f"  Alpha: {best_row['Alpha']} | Beta: {best_row['Beta']}")
print(f"  Total Score: {best_row['Total_Score']:.4f}")
print(f"  Celeb Acc  : {best_row['Celeb_Acc']:.4f}")
print(f"  Smile Acc  : {best_row['Smile_Acc']:.4f}")

# ── 상위 5개 조합 출력 ───────────────────────────────────────────────
print("\n[상위 5개 조합]")
top5 = df.nlargest(5, 'Total_Score')[['Alpha', 'Beta', 'Total_Score', 'Celeb_Acc', 'Smile_Acc']]
print(top5.to_string(index=False))