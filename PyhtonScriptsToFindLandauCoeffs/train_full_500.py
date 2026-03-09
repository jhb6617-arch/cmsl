import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

# ==========================================
# 설정
# ==========================================
afe_input_file  = 'fixed_field_data.txt'
fe_input_file   = 'Fe_field_data.txt'
landau_out_file = 'landau_params.txt'    # 란다우계수만 (13필드: 12값 + label)
phase_pool_file = 'phase_pool.txt'       # 상분율 풀 (3필드: AFE%, FE%, DE%)

DE_alpha = 9.5e+09
DE_a0    = 2.372e+09

NUM_LANDAU     = 1000   # 란다우계수 대표점 (K-Means 결과)
NUM_PHASE_POOL = 1000   # 상분율 풀 크기

# ==========================================
# 1. 란다우계수 로드
# ==========================================
print("1. 란다우 계수 데이터를 불러오는 중...")
landau_list = []

with open(afe_input_file, 'r') as f_afe, open(fe_input_file, 'r') as f_fe:
    next(f_afe)
    next(f_fe)
    for line_afe, line_fe in zip(f_afe, f_fe):
        parts_afe = line_afe.strip().split()[:5]
        parts_fe  = line_fe.strip().split()[:5]
        if len(parts_afe) < 5 or len(parts_fe) < 5:
            continue
        row = [float(x) for x in parts_afe] + \
              [float(x) for x in parts_fe] + \
              [DE_alpha, DE_a0]
        landau_list.append(row)

col_names = [
    'AFE_alpha', 'AFE_beta', 'AFE_gamma', 'AFE_g', 'AFE_a0',
    'FE_alpha',  'FE_beta',  'FE_gamma',  'FE_g',  'FE_a0',
    'DE_alpha',  'DE_a0'
]
df_landau = pd.DataFrame(landau_list, columns=col_names)
print(f" -> {len(df_landau)}개의 란다우 계수 세트 로드됨")

# ==========================================
# 2. 란다우계수 공간에서만 K-Means → landau_params.txt
#    (상분율과 Cross-join 없음)
# ==========================================
print(f"2. 란다우 공간에서 K-Means로 {NUM_LANDAU}개 대표점 추출 중...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_landau)

kmeans = MiniBatchKMeans(n_clusters=NUM_LANDAU, random_state=42, batch_size=5000)
kmeans.fit(X_scaled)

closest = pairwise_distances_argmin(kmeans.cluster_centers_, X_scaled)
df_selected = df_landau.iloc[closest].copy()

print(f"3. '{landau_out_file}'에 저장 중 (13필드: 란다우 12개 + label)...")
with open(landau_out_file, 'w') as f:
    for _, row in df_selected.iterrows():
        parts = [f"{row[c]:.6e}" for c in col_names]
        f.write("\t".join(parts) + "\tAFE\n")
print(f" -> {landau_out_file} 저장 완료 ({NUM_LANDAU}행)")

# ==========================================
# 3. 상분율 풀 독립 생성 → phase_pool.txt
#    (LHS, DE <= 50% 제한)
# ==========================================
print(f"4. 상분율 풀 {NUM_PHASE_POOL}개 생성 중 (LHS)...")
sampler = qmc.LatinHypercube(d=2, seed=0)
samples = sampler.random(n=NUM_PHASE_POOL)

frac_list = []
for s in samples:
    de      = s[0] * 50.0           # DE <= 50%
    remains = 100.0 - de
    afe     = s[1] * remains
    fe      = remains - afe
    frac_list.append([afe, fe, de])

with open(phase_pool_file, 'w') as f:
    for afe, fe, de in frac_list:
        f.write(f"{afe:.6f}\t{fe:.6f}\t{de:.6f}\n")
print(f" -> {phase_pool_file} 저장 완료 ({NUM_PHASE_POOL}행)")

print("\n완료!")
print(f"  {landau_out_file}: {NUM_LANDAU}행 (란다우계수만, CUDA 입력용)")
print(f"  {phase_pool_file}: {NUM_PHASE_POOL}행 (상분율 풀, inputs/ 폴더에 복사 필요)")
