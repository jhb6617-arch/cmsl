import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

# ==========================================
# 1. 환경 및 파일 설정
# ==========================================
afe_input_file = 'fixed_field_data.txt'  # AFE 파라미터 파일
fe_input_file  = 'Fe_field_data.txt'     # FE 파라미터 파일
output_file    = 'train_500_optimized.txt' # 최종 500개 결과 파일

DE_alpha = 9.5e+09
DE_a0    = 2.372e+09

NUM_FRACTIONS = 100       # 생성할 상분율 개수
NUM_CENTROIDS = 500       # 최종 추출할 조합 개수

print("1. 란다우 계수 데이터를 불러오는 중...")
landau_list = []

# 기존 코드처럼 파일에서 AFE와 FE 파라미터를 읽어옵니다.
with open(afe_input_file, 'r') as f_afe, open(fe_input_file, 'r') as f_fe:
    next(f_afe) # 헤더 스킵
    next(f_fe)
    for line_afe, line_fe in zip(f_afe, f_fe):
        parts_afe = line_afe.strip().split()[:5]
        parts_fe = line_fe.strip().split()[:5]
        
        if len(parts_afe) < 5 or len(parts_fe) < 5: 
            continue
            
        # [AFE 5개, FE 5개, DE 2개]
        row = [float(x) for x in parts_afe] + [float(x) for x in parts_fe] + [DE_alpha, DE_a0]
        landau_list.append(row)

# 데이터프레임으로 변환
col_names = [
    'AFE_alpha', 'AFE_beta', 'AFE_gamma', 'AFE_g', 'AFE_a0',
    'FE_alpha', 'FE_beta', 'FE_gamma', 'FE_g', 'FE_a0',
    'DE_alpha', 'DE_a0'
]
df_landau = pd.DataFrame(landau_list, columns=col_names)
print(f" -> {len(df_landau)}개의 란다우 계수 세트가 로드되었습니다.")

# ==========================================
# 2. Phase Fraction 100개 균일 생성 (LHS 활용)
# ==========================================
print("2. 상분율 100개 세트 생성 중 (LHS, DE <= 50% 제한)...")
# 2차원 공간을 고르게 샘플링
sampler = qmc.LatinHypercube(d=2, seed=42)
samples = sampler.random(n=NUM_FRACTIONS)

frac_list = []
for s in samples:
    # 제한조건: DE는 최대 50%
    de = s[0] * 50.0  
    
    # 남은 비율(100 - de)을 AFE와 FE가 나눠 가짐
    remains = 100.0 - de
    afe = s[1] * remains
    fe = remains - afe
    
    frac_list.append([afe, fe, de])

df_frac = pd.DataFrame(frac_list, columns=['AFE_p', 'FE_p', 'DE_p'])

# ==========================================
# 3. 100만 개 (10,000 x 100) 조합 생성 (Cross Join)
# ==========================================
print("3. 100만 개 파라미터 조합 생성 중 (Cross Join)...")
df_landau['key'] = 1
df_frac['key'] = 1
df_full = pd.merge(df_landau, df_frac, on='key').drop('key', axis=1)

print(f" -> 총 {len(df_full):,} 개의 데이터 조합이 생성되었습니다.")

# ==========================================
# 4. 데이터 스케일링 (정규화)
# ==========================================
print("4. 데이터 정규화(Scaling) 진행 중...")
# 란다우 계수(10^9 단위)와 상분율(0~100 단위)의 스케일 차이를 맞춰줌
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_full)

# ==========================================
# 5. K-Means 클러스터링을 통한 500개 대표점 추출
# ==========================================
print(f"5. MiniBatchKMeans로 {NUM_CENTROIDS}개의 대표 중심점 탐색 중 (약 1~2분 소요)...")
# 데이터가 100만 개이므로 속도가 빠른 MiniBatchKMeans 사용
kmeans = MiniBatchKMeans(n_clusters=NUM_CENTROIDS, random_state=42, batch_size=10000)
kmeans.fit(X_scaled)

print("6. 계산된 중심점과 가장 가까운 실제 조합 추출 중...")
# 500개의 가상 중심점과 가장 가까운 실제 데이터 500개의 인덱스를 찾음
centroids = kmeans.cluster_centers_
closest_indices = pairwise_distances_argmin(centroids, X_scaled)

# 500개의 데이터프레임 추출
df_selected = df_full.iloc[closest_indices].copy()

# ==========================================
# 7. 최종 결과를 기존 형식에 맞춰 저장
# ==========================================
print(f"7. 최종 500개 데이터를 '{output_file}'에 저장합니다...")
with open(output_file, 'w') as f_out:
    for _, row in df_selected.iterrows():
        # AFE 파라미터 (5) + FE 파라미터 (5) + DE (2) + Fraction (3)
        parts = [
            f"{row['AFE_alpha']:.6e}", f"{row['AFE_beta']:.6e}", f"{row['AFE_gamma']:.6e}", f"{row['AFE_g']:.6e}", f"{row['AFE_a0']:.6e}",
            f"{row['FE_alpha']:.6e}", f"{row['FE_beta']:.6e}", f"{row['FE_gamma']:.6e}", f"{row['FE_g']:.6e}", f"{row['FE_a0']:.6e}",
            f"{row['DE_alpha']:.6e}", f"{row['DE_a0']:.6e}",
            f"{row['AFE_p']:.6f}", f"{row['FE_p']:.6f}", f"{row['DE_p']:.6f}"
        ]
        # 기존 train_full.py 처럼 맨 마지막에 임의의 라벨('AFE')을 덧붙임
        f_out.write("\t".join(parts) + "\tAFE\n")

print("🎉 모든 작업이 성공적으로 완료되었습니다!")