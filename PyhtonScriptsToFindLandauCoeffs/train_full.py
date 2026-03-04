import random

# ==========================================
# 1. 파일 이름 설정
# ==========================================
afe_input_file = 'fixed_field_data.txt'  # AFE 파라미터 파일 (afe_range_1.py 결과)
fe_input_file  = 'Fe_field_data.txt'     # FE 파라미터 파일 (fe_range_1.py 결과)
output_file    = 'train.txt'             # 최종 결과 파일

# ==========================================
# DE 파라미터 고정값 설정 (사용자 요청)
# ==========================================
DE_alpha = 9.5e+09
DE_a0    = 2.372e+09

def get_random_fractions():
    """3개 상의 합이 100이 되도록 랜덤 생성"""
    r1 = random.random()
    r2 = random.random()
    r3 = random.random()
    total = r1 + r2 + r3
    
    # 퍼센트(0~100) 단위로 변환
    afe = (r1 / total) * 100.0
    fe  = (r2 / total) * 100.0
    de  = (r3 / total) * 100.0
    return afe, fe, de

# ==========================================
# 2. 데이터 병합 및 생성
# ==========================================
print("파일 병합을 시작합니다...")

try:
    with open(afe_input_file, 'r') as f_afe, \
         open(fe_input_file, 'r') as f_fe, \
         open(output_file, 'w') as f_out:

        # 헤더 건너뛰기
        next(f_afe)
        next(f_fe)

        # 두 파일을 한 줄씩 읽어서 합치기
        for line_afe, line_fe in zip(f_afe, f_fe):
            
            # 1. AFE 파라미터 (5개) 추출
            parts_afe = line_afe.strip().split()[:5]
            if len(parts_afe) < 5: continue 

            # 2. FE 파라미터 (5개) 추출
            parts_fe = line_fe.strip().split()[:5]
            if len(parts_fe) < 5: continue

            # 3. DE 파라미터 (2개) - 리스트로 생성
            list_de = [f"{DE_alpha:.6e}", f"{DE_a0:.6e}"]

            # 4. Phase Fraction (3개) - 랜덤 생성
            afe_p, fe_p, de_p = get_random_fractions()
            list_frac = [f"{afe_p:.6f}", f"{fe_p:.6f}", f"{de_p:.6f}"]

            # 5. Reference String (1개)
            list_ref = ["AFE"]

            # 6. 모든 컬럼 합치기 (총 16개)
            # [AFE 5개] + [FE 5개] + [DE 2개] + [Frac 3개] + [Ref 1개]
            full_list = parts_afe + parts_fe + list_de + list_frac + list_ref
            
            # 7. 탭(\t)으로 연결하여 쓰기 (afe_fe.txt와 동일 양식)
            f_out.write("\t".join(full_list) + "\n")

    print(f"성공! '{output_file}' 파일이 생성되었습니다.")
    print("생성된 파일은 afe_fe.txt와 동일하게 탭(Tab)으로 구분된 16개 컬럼을 가집니다.")

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. ({e})")