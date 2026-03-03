import os
import re
import shutil
import subprocess
import random

# =====================================
# 설정 부분 (WSL 절대 경로 적용)
# =====================================
NUM_STRUC = 100
# 윈도우 C드라이브 경로를 WSL 방식(/mnt/c/...)으로 변경합니다.
OUTPUT_DIR = "/mnt/c/Users/bangjinhyun/Desktop/lab/pankajcode/HZO_Code/inputs"

# 메인 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 총 {NUM_STRUC}개의 미세조직을 {OUTPUT_DIR} 에 생성합니다...")

for i in range(1, NUM_STRUC + 1):
    print(f"[{i}/{NUM_STRUC}] 구조 생성 및 복사 중...")
    
    # ---------------------------------------------------------
    # 1. parameters.h의 SEED 값 및 ng_total 랜덤 변경
    # ---------------------------------------------------------
    with open("parameters.h", "r", encoding="utf-8") as f:
        content = f.read()

    new_seed = -random.randint(1000000, 9999999)
    content = re.sub(r"long SEED = -?\d+;", f"long SEED = {new_seed};", content)

    # ng_total 이 grain 개수를 결정하므로, 이를 랜덤으로 변경하면 grain size가 달라짐
    # 값이 클수록 grain이 작아지고, 작을수록 grain이 커짐
    # np = 2000 (max centroid storage) 보다 작아야 함
    new_ng_total = random.randint(400, 1800)
    content = re.sub(r"int ng_total = \d+;", f"int ng_total = {new_ng_total};", content)

    with open("parameters.h", "w", encoding="utf-8") as f:
        f.write(content)
        
    # ---------------------------------------------------------
    # 2. 코드 컴파일 (compile.sh와 동일하게 세팅)
    # ---------------------------------------------------------
    subprocess.run(["gcc", "voronoi.c", "-o", "genpolcr.exe", "-lm"], check=True)
    
    # ---------------------------------------------------------
    # 3. 코드 실행 (run.sh와 동일하게 세팅)
    # ---------------------------------------------------------
    subprocess.run(["./genpolcr.exe"], check=True)
    
    # ---------------------------------------------------------
    # 4. 지정한 경로로 파일 이동 및 이름 변경 (_poly 붙이기)
    # ---------------------------------------------------------
    # 예: /mnt/c/.../inputs/struc1
    struc_dir = os.path.join(OUTPUT_DIR, f"struc{i}")
    os.makedirs(struc_dir, exist_ok=True)
    
    # 생성된 파일 위치 찾기 (C 코드에 따라 polc0000 폴더 안에 생길 수도, 현재 폴더에 생길 수도 있음)
    base_path = "polc0000" if os.path.exists("polc0000/gchar.txt") else "."
    
    # 파일 이름 변경하며 이동 (gchar.txt -> gchar_poly.txt)
    shutil.copy(os.path.join(base_path, "gchar.txt"), os.path.join(struc_dir, "gchar_poly.txt"))
    shutil.copy(os.path.join(base_path, "gdata.bin"), os.path.join(struc_dir, "gdata_poly.bin"))
    
    # 공통 파일인 gnumb.txt는 struc 폴더 바깥(inputs 폴더 바로 아래)에 한 번만 이름 바꿔서 저장
    if i == 1:
        shutil.copy(os.path.join(base_path, "gnumb.txt"), os.path.join(OUTPUT_DIR, "gnumb_poly.txt"))

print(f"\n✅ 완료! '{OUTPUT_DIR}' 폴더에 100개의 구조가 완벽하게 세팅되었습니다.")