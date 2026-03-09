import os
import shutil
import subprocess

# =====================================
# 설정 부분 (WSL 절대 경로 적용)
# =====================================
NUM_STRUC = 1000
# 윈도우 C드라이브 경로를 WSL 방식(/mnt/c/...)으로 변경합니다.
OUTPUT_DIR = "/mnt/c/Users/bangjinhyun/Desktop/lab/pankajcode/HZO_Code/inputs"

# 메인 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 총 {NUM_STRUC}개의 미세조직을 {OUTPUT_DIR} 에 생성합니다...")

for i in range(1, NUM_STRUC + 1):
    print(f"[{i}/{NUM_STRUC}] 구조 생성 및 복사 중...")

    # ---------------------------------------------------------
    # 1. 코드 실행 (voronoi.c가 내부적으로 ng_total을 랜덤화함)
    # ---------------------------------------------------------
    subprocess.run(["./genpolcr.exe"], check=True)

    # ---------------------------------------------------------
    # 2. 지정한 경로로 파일 이동 및 이름 변경 (_poly 붙이기)
    # ---------------------------------------------------------
    # 예: /mnt/c/.../inputs/struc1
    struc_dir = os.path.join(OUTPUT_DIR, f"struc{i}")
    os.makedirs(struc_dir, exist_ok=True)

    # 생성된 파일 위치 찾기 (C 코드에 따라 polc0000 폴더 안에 생길 수도, 현재 폴더에 생길 수도 있음)
    base_path = "polc0000" if os.path.exists("polc0000/gchar.txt") else "."

    # 파일 이름 변경하며 이동 (gchar.txt -> gchar_poly.txt)
    shutil.copy(os.path.join(base_path, "gchar.txt"), os.path.join(struc_dir, "gchar_poly.txt"))
    shutil.copy(os.path.join(base_path, "gdata.bin"), os.path.join(struc_dir, "gdata_poly.bin"))

    # gnumb.txt를 각 struc 폴더에 저장 (구조마다 ng_total이 다르므로 각각 저장해야 함)
    shutil.copy(os.path.join(base_path, "gnumb.txt"), os.path.join(struc_dir, "gnumb_poly.txt"))

print(f"\n✅ 완료! '{OUTPUT_DIR}' 폴더에 {NUM_STRUC}개의 구조가 완벽하게 세팅되었습니다.")
