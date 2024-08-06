import os
import shutil
from tqdm import tqdm

srcFolder_dir = r'Z:\yaogan\2024zunyiyancaojiance\yandiguihua\xishui\01OriginalImage\20240524XS（矢量已全）'
desFolder_dir = r'Z:\yaogan\2024zunyiyancaojiance\yandiguihua\xishui\01OriginalImage\All_tifs'

def copyFiles(srcFolder_dir, desFolder_dir):
    if not os.path.exists(srcFolder_dir):
        os.makedirs(desFolder_dir)

    files = [f for f in os.listdir(srcFolder_dir) if f.endswith('.tif')]

    for f_name in tqdm(files, desc=f'Copying files from {os.path.basename(srcFolder_dir)}'):
        src_file = os.path.join(srcFolder_dir, f_name)
        des_file = os.path.join(desFolder_dir, f_name)

        try:
            if not os.path.exists(des_file):
                shutil.copy2(src_file, des_file)
        except Exception as e:
            print(f"Failed to copy {src_file} to {des_file}, the error is {e}")

copyFiles(srcFolder_dir, desFolder_dir)
print("全部复制完成！")


