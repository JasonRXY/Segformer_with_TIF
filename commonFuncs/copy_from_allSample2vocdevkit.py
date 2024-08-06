import os
import shutil
from tqdm import tqdm


def copy_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取文件列表
    files = [f for f in os.listdir(source_dir) if f.endswith('.tif')]

    # 使用tqdm显示进度条
    for file_name in tqdm(files, desc=f'Copying files from {os.path.basename(source_dir)}'):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        try:
            if not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)
        except Exception as e:
            print(f"Failed to copy {file_name} from {source_file} to {target_file}. Error: {e}")


# 定义文件夹路径
jpeg_images_source = r"F:\ranXY\all_samples_from_Zdrive\all_samples\JPEGImages"
jpeg_images_target = r"C:\Users\obt\Desktop\segformer-pytorch-master\VOCdevkit\VOC2007\JPEGImages"

segmentation_class_source = r"F:\ranXY\all_samples_from_Zdrive\all_samples\SegmentationClass"
segmentation_class_target = r"C:\Users\obt\Desktop\segformer-pytorch-master\VOCdevkit\VOC2007\SegmentationClass"

# 复制JPEGImages文件夹下的文件
copy_files(jpeg_images_source, jpeg_images_target)

# 复制SegmentationClass文件夹下的文件
copy_files(segmentation_class_source, segmentation_class_target)
