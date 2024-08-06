import os
import shutil
import glob
import time
import traceback

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
# from torchvision import transforms
from commonFuncs.gdal_cut_perImage import split_tif
from commonFuncs.merge_pred_images import mosaic
from commonFuncs.segformer import SegFormer_Segmentation

from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['PROJ_LIB'] = r"C:\\Users\\obt\\.conda\\envs\\SegBB\\Lib\\site-packages\\osgeo\\data\\proj"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def check_dir_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created:", path)
    else:
        print("Directory already exists:", path)


def writeTiff(im_data, im_width, im_height, im_bands, path, path1):
    from osgeo import gdal
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float64

    if len(im_data.shape) == 3 and im_data.shape[2] == im_bands:
        im_data = im_data.transpose(2, 0, 1)
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands = 1
    else:
        raise ValueError("Unsupported data shape")

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)

    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    data = gdal.Open(path1)
    im_geotrans = data.GetGeoTransform()
    im_proj = data.GetProjection()
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    del dataset


def delete_files_except_result(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and "result" not in filename:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file: {file_path} - {e}")

def process_image(segformer, img_name, dir_origin_path, dir_save_path):
    image_path = os.path.join(dir_origin_path, img_name)
    try:
        cv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if cv_image is None:
            raise ValueError(f"Open Error for {img_name} with OpenCV! Skipping...")

        max_val = np.max(cv_image)
        if max_val > 0:
            cv_image = (255 * (cv_image / max_val)).astype(np.uint8)
        else:
            cv_image = cv_image.astype(np.uint8)

        if cv_image.ndim == 2:  # 灰度图像
            image = Image.fromarray(cv_image)
        else:  # 彩色图像
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Open Error for {img_name}! Skipping...")
        print(traceback.format_exc())
        return

    r_image = segformer.detect_image(image)
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)
    ori_tif_path = os.path.join(dir_origin_path, img_name)
    result = np.array(r_image)
    writeTiff(result, result.shape[1], result.shape[0], 3, os.path.join(dir_save_path, img_name), ori_tif_path)
    image.close()

def predict_batches(dir_origin_path, dir_save_path):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segformer = SegFormer_Segmentation(num_classes=2)
    #
    # composed_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    img_names = os.listdir(dir_origin_path)
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(process_image, segformer, img_name, dir_origin_path, dir_save_path) for img_name in img_names if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting Images", unit="image"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")

# def handle_remove_error(func, path, exc_info):
#     print(f"Error removing {path}. Retrying...")
#     time.sleep(5)  # 延迟一段时间后重试
#     try:
#         os.chmod(path, 0o777)  # 更改权限
#         func(path)  # 再次尝试删除
#     except Exception as e:
#         print(f"Failed to remove {path} again: {e}")


def process_tif_file(input_file, clip_dir, pre_dir, output_file, tile_size=256):
    print(f"Splitting {input_file}...")
    split_tif(input_file, clip_dir, tile_size)

    print(f"Predicting on tiles in {clip_dir}...")
    predict_batches(clip_dir, pre_dir)

    print(f"Merging predictions into {output_file}...")
    mosaic(pre_dir, output_file)

    # print(f"Cleaning up temporary files...")
    # time.sleep(5)  # 添加延迟以确保文件操作完成
    # shutil.rmtree(clip_dir, onerror=handle_remove_error)
    # shutil.rmtree(pre_dir, onerror=handle_remove_error)


def process_all_tif_files(input_dir, clip_root_dir, pre_root_dir, output_dir, tile_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    for index, file in enumerate(tif_files, start=1):
        print(f"==============================={index}/{len(tif_files)}===============================")
        print(file)

        base_name = os.path.basename(file)
        clip_dir = os.path.join(clip_root_dir, base_name.split(".")[0], "image_256")
        pre_dir = os.path.join(pre_root_dir, base_name.split(".")[0], "result_256")
        output_file = os.path.join(output_dir, base_name)

        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping.")
            continue

        os.makedirs(clip_dir, exist_ok=True)
        os.makedirs(pre_dir, exist_ok=True)

        process_tif_file(file, clip_dir, pre_dir, output_file, tile_size)


if __name__ == "__main__":
    input_dir = r"E:\dataset\test\temp"
    clip_root_dir = r"F:\ranXY\temp\clip1"
    pre_root_dir = r"F:\ranXY\temp\result1"
    output_dir = r"F:\ranXY\segformer_predAll_res"
    tile_size = 1000

    process_all_tif_files(input_dir, clip_root_dir, pre_root_dir, output_dir, tile_size)
