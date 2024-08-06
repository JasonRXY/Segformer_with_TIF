import time
import os
import traceback
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed
from segformer import SegFormer_Segmentation
import torch

import time
import os
import traceback
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed
from segformer import SegFormer_Segmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def writeTiff(im_data, im_width, im_height, im_bands, path, path1):
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

def process_image(args):
    img_name, dir_origin_path, dir_save_path, processed_dir, segformer, count, name_classes = args
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
        if cv_image.ndim == 2:
            image = Image.fromarray(cv_image)
        else:
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Open Error for {img_name}! Skipping...")
        print(traceback.format_exc())
        return

    r_image = segformer.detect_image(image, count=count, name_classes=name_classes)
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)
    ori_tif_path = os.path.join(dir_origin_path, img_name)
    result = np.array(r_image)
    writeTiff(result, result.shape[1], result.shape[0], 3, os.path.join(dir_save_path, img_name), ori_tif_path)
    # processed_path = os.path.join(processed_dir, img_name)
    # postprocess_prediction_single(os.path.join(dir_save_path, img_name), processed_path)

def batch_process_images(img_names, dir_origin_path, dir_save_path, processed_dir, segformer, count, name_classes, batch_size=128):
    num_batches = (len(img_names) + batch_size - 1) // batch_size

    """
    ThreadPoolExecutor的使用
    """
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i in range(num_batches):
            batch = img_names[i * batch_size:(i + 1) * batch_size]
            for img_name in batch:
                futures.append(executor.submit(process_image, (img_name, dir_origin_path, dir_save_path, processed_dir, segformer, count, name_classes)))

        with tqdm(total=len(img_names), desc="Processing Images", unit="image") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing image: {e}")
                pbar.update(1)

if __name__ == "__main__":
    segformer = SegFormer_Segmentation(num_classes=2)

    mode = "batch_predict"
    count = False
    name_classes = ["background", "hole"]

    dir_origin_path = r"F:\ranXY\segformer_clip2pred"
    dir_save_path = r"F:\ranXY\segformer_clip_pred_res"
    processed_dir = os.path.join(dir_save_path, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if mode == "batch_predict":
        img_names = [img for img in os.listdir(dir_origin_path) if img.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]
        batch_process_images(img_names, dir_origin_path, dir_save_path, processed_dir, segformer, count, name_classes, batch_size=128)
    # elif mode == "export_onnx":
    #     segformer.convert_to_onnx(simplify, onnx_save_path)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict', or 'batch_predict'.")
