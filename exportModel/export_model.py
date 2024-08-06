import glob
import os
import traceback

import cv2
import torch
import onnx
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from nets.segformer import SegFormer
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
from predict_all import *
from commonFuncs.convertRGB2Binary import *

def convert_to_onnx(model, model_path, onnx_path, input_shape=(3, 224, 224), batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建一个假的输入张量
    im = torch.zeros(batch_size, *input_shape).to(device)  # image size(batch_size, 3, 224, 224)

    input_layer_names = ["input"]
    output_layer_names = ["output"]

    # 导出模型
    print(f'Starting export with onnx {onnx.__version__}.')
    torch.onnx.export(model,
                      im,
                      f=onnx_path,
                      verbose=False,
                      opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=input_layer_names,
                      output_names=output_layer_names,
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # 检查导出的模型
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(f'Model has been successfully converted to ONNX and saved at {onnx_path}.')

def check_onnx(onnx_file_path):
    # 我们可以使用异常处理的方法进行检验
    try:
        # 当我们的模型不可用时，将会报出异常
        onnx.checker.check_model(onnx_file_path)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        # 模型可用时，将不会报出异常，并会输出“The model is valid!”
        print("The model is valid!")

def infer_onnx(onnx_path, input_image):
    # 创建 ONNX Runtime 推理会话
    ort_session = ort.InferenceSession(onnx_path)

    # 准备输入数据
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    input_image = input_image.astype(np.float32)

    # 运行推理
    ort_inputs = {input_name: input_image}
    ort_outs = ort_session.run([output_name], ort_inputs)

    return ort_outs


if __name__ == "__main__":
    model_path = r'F:\ranXY\segformer-pytorch-master\logs\loss_2024_07_25_RG4Band_b5_1000_1000\ep180-loss0.193-val_loss0.206.pth'
    onnx_path = r"F:\ranXY\segformer-pytorch-master\segformer.onnx"
    input_shape = (3, 224, 224)  # 输入图像的形状 (C, H, W)
    batch_size = 1  # 批量大小
    is_convert_to_onnx = False
    if is_convert_to_onnx:
        # 创建模型并加载权重
        torch_model = SegFormer(num_classes=2, phi='b5', pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch_model.load_state_dict(torch.load(model_path, map_location=device))
        torch_model.eval()
        print('{} model and classes loaded.'.format(model_path))

        # 转换为 ONNX
        convert_to_onnx(torch_model, model_path, onnx_path, input_shape, batch_size)
        print("Export Finished!")

    # 检查onnx文件是否正确
    check_onnx(onnx_path)

    # 准备一个输入图像进行推理
    input_image = np.random.randn(1, *input_shape).astype(np.float32)

    # 使用 ONNX Runtime 进行推理
    output = infer_onnx(onnx_path, input_image)
    print("Inference output:", output)

    # 进行实际预测
    input_dir = r"E:\dataset\onnx_test\temp"  # 输入图像目录
    clip_root_dir = r"E:\dataset\onnx_test\clip"  # 临时剪切图像存储目录
    pre_root_dir = r"E:\dataset\onnx_test\result"  # 临时预测结果存储目录
    output_dir = r"E:\dataset\onnx_test\pred_res"  # 输出结果目录
    tile_size = 1000  # 裁剪大小

    # 对.tif图片继续预测
    process_all_tif_files(input_dir, clip_root_dir, pre_root_dir, output_dir, tile_size)

    # 调用convert.py中的函数进行二值化处理
    process_folder(output_dir)
