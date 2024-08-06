import os
import numpy as np
from osgeo import gdal
from osgeo import gdalconst
os.environ['PROJ_LIB'] = r"C:\Users\obt\.conda\envs\SegBB\Lib\site-packages\osgeo\data\proj"

def split_tif(input_file, output_folder, tile_size):
    # 打开输入文件
    dataset = gdal.Open(input_file, gdalconst.GA_ReadOnly)

    # 获取输入文件的地理信息和投影信息
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # 获取输入文件的大小和波段数
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    # 计算裁剪后小图的行列数
    x_tiles = (width + tile_size - 1) // tile_size # 确保图像的每个部分都能够被完整地覆盖，并且不会遗漏边缘部分
    y_tiles = (height + tile_size - 1) // tile_size

    # 循环遍历所有小图
    for y in range(y_tiles):
        print("============={}/{}=============".format(y+1, y_tiles))
        for x in range(x_tiles):
            # 计算当前小图的起始和结束列号和行号
            xstart = x * tile_size
            xend = min(xstart + tile_size, width)
            ystart = y * tile_size
            yend = min(ystart + tile_size, height)

            # 读取当前小图数据
            tile_data = dataset.ReadAsArray(xstart, ystart, xend-xstart, yend-ystart)

            # 创建输出文件名
            output_file = os.path.join(output_folder, f"{x}_{y}.tif")

            # 创建输出文件
            driver = gdal.GetDriverByName("GTiff")
            out_dataset = driver.Create(output_file, tile_size, tile_size, bands, gdalconst.GDT_Float32)

            # 设置输出文件的地理信息和投影信息
            out_dataset.SetGeoTransform((geotransform[0] + xstart * geotransform[1],  # 新的左上角像素的 X 坐标
                                         geotransform[1],                             # 水平分辨率（像素宽度）
                                         0,                                           # 旋转角度（保持为 0）
                                         geotransform[3] + ystart * geotransform[5],  # 新的左上角像素的 Y 坐标
                                         0,                                           # 旋转角度（保持为 0）
                                         geotransform[5]))                            # 垂直分辨率（像素高度
            out_dataset.SetProjection(projection)

            # 创建空白数组，用0填充边缘不满足256×256大小的像元
            fill_array = np.zeros((bands, tile_size, tile_size), dtype=np.float32)
            # tile_data的形状为（bands, height, weight）
            fill_array[:, :tile_data.shape[1], :tile_data.shape[2]] = tile_data

            # 将小图数据写入输出文件
            for band in range(bands):
                out_dataset.GetRasterBand(band + 1).WriteArray(fill_array[band])

            # 关闭输出文件
            out_dataset = None
    # 关闭输入文件
    dataset = None


# 创建文件夹
def check_dir_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created:", path)
    else:
        print("Directory already exists:", path)

if __name__ == '__main__':
    input_file = r"F:\ranXY\test_other_images\huangyangzhen_008_xiashikan.tif"
    output_folder = r"F:\ranXY\segformer_clip2pred"
    check_dir_exist(output_folder)
    tile_size = 256
    split_tif(input_file, output_folder, tile_size)
    print("saved in {}".format(output_folder))
