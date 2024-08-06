import os
import rasterio

def print_tif_dimensions(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            filepath = os.path.join(directory, filename)
            with rasterio.open(filepath) as src:
                width = src.width
                height = src.height
                count = src.count
                print(f"{filename}: width={width}, height={height}, band_count={count}")

if __name__ == "__main__":
    directory = r'E:\dataset\ds_RG4Band\JPEGImages\103_tile_5_7.tif'  # 替换为你的 .tif 文件所在目录
    print_tif_dimensions(directory)
