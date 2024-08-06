# from osgeo import gdal
# import numpy as np
# import cv2
# import os
#
# from tqdm import tqdm
#
#
# def read_tif_image(file_path):
#     dataset = gdal.Open(file_path)
#     print(dataset.ReadAsArray())
#     print(dataset.ReadAsArray().shape)
#     if not dataset:
#         raise ValueError(f"Unable to open {file_path}")
#
#     geotransform = dataset.GetGeoTransform()
#     projection = dataset.GetProjection()
#
#     # Assuming the image is a 3-band (RGB) image
#     r_band = dataset.GetRasterBand(1).ReadAsArray()
#     g_band = dataset.GetRasterBand(2).ReadAsArray()
#     b_band = dataset.GetRasterBand(3).ReadAsArray()
#
#     image = np.dstack((r_band, g_band, b_band))
#
#     print(f'image.shape = {image.shape}')
#
#     return image, geotransform, projection
#
#
# def write_tif_image(file_path, image, geotransform, projection):
#     driver = gdal.GetDriverByName("GTiff")
#     rows, cols = image.shape
#     dataset = driver.Create(file_path, cols, rows, 1, gdal.GDT_Byte)
#
#     if not dataset:
#         raise ValueError(f"Unable to create {file_path}")
#
#     dataset.SetGeoTransform(geotransform)
#     dataset.SetProjection(projection)
#     dataset.GetRasterBand(1).WriteArray(image)
#     dataset.FlushCache()
#
#
# def binarize_image(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
#     binary_image[binary_image == 255] = 1  # Convert white to 1
#     return binary_image
#
#
# def process_folder(folder_path):
#     for filename in tqdm(os.listdir(folder_path), desc='Processing Images'):
#         if filename.endswith(".tif"):
#             input_file_path = os.path.join(folder_path, filename)
#             print(f'input_file_path = {input_file_path}')
#             output_file_path = os.path.join(folder_path, filename.replace(".tif", "_binary.tif"))
#
#             try:
#                 image, geotransform, projection = read_tif_image(input_file_path)
#                 binary_image = binarize_image(image)
#                 write_tif_image(output_file_path, binary_image, geotransform, projection)
#                 # print(f"Binarization complete and saved to {output_file_path}")
#
#                 os.remove(input_file_path)
#                 # print(f"Original file {input_file_path} deleted.")
#             except Exception as e:
#                 print(f"Error processing {input_file_path}: {e}")
#
#
# def main():
#     folder_path = r"E:\dataset\test\result_2048"
#     process_folder(folder_path)
#
#
# if __name__ == "__main__":
#     main()


from osgeo import gdal
import numpy as np
import cv2
import os
from tqdm import tqdm


def read_tif_image(file_path):
    dataset = gdal.Open(file_path)
    if not dataset:
        raise ValueError(f"Unable to open {file_path}")

    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    bands = dataset.RasterCount

    # Read each band and stack them
    band_data = []
    for band in range(1, bands + 1):
        band_data.append(dataset.GetRasterBand(band).ReadAsArray())

    image = np.dstack(band_data)

    return image, geotransform, projection


def write_tif_image(file_path, image, geotransform, projection):
    driver = gdal.GetDriverByName("GTiff")
    rows, cols, bands = image.shape
    dataset = driver.Create(file_path, cols, rows, bands, gdal.GDT_Byte)

    if not dataset:
        raise ValueError(f"Unable to create {file_path}")

    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)

    for i in range(bands):
        dataset.GetRasterBand(i + 1).WriteArray(image[:, :, i])

    dataset.FlushCache()


def binarize_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    binary_image[binary_image == 255] = 1  # Convert white to 1
    return binary_image


def process_folder(folder_path):
    for filename in tqdm(os.listdir(folder_path), desc='Processing Images'):
        if filename.endswith(".tif"):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(folder_path, filename.replace(".tif", "_binary.tif"))

            try:
                image, geotransform, projection = read_tif_image(input_file_path)
                binary_image = binarize_image(image)
                write_tif_image(output_file_path, binary_image[:, :, np.newaxis], geotransform, projection)
                print(f"Binarization complete and saved to {output_file_path}")

                os.remove(input_file_path)
                print(f"Original file {input_file_path} deleted.")
            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")


def main():
    folder_path = r"E:\dataset\test\result_2048"
    process_folder(folder_path)


if __name__ == "__main__":
    main()

