import os
import shutil
from osgeo import gdal
from tqdm import tqdm


def delete_files_except_result(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and "result" not in filename:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file: {file_path} - {e}")


def VRT(input_dir, vrt_path, out_path):
    os.chdir(input_dir)  # 定位到指定文件夹
    print(os.listdir())  # 打印文件夹内所含文件查看是否正确
    print("总文件数：{}".format(len(os.listdir())))
    tifs = os.listdir()

    print('拼接..')
    print('输入文件列表:', tifs)  # 打印输入文件列表以检查是否正确读取

    vrt = gdal.BuildVRT(vrt_path, tifs)

    if vrt is None:
        print('Error: Failed to create VRT')
        return
    else:
        print('VRT successfully created')

    print('done..')
    out_options = gdal.TranslateOptions(outputType=gdal.GDT_Byte, creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'])
    print('输出tif...')
    print(out_path)

    try:
        gdal.Translate(out_path, vrt, options=out_options)
        print('TIF successfully created')
    except Exception as e:
        print(f"Error occurred while creating TIF: {e}")

    vrt = None


def delete_folder(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Folder '{path}' has been deleted successfully.")
        else:
            print(f"Folder '{path}' does not exist.")
    except Exception as e:
        print(f"Error occurred while deleting folder '{path}': {e}")


def mosaic(dir, save_path):
    input_dir = dir
    vrt_path = r"C:\Users\obt\Desktop\test\all.vrt"
    out_path = save_path

    print(vrt_path)

    VRT(input_dir, vrt_path, out_path)

    print("saved in : {}".format(out_path))
    print("Del files......")
    # delete_files_except_result(input_dir)


def main():
    save_path = r'F:\ranXY\result\meitan_xihe_qinggangwan_processed.tif'
    pred_dir = r'F:\ranXY\temp\result1\meitan_xihe_qinggangwan\processed_result_256'
    mosaic(pred_dir, save_path)


if __name__ == "__main__":
    main()
