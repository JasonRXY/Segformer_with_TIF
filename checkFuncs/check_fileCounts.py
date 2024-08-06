import os

folder_path1 = r'E:\dataset\ds_RG4Band\JPEGImages'
folder_path2 = r'E:\dataset\ds_RG4Band\SegmentationClass'

def get_tif_filenames(directory):
    """获取目录中所有 .tif 文件的文件名"""
    return sorted([f for f in os.listdir(directory) if f.endswith('.tif')])


def compare_tif_filenames(directory1, directory2):
    """比较两个目录中的 .tif 文件名是否一一对应"""
    files1 = get_tif_filenames(directory1)
    files2 = get_tif_filenames(directory2)

    if len(files1) != len(files2):
        print("文件数量不匹配。")
        print(f"{directory1} 中的文件数量: {len(files1)}")
        print(f"{directory2} 中的文件数量: {len(files2)}")
    else:
        print("文件数量匹配。")

    all_matched = True
    for file1, file2 in zip(files1, files2):
        print(f'file1 = {file1} and file2 = {file2}')
        if file1 != file2:
            print(f"文件名不匹配: {file1} != {file2}")
            all_matched = False

    if all_matched:
        print("所有文件名都匹配。")
    else:
        print("部分文件名不匹配。")


if __name__ == "__main__":
    folder_path1 = r'E:\dataset\ds_RG4Band\JPEGImages'
    folder_path2 = r'E:\dataset\ds_RG4Band\SegmentationClass'

    compare_tif_filenames(folder_path1, folder_path2)
