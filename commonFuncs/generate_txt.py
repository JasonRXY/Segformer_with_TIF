import glob
import os.path
import random

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def create_txt(txt_path,contents):
    with open(txt_path, 'w') as f:
        for x in contents[:-1]:
            f.write(x + '\n')
        f.write(contents[-1])
    f.close()


# 所有图片存放文件夹
image_dir = r"F:\ranXY\segformer-pytorch-master\datasets\JPEGImages"

# 所有图片路径列表
file_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)if file.endswith('.jpg')]

print(file_list)

# 所有文件名字列表
name_list = []

for path in file_list:
    name = path.split("\\")[-1].split(".")[0]
    name_list.append(name)

train = []
train_name_list = []
val_name_list = []
test_name_list = []

# for name in name_list:
#     if "train" in name or "image1" in name:
#         train.append(name)
#     if "test" in name:
#         test_name_list.append(name)

val_name_list,train_name_list = data_split(name_list, ratio=0.2, shuffle=True)


print("训练样本数：{}".format(len(train_name_list)))
print("验证样本数：{}".format(len(val_name_list)))
print("测试样本数：{}".format(len(test_name_list)))


# txt文件路径
train_txt = r"F:\ranXY\segformer-pytorch-master\datasets\train_label_txt\train.txt"
# trainval_txt = r"F:\ranXY\fine_tuning_ds\VOC2007\ImageSets\Segmentation\trainval.txt"
val_txt = r"F:\ranXY\segformer-pytorch-master\datasets\val_label_txt\val.txt"
test_txt = r"F:\ranXY\fine_tuning_ds\VOC2007\ImageSets\test.txt"

# 写入txt内容
create_txt(train_txt,train_name_list)
# create_txt(trainval_txt,train_name_list + val_name_list)
create_txt(val_txt,val_name_list)
# create_txt(test_txt,val_name_list)
