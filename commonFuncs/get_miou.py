import os
from PIL import Image
from tqdm import tqdm
from segformer import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    miou_mode = 0
    num_classes = 2
    name_classes = ["background", "hole"]
    VOCdevkit_path = '../VOCdevkit'
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')
    img_extension = ".tif"

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        segformer = SegFormer_Segmentation()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, f"VOC2007/JPEGImages/{image_id}{img_extension}")
            image = Image.open(image_path)
            image = segformer.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".tif"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes,
                                                        img_extension=img_extension)
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
