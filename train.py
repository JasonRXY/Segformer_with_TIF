import datetime
import os
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.segformer import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

if __name__ == "__main__":

    Cuda            = True

    seed            = 888

    distributed     = False

    sync_bn         = True

    fp16            = True

    num_classes     = 2

    phi             = "b5"

    pretrained      = False

    model_path      = ""

    input_shape     = [1000, 1000]
    

    Init_Epoch          = 0
    Freeze_Epoch        = 500
    Freeze_batch_size   = 8

    UnFreeze_Epoch      = 500
    Unfreeze_batch_size = 4

    Freeze_Train        = False

    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0

    lr_decay_type       = 'step'

    save_period         = 10

    save_dir            = 'logs'

    eval_flag           = True
    eval_period         = 239

    VOCdevkit_path  = r'E:\dataset\ds_RG4Band'

    dice_loss       = True

    focal_loss      = True

    cls_weights     = np.ones([num_classes], np.float32)

    num_workers     = 4

    seed_everything(seed)
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="gloo")
        local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        print(f'local rank = {local_rank}')
        rank        = int(os.environ.get("RANK", 0))
        print(f'rank = {rank}')
        device      = torch.device("cuda", local_rank)
        dist.barrier()  # 确保所有进程同步
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
            dist.barrier()  # 确保所有进程同步
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   下载预训练权重
    #----------------------------------------------------#
    print('before downloading weights files')
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)
    # weights = None
    #
    # if pretrained:
    #     if distributed:
    #         if local_rank == 0:
    #             weights = download_weights(phi)
    #         torch.distributed.barrier()
    #     else:
    #         weights = download_weights(phi)

        # 将加载的预训练权重应用到你的模型中
        # model.load_state_dict(weights)
    print("Downloading weights successfully!")
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained)
    # model.load_state_dict(weights)
    if not pretrained:
        weights_init(model)
    # if pretrained:
    #     # 确保预训练权重已被正确加载
    #     if weights is not None:
    #         model_dict = model.state_dict()
    #         pretrained_dict = OrderedDict({k: v for k, v in weights.items() if k in model_dict})
    #         print("Pretrained keys:", pretrained_dict.keys())
    #         model_dict.update(pretrained_dict)
    #         model.load_state_dict(pretrained_dict)
    #         print("Model state_dict keys:", model_dict.keys())
    #     else:
    #         raise ValueError("Pretrained weights were not loaded properly.")
    # else:
    #     weights_init(model)
    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # print("根据预训练权重的Key和模型的Key进行加载")
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        print("显示没有匹配上的Key")
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print('Record loss')
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    print('model trained!')

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        print("Sync_bn supports in distributed.")
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        # if distributed:
        #     #----------------------------#
        #     #   多卡平行运行
        #     #----------------------------#
        #     print(f"多卡平行运行之前 (rank = {rank}, local_rank = {local_rank})")
        #     model_train = model_train.cuda(local_rank)
        #     print(f"多卡平行运行1 (rank = {rank}, local_rank = {local_rank})")
        #     dist.barrier()  # 确保所有进程同步
        #     model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        #     print(f"多卡平行运行 (rank = {rank}, local_rank = {local_rank})")
        #     dist.barrier()  # 确保所有进程同步
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            print(f"多卡平行运行之前 (rank = {rank}, local_rank = {local_rank})")
            model_train = model_train.cuda(local_rank)
            print(f"多卡平行运行1 (rank = {rank}, local_rank = {local_rank})")
            try:
                dist.barrier()  # 确保所有进程同步
            except Exception as e:
                print(f"Barrier failed at rank = {rank} with exception: {e}")
            print('确保所有进程同步！')
            try:
                print('torch.nn.parallel.DistributedDataParallel 之前')
                model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                        find_unused_parameters=False)
                print(f"多卡平行运行 (rank = {rank}, local_rank = {local_rank})")
            except Exception as e:
                print(f"DDP failed at rank = {rank} with exception: {e}")
            # print('torch.nn.parallel.DistributedDataParallel 成功运行！')
            try:
                dist.barrier()  # 确保所有进程同步
                print(f"Barrier passed at rank = {rank}")
            except Exception as e:
                print(f"Barrier failed at rank = {rank} with exception: {e}")
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    # 添加日志以确保每个进程进入训练阶段
    if local_rank == 0:
        print("进入训练阶段")

    print('读取数据集对应的txt')
    with open(os.path.join(VOCdevkit_path, r"ImageSets\Segmentation\train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, r"ImageSets\Segmentation\val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        wanted_step = 1.5e4 if optimizer_type == "adamw" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch


    if True:
        UnFreeze_flag = False

        print('backocne in freeze_train')
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]


        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        # print('before training')
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                print('before backcone')
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
                dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            #------------------------- 清理现存，避免显存泄露 ----------------------#
            torch.cuda.empty_cache()
            # ------------------------- 清理现存，避免显存泄露 ----------------------#

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
