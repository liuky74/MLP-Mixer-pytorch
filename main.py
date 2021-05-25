import torch
import os
import numpy as np
from tqdm import tqdm
from data_package.smoke_dataset import SmokeDataset
from model_package.mlp_mixer import MLPMixer
from model_package.resnet import resnet18, resnet34,resnext50_32x4d
from torch.utils.data import DataLoader
from data_package.data_transform import VideoTransform

root_dirs = [
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/base_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/DeAn_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/ZhangYe_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/XinXiang_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/HeNeng_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/TongHua_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/GuRun_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/YunJing_dataset",
    # "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/WanZai_dataset",

    # "D:\data\smoke_car\MLP_data\\base_dataset",
    # "D:\data\smoke_car\MLP_data\\DeAn_dataset",
    # "D:\data\smoke_car\MLP_data\\ZhangYe_dataset",
    # "D:\data\smoke_car\MLP_data\\XinXiang_dataset",
    # "D:\data\smoke_car\MLP_data\\HeNeng_dataset",
    # "D:\data\smoke_car\MLP_data\\TongHua_dataset",
    # "D:\data\smoke_car\MLP_data\\GuRun_dataset",
    # "D:\data\smoke_car\MLP_data\\YunJing_dataset"
]
test_dris = [
    # "D:\data\smoke_car\MLP_data\\WanZai_dataset",
    "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/WanZai_dataset",
    # "/home/liuky/HDD_1/data/smoke/train_data/smoke_classification_data/test_dataset",
]
load_weight = "weights/ResNet_C2_E60.snap"
save_model_name = "ResNet"
batch_size = 16
init_lr = 0.01
lr_steps = [50, 100, 150, 200, 250]
start_epoch = 0
max_epoch = 300
use_cuda = False


def train():
    # model = MLPMixer(in_channels=96,
    #                  num_patch=25 * 25,
    #                  patch_size=25,
    #                  num_classes=2,
    #                  dim=512,
    #                  depth=8,
    #                  token_dim=256,
    #                  channel_dim=2048
    #                  )
    model = resnext50_32x4d(
        True if start_epoch > 0 else True,
        num_classes=2)
    if use_cuda:
        model = model.cuda()
    if len(load_weight) > 0 and start_epoch > 0:
        print("|INFO|loading model:%s|" % load_weight)
        static_dict = torch.load(load_weight)
        model.load_state_dict(static_dict)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam([{"params": model.parameters(), "initial_lr": init_lr}], lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, lr_steps, 0.1, last_epoch=start_epoch)
    dataset = SmokeDataset(root_dirs=root_dirs,
                           transform=VideoTransform(size=100, flip_p=0.5, std=255.,
                                                    use_bright_contrast=True,
                                                    horizontal_flip=True,
                                                    vertical_flip=True,
                                                    random_sample=False)

                           )
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=SmokeDataset.collate)
    test_dataset = SmokeDataset(root_dirs=test_dris,
                                transform=VideoTransform(size=100, flip_p=0.5, std=255.,
                                                         use_bright_contrast=False,
                                                         horizontal_flip=False,
                                                         vertical_flip=False,
                                                         random_sample=False)
                                )
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=SmokeDataset.collate)
    for epoch in range(start_epoch, max_epoch):
        process_bar = tqdm(data_loader, ncols=180)
        total_loss = 0
        total_acc = 0

        model.eval()
        for idx, (data, label) in enumerate(test_data_loader):
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            logit = model(data)
            pred = torch.argmax(logit, 1)
            acc = pred == label
            acc = acc.sum() / batch_size
            total_acc += acc
        print("\n|INFO|acc:%.4f|" % (total_acc / (idx + 1)))

        model.train()
        for idx, (data, label) in enumerate(process_bar):
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            logit = model(data)
            optim.zero_grad()
            loss = criterion.forward(input=logit, target=label)
            loss.backward()
            optim.step()
            total_loss += loss.data
            process_bar.desc = "|INFO|epoch:%d|step:%d|loss:%.4f/%.4f|lr:%f|" % (
                epoch, idx, loss.data, total_loss.data / (idx + 1), optim.param_groups[0]["lr"])
        lr_scheduler.step()

        # test
        if (epoch % 10 == 0) and epoch > 0:
            save_path = os.path.abspath('weights/%s_C2_E%d.snap' % (save_model_name,epoch))
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            torch.save(model.state_dict(), save_path)
            print("\n|INFO|save model in %s|" % save_path)


if __name__ == '__main__':
    train()
    from torchvision.models import resnext50_32x4d
