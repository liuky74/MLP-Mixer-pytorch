from data_package.basic_dataset import BasicDataset
from data_package.data_transform import VideoTransform
import cv2
import numpy as np
import os
import torch
import torchvision


class SmokeDataset(BasicDataset):

    def __getitem__(self, item):
        [data_file_path, img_label] = self.file_list[item]
        img_data = self._data_read(data_file_path)
        # img_label = self.target_transform(label=img_label)
        img_data, img_label = self.data_transform(img_data, img_label)
        return img_data, img_label

    def add_params(self):
        pass

    def set_resize(self, size):
        if isinstance(size, tuple):
            self.resize = size
        elif isinstance(size, list):
            self.resize = (size[0], size[1])
        elif isinstance(size, int):
            self.resize = (size, size)
        else:
            raise Exception('wrong')

    def target_transform(self,label):
        label_np = np.zeros(shape=[2],dtype=np.int)
        label_np[label]=1
        return label_np


    def data_transform(self, datas, labels):
        if self.transform is None:
            return datas, labels
        else:
            transform = torchvision.transforms.Compose([
                # VideoTransform(size=224, flip_p=0.5,std=255.,
                #                use_bright_contrast=True,
                #                horizontal_flip=True,
                #                vertical_flip=True,
                #                random_sample=False)
                self.transform
            ])
        datas = transform(datas)
        return datas, labels

    @staticmethod
    def collate(batch_data):
        res_label = []
        res_video = []
        for idx, (video, label) in enumerate(batch_data):
            res_video.append(video)
            # label = label.tolist()
            res_label.append(label)
        # [bbox_num,[x,y,x,y,cls,img_idx]]
        res_label = np.array(res_label, dtype=np.long)
        # 确保空间连续
        res_video = np.ascontiguousarray(np.array(res_video, dtype=np.float32).transpose((0, 3, 1, 2)))  # [N,C,H,W]
        return torch.from_numpy(res_video), torch.from_numpy(res_label).long()

    @staticmethod
    def MLP_collate(batch_data):
        res_label = []
        res_video = []
        for idx, (video, label) in enumerate(batch_data):
            res_video.append(video)
            # label = label.tolist()
            res_label.append(label)
        # [bbox_num,[x,y,x,y,cls,img_idx]]
        res_label = np.array(res_label, dtype=np.long)
        # 确保空间连续
        res_video = np.ascontiguousarray(np.array(res_video, dtype=np.float32).transpose((0, 4,3, 1, 2)))  # [N,C,H,W]

        return torch.from_numpy(res_video), torch.from_numpy(res_label).long()


    def _data_read(self, filename):
        img_data = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        img_data = np.split(img_data,12,0)
        # todo
        img_data = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)[np.newaxis,...] for img in img_data]
        img_data = np.concatenate(img_data, 0).transpose([1, 2, 0])
        # img_data = [img[np.newaxis, ...] for img in img_data]
        # img_data = np.concatenate(img_data, 0).transpose([1,2,3,0])



        # img_data = cv2.resize(img_data,(50,600))
        # cv2.imwrite("tmp1.jpg",img_data[:50,:,:])
        # cv2.imwrite("tmp2.jpg", img_data[50:100, :, :])
        # cv2.imwrite("tmp3.jpg", img_data[100:150, :, :])
        return img_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    smoke_data = SmokeDataset(root_dirs="D:\data\smoke_car\\rebuild_data_slim\\base_dataset\MLP_data",
                              data_folder_name="data",
                              label_folder_name="label")

    data_loader = DataLoader(smoke_data, batch_size=8, collate_fn=smoke_data.collate)
    data_iter = iter(data_loader)
    for data in data_iter:
        target = data[1]
        print('')
