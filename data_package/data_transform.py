import cv2
import numpy as np
tmp_file_idx = 0
# 这个类每次进行预处理的时候都会重新声明,所以init每次预处理都会执行,随机数并不会被固定
class VideoTransform():
    def __init__(self, size,
                 flip_p=0.5,  # 反转概率
                 mean=127,  # 通道均值
                 std=1.,
                 horizontal_flip= False,  # 随机水平翻转
                 vertical_flip = False,  # 随机垂直翻转
                 random_sample = False,  # 随机重采样
                 use_bright_contrast = False,  # 随机调整亮度对比度

                 ):
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, list):
            self.size = (size[0], size[1])
        else:
            self.size = (size, size)
        self.flip_p = flip_p
        self.mean = mean
        self.std = std
        self.use_horizontal_flip = horizontal_flip
        self.use_vertical_flip =vertical_flip
        self.use_random_sample =random_sample
        self.use_bright_contrast = use_bright_contrast

        self.constrast_p = np.random.rand()
        self.brightness_p = np.random.rand()
        self.horizontal_p = np.random.rand()  # 翻转的随机数
        self.vertical_p = np.random.rand()
        self.sample_p = np.random.rand()

        if self.brightness_p>0.5: # 亮度
            self.beta = 0
        else:
            self.beta = np.random.uniform(-50,50)
        if self.constrast_p >0.5: # 对比度
            self.alpha = 1
        else:
            self.alpha = np.random.uniform(0.7, 1.3)

        # 随机裁剪大小
        self.sample_new_h = np.random.randint(int(self.size[1] / 2), self.size[1])
        self.sample_new_w = np.random.randint(int(self.size[0] / 2), self.size[0])
        # 随机裁剪起点坐标
        self.sample_new_y = np.random.randint(0, self.size[1] - self.sample_new_h)
        self.sample_new_x = np.random.randint(0, self.size[0] - self.sample_new_w)
        # 随即裁剪填充起点
        self.sample_padding_y = np.random.randint(0, self.size[1] - self.sample_new_h)
        self.sample_padding_x = np.random.randint(0, self.size[0] - self.sample_new_w)

    # 随机亮度&对比度变换
    def bright_constrast_adjust(self,data):

        blank = np.zeros_like(data)
        data = cv2.addWeighted(data,self.alpha,blank,1-self.alpha,self.beta)
        return data

    # 水平翻转
    def horizontal_flip(self,
                        type,  # 需要反转的类型,接受图像或者是lebel
                        data):
        if self.horizontal_p > self.flip_p:
            return data
        if type == 'img':  # 针对图像做水平反转
            img = data
            img = img[:, ::-1].copy()
            return img
        elif type == 'label':  # 针对leibel做水平反转
            label = data
            for label_idx in range(len(label)):
                label_t = label[label_idx]
                label_t[0] = 1. - label_t[0]
                label_t[2] = 1. - label_t[2]
                label_t = label_t[[2, 1, 0, 3, 4]]  # 左右翻转后会从左上右下点变成右上左下点，重新修正回左上右下的格式
                label[label_idx] = label_t
            return label.copy()
        else:
            raise Exception('wrong')

    # 垂直翻转
    def vertical_flip(self, type, data):
        if self.vertical_p > self.flip_p:
            return data
        if type == 'img':
            img = data
            img = img[::-1, :].copy()
            return img
        elif type == 'label':
            label = data
            for label_idx in range(len(label)):
                label_t = label[label_idx]
                label_t[1] = 1. - label_t[1]
                label_t[3] = 1. - label_t[3]
                label_t = label_t[[0, 3, 2, 1, 4]]  # 上下翻转后会从左上右下点变成右上左下点，重新修正回左上右下的格式
                label[label_idx] = label_t
            return label.copy()
        else:
            raise Exception('wrong')

    def __call__(self, data):
        video = data
        # todo
        res_video = np.empty((self.size[0], self.size[1],video.shape[-1]), dtype=np.float32)
        # res_video = np.empty((self.size[0], self.size[1],video.shape[-2], video.shape[-1]), dtype=np.float32)

        # img
        for idx in range(video.shape[-1]):
            # resize
            img = video[...,idx]
            if self.size != img.shape[:2]:
                img = cv2.resize(img, self.size)
            # 亮度对比度
            if self.use_bright_contrast:
                img = self.bright_constrast_adjust(img)
            # # 裁切
            # if self.use_random_sample:
            #     img = self.rand_sample("img", img)
            # 翻转
            if self.use_horizontal_flip:
                img = self.horizontal_flip('img',img)
            if self.use_vertical_flip:
                img = self.vertical_flip('img',img)

            img = img.astype(np.float32)
            img = img - self.mean
            img = img * (self.std / 255.)
            res_video[...,idx] = img.copy()
        return res_video


class ImgBrightAndContrastTransform():
    def __init__(self):
        self.rand_bright = np.random

def show(video, label):
    img = video[-1]
    h, w, c = img.shape
    for label_t in label:
        label_t = label_t * (w, h, w, h, 1)
        label_t = label_t.astype(np.int32)
        cv2.rectangle(img, (label_t[0], label_t[1]), (label_t[2], label_t[3]), (0, 0, 255))
    cv2.imwrite("tmp.jpg", img)
