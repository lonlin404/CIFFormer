# test代码的作用就是使用我们训练好的模型将图像进行分割，得到我们预测出的mask图像
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt1 import PolypPVT
from util.dataloader import test_dataset
import cv2

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = PolypPVT().cuda()
    # 通过Train1文件跑出模型的权重
    model_weights_path = "model_pth/PolypPVT_2/84PolypPVT-best.pth"
    # 讲训练好的模型权重加到模型中
    model.load_state_dict(torch.load(model_weights_path))
    # 保存
    model.eval()
    # 输出结果，并且保存在result_map文件夹中
    for _data_name in ['Kvasir']:

        ##### put data_path here #####
        data_path = './dataset/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/PolypPVT/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()  # 注意图像数据一定要放到cuda上
            P1 = model(image)
            res = F.upsample(P1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res * 255)
        print(_data_name, 'Finish!')
