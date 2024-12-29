# 为了适应改变的分割器，适配pvt2文件
# 增加了损失值绘制内容
# 使用两种loss
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt2 import PolypPVT
from util.dataloader import get_loader, test_dataset
from util.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import torch.nn as nn
import matplotlib.pyplot as plt


# 损失函数，计算损失值
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def generalized_dice_coefficient(pred, mask):
    smooth = 1.
    y_true_f = torch.flatten(mask)
    y_pred_f = torch.flatten(pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def dice_loss(pred, mask):
    loss = 1 - generalized_dice_coefficient(mask, pred)
    return loss


def log_cosh_dice_loss(pred, mask):
    x = dice_loss(mask, pred)
    return torch.log(((torch.exp(x) + torch.exp(-x)) / 2.0))


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal * weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wfocal + wiou).mean()

def structure_loss_1(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    wlodl = log_cosh_dice_loss(pred, mask)
    wlodl = (wlodl * weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    return (wbce + wiou + wlodl).mean()


# 得到IoU数据
def test_iou(model, path, dataset):
    data_path = os.path.join(path, dataset)  # /dataset/TestDataset/CVC-300...等
    image_root = '{}/images/'.format(data_path)  # 测试集中的原始图像
    gt_root = '{}/masks/'.format(data_path)  # 测试集中的mask图像
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)  # 每张照片的大小是352*352的
    # IOU是记录总的IoU的，每一张照片之和
    IOU = 0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(image)
        res = lateral_map_4
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        output = res
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(output, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)  # 交集
        union = ((input_flat + target_flat) - intersection)  # 并集
        iou = (intersection.sum() + smooth) / (union.sum() + smooth)
        # dice = (2 * intersection.sum() + smooth) / (output.sum() + target.sum() + smooth)
        # dice = '{:.4f}'.format(dice)
        # dice = float(dice)
        # iou = dice / (2 - dice)
        IOU = IOU + iou
    IOU_num = IOU / num1
    IOU_num = '{:.3f}'.format(IOU_num)
    IOU_num = float(IOU_num)
    return IOU_num


# 得到DIC这个数据
def test_dice(model, path, dataset):
    data_path = os.path.join(path, dataset)  # /dataset/TestDataset/CVC-300...等
    image_root = '{}/images/'.format(data_path)  # 测试集中的原始图像
    gt_root = '{}/masks/'.format(data_path)  # 测试集中的mask图像
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)  # 每张照片的大小是352*352的
    # DSC是记录总的dice的，每一张照片之和
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(image)
        res = lateral_map_4
        # eval Dice
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        # dice = '{:.4f}'.format(dice)
        # dice = float(dice)
        DSC = DSC + dice
    DSC_num = DSC / num1
    DSC_num = '{:.3f}'.format(DSC_num)
    DSC_num = float(DSC_num)
    # 这里的DSC/num1就是mDic
    return DSC_num


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best_dice
    global best_iou
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()  # 记录损失值
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare数据准备 ----
            images, gts = pack
            images = Variable(images).cuda()  # 原始图像
            gts = Variable(gts).cuda()  # mask的图像
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            map4, map3, map2, map1 = model(images)
            map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            loss = structure_loss_1(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(
                map4, gts)
            # ---- backward ----
            loss.backward()  # 反向传播
            clip_gradient(optimizer, opt.clip)
            optimizer.step()  # 梯度更新
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))  # loss_P1_record就是当前模型的损失值，在终端进行显示

    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + str(epoch) + 'PolypPVT.pth')
    # choose the best model
    # 画图使用
    global dict_plot_dice
    global dict_plot_iou

    # 调用test_dice函数计算dice值
    test1path = './dataset/TestDataset/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['ClinicDB', 'Kvasir', 'Endoscene', 'ETIS-Larib', 'ColonDB']:
            dataset_dice = test_dice(model, test1path, dataset)  # 计算dice，在日志进行显示
            dataset_iou = test_iou(model, test1path, dataset)  # 计算iou，在日志进行显示
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            logging.info('epoch: {}, dataset: {}, iou: {}'.format(epoch, dataset, dataset_iou))
            print(dataset, ' dice: ', dataset_dice)  # 在终端显示当前epoch每个数据集的dice值
            print(dataset, ' iou:', dataset_iou)  # 在终端显示当前epoch每个数据集的iou值
            dict_plot_dice[dataset].append(dataset_dice)
            dict_plot_iou[dataset].append(dataset_iou)
        # 上面的for是将每个数据集进行测试，得到每个数据集的dice和iou.mean是指测试test数据集，test数据集包含每个数据集中的一部分
        # meandice = test_dice(model, test_path, 'test')
        # meaniou = test_iou(model, test_path, 'test')
        # dict_plot_dice['test'].append(meandice)
        # dict_plot_iou['test'].append(meaniou)
        # if meandice > best_dice or meaniou > best_iou:
        #     best_dice = meandice
        #     best_iou = meaniou
        #     torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
        #     torch.save(model.state_dict(), save_path + str(epoch) + 'PolypPVT-best.pth')
        #     print('##############################################################################best_dice', best_dice)
        #     print('##############################################################################best_iou', best_iou)
        #     logging.info(
        #         '##############################################################################best_dice:{}'.format(
        #             best_dice))
        #     logging.info(
        #         '##############################################################################best_iou:{}'.format(
        #             best_iou))


# 绘制两个数据集的dice和iou
def plot_train_dice(dict_plot_dice=None, name=None):
    plt.subplot(2, 1, 1)
    a = np.array(dict_plot_dice[name[0]])
    b = np.array(dict_plot_dice[name[1]])
    max_indx_1 = np.argmax(a)
    max_indx_2 = np.argmax(b)
    # 图上展示的样式
    show_max = '[' + str(max_indx_1) + ' ' + str(a[max_indx_1]) + ']'
    show_max_2 = '[' + str(max_indx_2) + ' ' + str(b[max_indx_2]) + ']'
    color = ['red', 'blue']
    line = ['-']
    for i in range(len(name)):
        plt.plot(dict_plot_dice[name[i]], label=name[i], color=color[i], linestyle=line[0])
    # plt.xlabel("Epoch")
    # plt.grid(axis='x')
    plt.ylabel("Dice")
    # plt.title('Train_Dice')
    plt.annotate(show_max, xytext=(max_indx_1, a[max_indx_1]), xy=(max_indx_1, a[max_indx_1]))
    plt.annotate(show_max_2, xytext=(max_indx_2, b[max_indx_2]), xy=(max_indx_2, b[max_indx_2]))
    plt.legend()


def plot_train_iou(dict_plot_iou=None, name=None):
    plt.subplot(2, 1, 2)
    a = np.array(dict_plot_iou[name[0]])
    b = np.array(dict_plot_iou[name[1]])
    max_indx_1 = np.argmax(a)
    max_indx_2 = np.argmax(b)
    show_max = '[' + str(max_indx_1) + ' ' + str(a[max_indx_1]) + ']'
    show_max_2 = '[' + str(max_indx_2) + ' ' + str(b[max_indx_2]) + ']'
    color = ['green', 'orange']
    line = ['-']
    for i in range(len(name)):
        plt.plot(dict_plot_iou[name[i]], label=name[i], color=color[i], linestyle=line[0])
    # plt.grid(axis='x')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    # plt.title('Train_IoU')
    plt.annotate(show_max, xytext=(max_indx_1, a[max_indx_1]), xy=(max_indx_1, a[max_indx_1]))
    plt.annotate(show_max_2, xytext=(max_indx_2, b[max_indx_2]), xy=(max_indx_2, b[max_indx_2]))
    plt.legend()


def dataset_dice(dict_plot_dice=None, name=None):
    plt.subplot(2, 1, 1)
    a = np.array(dict_plot_dice[name[2]])
    b = np.array(dict_plot_dice[name[3]])
    c = np.array(dict_plot_dice[name[4]])
    max_indx_1 = np.argmax(a)
    max_indx_2 = np.argmax(b)
    max_indx_3 = np.argmax(c)
    # 图上展示的样式
    show_max = '[' + str(max_indx_1) + ' ' + str(a[max_indx_1]) + ']'
    show_max_2 = '[' + str(max_indx_2) + ' ' + str(b[max_indx_2]) + ']'
    show_max_3 = '[' + str(max_indx_3) + ' ' + str(b[max_indx_3]) + ']'
    color = ['red', 'blue', 'green']
    line = ['-']
    for i in range(len(name)):
        plt.plot(dict_plot_dice[name[i]], label=name[i], color=color[i], linestyle=line[0])
    # plt.xlabel("Epoch")
    # plt.grid(axis='x')
    plt.ylabel("Dice")
    # plt.title('Train_Dice')
    plt.annotate(show_max, xytext=(max_indx_1, a[max_indx_1]), xy=(max_indx_1, a[max_indx_1]))
    plt.annotate(show_max_2, xytext=(max_indx_2, b[max_indx_2]), xy=(max_indx_2, b[max_indx_2]))
    plt.annotate(show_max_3, xytext=(max_indx_3, c[max_indx_3]), xy=(max_indx_3, c[max_indx_3]))
    plt.legend()


def dataset_iou(dict_plot_iou=None, name=None):
    plt.subplot(2, 1, 2)
    a = np.array(dict_plot_iou[name[2]])
    b = np.array(dict_plot_iou[name[3]])
    c = np.array(dict_plot_iou[name[4]])
    max_indx_1 = np.argmax(a)
    max_indx_2 = np.argmax(b)
    max_indx_3 = np.argmax(c)
    # 图上展示的样式
    show_max = '[' + str(max_indx_1) + ' ' + str(a[max_indx_1]) + ']'
    show_max_2 = '[' + str(max_indx_2) + ' ' + str(b[max_indx_2]) + ']'
    show_max_3 = '[' + str(max_indx_3) + ' ' + str(b[max_indx_3]) + ']'
    color = ['orange', 'deepskyblue', 'darkorchid']
    line = ['-']
    for i in range(len(name)):
        plt.plot(dict_plot_dice[name[i]], label=name[i], color=color[i], linestyle=line[0])
    # plt.grid(axis='x')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    # plt.title('Train_Dice')
    plt.annotate(show_max, xytext=(max_indx_1, a[max_indx_1]), xy=(max_indx_1, a[max_indx_1]))
    plt.annotate(show_max_2, xytext=(max_indx_2, b[max_indx_2]), xy=(max_indx_2, b[max_indx_2]))
    plt.annotate(show_max_3, xytext=(max_indx_3, c[max_indx_3]), xy=(max_indx_3, c[max_indx_3]))
    plt.legend()


if __name__ == '__main__':
    # dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
    #              'test': []}
    # name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    # 这里dict_plot_dice和dict_plot_iou分别为两个字典
    dict_plot_dice = {'ClinicDB': [], 'Kvasir': [], 'Endoscene': [], 'ColonDB': [], 'ETIS-Larib': []}
    dict_plot_iou = {'ClinicDB': [], 'Kvasir': [], 'Endoscene': [], 'ColonDB': [], 'ETIS-Larib': []}
    name = ['ClinicDB', 'Kvasir', 'Endoscene', 'ColonDB', 'ETIS-Larib']
    ##################model_name#############################
    model_name = 'PolypPVT'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/' + model_name + '/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PolypPVT().cuda()

    best_dice = 0
    best_iou = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)  # train数据中的原始图像
    gt_root = '{}/masks/'.format(opt.train_path)  # train数据中的mask图像

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    # dict_plot_dice已经在train这个函数中进行了更新
    plot_train_dice(dict_plot_dice, name)
    plot_train_iou(dict_plot_iou, name)
    plt.savefig('eval.png')
    # 绘制其他三个数据集的dice和iou
    # dataset_dice(dict_plot_dice, name)
    # dataset_iou(dict_plot_iou, name)
    # plt.savefig('dataset.png')

    plt.show()

    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
