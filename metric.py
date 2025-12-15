import os, sys
import cv2
import numpy as np
from multiprocessing import Pool 
#import copy_reg
import pickle
import types
import argparse
parser = argparse.ArgumentParser()
import os
from PIL import Image

# 添加命令行参数 --test_ids
#parser.add_argument('--test_ids', type=str, help=r'E:\UTB-finish\UTB_master\out_mask')
#parser.add_argument('--pred_dir', type=str, default=r'E:\UTB-finish\UTB_master\results', help=r'E:\UTB-finish\UTB_master\results')
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

#pickle.dump(types.MethodType, _pickle_method)

def get_iou(data_list, class_num, save_path=r'E:\UTB-finish\UTB_master'):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))


    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

#混淆矩阵类，用于计算个处理分类问题的混淆矩阵
class ConfusionMatrix(object):
    def __init__(self, nclass, classes=None):
        self.nclass = 39
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    #gt表示真实类别，pred表示预测类别
    def add(self, gt, pred):
        assert (len(gt) == len(pred))

        # 调整断言条件
        assert (np.max(pred) <= self.nclass)

        # 更新混淆矩阵
        for i in range(len(gt)):
            self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])
            Recall=recall/self.nclass
        print(Recall)


    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])
        Accuracy=accuracy/self.nclass
        print(Accuracy)
#平均Jaccard指数（也称为IoU）
    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m



def main():
    # label_folder = r'E:\UTB-finish\UTB_master\Train_dataset\test_mask'  # 存储标签图像的文件夹路径
    # pred_folder = r'E:\UTB-finish\UTB_master\mytest01'  # 存储预测图像的文件夹路径
    groundtruth_path = r'E:\UTB-finish\UTB_master\Train_dataset\test_mask'
    prediction_path = r'E:\UTB-finish\UTB_master\Train_dataset\111'
    res=get_corresponding_filenames(groundtruth_path,prediction_path)

    num_images = len(res['pred'])  # 每个文件夹中图像的数量
    # 创建ConfusionMatrix对象
    cm = ConfusionMatrix(nclass=39)
    nclass = 39
    for i in range(num_images):
        # 加载真实标签图像和预测标签图像
        label_path =  os.path.join(prediction_path,res['pred'][i])
        pred_path =os.path.join(groundtruth_path,res['truth'][i])

        label_image = Image.open(label_path).convert('L')
        pred_image = Image.open(pred_path).convert('L')

        # 获取图像的宽度和高度
        width, height = label_image.size

        # 将图像转换为一维数组
        gt = list(label_image.getdata())
        pred = list(pred_image.getdata())

        # 添加样本到混淆矩阵
        cm.add(gt, pred)

    # 计算并打印准确率和Jaccard指数
    accuracy = cm.accuracy()
    jaccard, jaccard_perclass, matrix = cm.jaccard()
    print(f"准确率: {accuracy}")
    print(f"平均Jaccard指数: {jaccard}")
    print(f"每个类别的Jaccard指数: {jaccard_perclass}")

    # 显示混淆矩阵
    print("混淆矩阵:")
    print(matrix)

import os

import os


def get_corresponding_filenames(prediction_path, groundtruth_path):
    """Finds corresponding filenames between two directories and returns a dictionary with keys 'pred' and 'truth'.

    Args:
        prediction_path (str): Path to the directory containing prediction images.
        groundtruth_path (str): Path to the directory containing groundtruth images.

    Returns:
        dict: Dictionary with keys 'pred' and 'truth' containing corresponding filenames.
    """

    prediction_filenames = os.listdir(prediction_path)
    groundtruth_filenames = os.listdir(groundtruth_path)

    # Create a dictionary to store corresponding filenames
    corresponding_filenames = {}

    for prediction_filename in prediction_filenames:
        # Extract the base filename (without extension)
        prediction_base_filename = os.path.splitext(prediction_filename)[0]

        # Find the corresponding groundtruth filename with the same base name
        for groundtruth_filename in groundtruth_filenames:
            groundtruth_base_filename = os.path.splitext(groundtruth_filename)[0]

            if prediction_base_filename == groundtruth_base_filename:
                corresponding_filenames[prediction_filename] = groundtruth_filename
                break

    # Convert to a dictionary with 'pred' and 'truth' keys
    pred_filenames = list(corresponding_filenames.keys())
    truth_filenames = list(corresponding_filenames.values())

    corresponding_data = {"pred": pred_filenames, "truth": truth_filenames}

    return corresponding_data

if __name__ == '__main__':
    main()





