import argparse
import os
import sys
import random
import timeit
from email.policy import strict

import cv2
import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from deeplabv3plus import DeeplabV3plus, Res50_DeeplabV3plus

from discriminator import discriminator_tree
from loss import CrossEntropy2d
from data.voc_dataset import VOCDataSet, VOCDataSet_remain
#from data import get_loader, get_data_path
from data.augmentations import *

start = timeit.default_timer()

DATA_DIRECTORY = r'E:\UTB-finish\UTB_master\Train_dataset'
DATA_LIST_PATH = r'E:\UTB-finish\UTB_master\Train_dataset\image_label\train_list.txt'
DATA_LIST_PATH2 = r'E:\UTB-finish\UTB_master\Train_dataset\train_remain_list.txt'
#检查点：模型在特定训练阶段的参数和状态的快照
CHECKPOINT_DIR = './checkpoints/semi_res101deeplabv3plus/'

GPU_NUMBER=0
os.environ['CUDA_VISIBLE_DEVICES']='0'

IMG_MEAN = np.array((102.2058,110.1798,120.2015), dtype=np.float32)
NUM_CLASSES = 2 # 两个类别的细分

BATCH_SIZE = 8 #每次迭代批处理的数量
NUM_STEPS = 60000 #迭代总数
SAVE_PRED_EVERY = 5000 #每迭代5000次保存一次模型参数

INPUT_SIZE = '2500,2500' #输入图像调整为：321*321
IGNORE_LABEL = 255 # 255 for PASCAL-VOC / -1 for PASCAL-Context / 250 for Cityscapes
#预先训练的 ResNet-101 模型作为分段网络的骨干，并在特定数据集上对其进行微调。
RESTORE_FROM = r'E:\UTB-finish\UTB_master\pretrain_model\mobilenet_v2-b0353104.pth'
print(RESTORE_FROM)

#分割网络需要学习生成准确的分割掩码。学习率略高允许在训练期间进行更快的更新和调整。
LEARNING_RATE = 2.5e-4
#鉴别器网络需要区分真实分割掩码和生成的分割掩码。较低的学习率专注于识别两者之间的细微差异。
LEARNING_RATE_D = 1e-4
POWER = 0.9 #超参数
WEIGHT_DECAY = 0.0005 #权重衰减
MOMENTUM = 0.9 #动量
NUM_WORKERS = 4
RANDOM_SEED = 1234
#权重参数
LAMBDA_FM = 0.01
LAMBDA_ST = 1.0
## 置信度阈值
THRESHOLD_VALUE= 0.55
THRESHOLD_ST = 0.6


import torch

# # Check PyTorch version
# print(torch.__version__)
#
# # Check CUDA availability
# print(torch.cuda.is_available())
#
# # Check CUDA version
# print(torch.version.cuda)
#
# # Check cuDNN version
# print(torch.backends.cudnn.version())

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    #创建解析器
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    #加载参数
    #选择GPU、列出了数据集中的图像、列出了数据集中的图像的txt文件、模型保存的位置
    parser.add_argument("--gpu", type=int, default=GPU_NUMBER,
                        help="choose gpu device.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")

    parser.add_argument("--data-list2", type=str, default=DATA_LIST_PATH2,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save checkpoints of the model.")

    #batch-size
    #线程数
    #包含PASCAL VOC数据集的目录的路径

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    #图像输入大小
    #学习率
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="lambda_fm for feature-matching loss.")
    parser.add_argument("--lambda-st", type=float, default=LAMBDA_ST,
                        help="lambda_st for self-training.")
						
    parser.add_argument("--threshold-st", type=float, default=THRESHOLD_ST,
                        help="threshold_st for the self-training threshold.")
    parser.add_argument("--threshold-value", type=float, default=THRESHOLD_VALUE,
                        help="threshold_value for the self-training threshold.")						
						
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--ignore-label", type=float, default=IGNORE_LABEL,
                        help="label value to ignored for loss calculation")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of iterations.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")

    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")

    return parser.parse_args()
#解析参数
args = get_arguments()

def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda()  # Ignore label ??
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()  # N,H,W
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)  # N,C,H,W
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def compute_argmax_map(output):
    output = output.detach().cpu().numpy() #  c,H,W
    output = output.transpose((1,2,0))  # H,W,c
    output = np.asarray(np.argmax(output, axis=2), dtype=int) # H,W; obtain the index thatrepresented the max value through the axis==2 (i.e., channel)
    output = torch.from_numpy(output).float()  # numpy-->torch-->torch float 
    return output
     
def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):  # N,C
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        #print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)) # n,c,h,w
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)) # n,h,w
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]  # get the pred_all[*] map large than threshold value 
                label_sel[num_sel] = compute_argmax_map(pred_all[j]) # score map --> label map with channel==1

                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 

def compute_ignore_mask(pred0, max_pred):
    pred0 = pred0.detach() # c,H,W    
    pred = torch.chunk(torch.squeeze(pred0,0),2,dim=0)
    pred_1 = torch.squeeze(pred[0],0)	# 1,h,w-->h,w
    pred_1 = pred_1.cpu().numpy() 
    pred_1[pred_1 > args.threshold_value] = 0
    pred_1[pred_1 < 1-args.threshold_value] = 0
    pred_1[pred_1 > 0] = 255    #h,w
    max_pred = max_pred.cpu().numpy() 	
    mask = 	max_pred + pred_1
    mask[mask > 2] = 255  	
    mask =torch.from_numpy(mask) #h,w
    
    return mask	

def find_good_maps_new(D_outs, pred_all, pred_all_2):
    count = 0
    for i in range(D_outs.size(0)):  # N,C
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        #print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)) # n,c,h,w
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)) # n,h,w
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]  # c,h,w; get the pred_all[*] map large than threshold value 
                #label_sel[num_sel] = compute_argmax_map(pred_all[j]) # H,W; score map --> label map with channel==1
                label_sel[num_sel] = compute_ignore_mask( pred_all_2[j], compute_argmax_map(pred_all[j]) )
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 
				



criterion = nn.BCELoss()  #二元交叉熵损失

def main():
    print(args)   # Print args parameters, output the first statement

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print(input_size)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    model = DeeplabV3plus(num_classes=args.num_classes)  # Ensure this integrates your modified ASPP
    print(model)

    # Load pretrained parameters
    saved_state_dict = torch.load(args.restore_from)

    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
        else:
            print(f"Skipping loading parameter {name}, size mismatch or not found in state_dict.")

    model.load_state_dict(new_params)

    # Uncomment this if you have a specific path for a different model checkpoint
    # model.load_state_dict(torch.load('./checkpoints/voc_semi_4sGAN_threshold_0.5/VOC_20000.pth'))

    model.train()  # Set the model to training mode
    model.cuda()   # Move model to GPU

    cudnn.benchmark = True  # Enable benchmark mode for improved performance

    # Optionally, you might want to add additional setup here
    # For example, setting up data loaders, optimizers, etc.

    # Your training loop or further processing code can follow here


    # init D
    model_D = discriminator_tree(num_classes=args.num_classes)

    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
		
    #model_D.load_state_dict(torch.load('./checkpoints/voc_semi_4sGAN_threshold_0.5/VOC_20000_D.pth'))		
    model_D.train()
    model_D.cuda()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
		
    # load data and do preprocessing,such as rescale,flip   加载数据并进行预处理，如缩放、翻转
    print('len(args.data_dir):',len(args.data_dir))
    print(args.data_dir)
    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                         mean=IMG_MEAN)
    train_remain_dataset = VOCDataSet_remain(args.data_dir, args.data_list2, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)						


    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    trainloader_gt = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        
    trainloader_remain = data.DataLoader(train_remain_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    trainloader_remain_iter = iter(trainloader_remain)


    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)

    # optimizer for segmentation network 分割网络优化器
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.SGD(model_D.parameters(), lr=args.learning_rate_D, momentum=args.momentum,weight_decay=args.weight_decay)


    #optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    y_real_, y_fake_ = Variable(torch.ones(args.batch_size, 1).cuda()), Variable(torch.zeros(args.batch_size, 1).cuda())


    for i_iter in range(args.num_steps):
    #for i_iter in  range(20001, args.num_steps+1):    
        loss_ce_value = 0
        loss_D_value = 0
        loss_fm_value = 0
        loss_S_value = 0
        loss_st_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train Segmentation Network 
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False  #表示冻住某些层，只修改全连接层

        ########################## 1. 仅针对标记数据的训练损失  #############################
        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, _, _ = batch
        images = Variable(images).cuda()
        #pred = interp(model(images))  # deeplabv2
        pred, pred_aux1,_ = model(images)    # deeplabv3plus     
        
        
        loss_ce = loss_calc(pred, labels) # Cross entropy loss for labeled data
        loss_ce_aux = loss_calc(pred_aux1, labels)


        
        ############################ 2. 剩余未标记数据的训练损失 ####################
        try:
            batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)
        
        images_remain, _, _, _ = batch_remain
        images_remain = Variable(images_remain).cuda()
        #pred_remain = interp(model(images_remain))  # deeplabv2 
        pred_remain,_,_= model(images_remain)  # deeplabv3plus      
        ###################### 3. 将预测结果与输入图像连接起来 ####################
        images_remain = (images_remain-torch.min(images_remain))/(torch.max(images_remain)- torch.min(images_remain))
        #print (pred_remain.size(), images_remain.size())
		
		###############################################################################
        pred_remain_2 = F.softmax(pred_remain, dim=1)
        mask1 = torch.chunk(pred_remain_2,2,dim=1)
        pred_cat = torch.cat( ( images_remain, mask1[1] ), dim=1  ) 
        
        ###############################################################################
          
        D_out_z, D_out_y_pred = model_D(pred_cat) # predicts the D ouput 0-1 and feature map for FM-loss 
  
        # find predicted segmentation maps above threshold 
        pred_sel, labels_sel, count = find_good_maps_new(D_out_z, pred_remain, pred_remain_2) 

        # training loss on above threshold segmentation predictions (Cross Entropy Loss)
        if count > 0 and i_iter > 1000:
            loss_st = loss_calc(pred_sel, labels_sel)
        else:
            loss_st = 0.0

        ################ 4.连接输入的图像和为选区“真实”输入的真实地图 ###############
        try:
            batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = iter(trainloader_gt)
            batch_gt = next(trainloader_gt_iter)

        images_gt, labels_gt, _, _, _ = batch_gt
        # Converts grounth truth segmentation into 'num_classes' segmentation maps. 
        D_gt_v = Variable(one_hot(labels_gt)).cuda()
		
        images_gt = images_gt.cuda()
        images_gt = (images_gt - torch.min(images_gt))/(torch.max(images)-torch.min(images))
        ###############################################################################
        mask2 = torch.chunk(D_gt_v,2,dim=1)	

        D_gt_v_cat = torch.cat( ( images_gt, mask2[1] ), dim=1  ) 		
        ###############################################################################  
        D_out_z_gt , D_out_y_gt = model_D(D_gt_v_cat)
        
        # L1 loss for Feature Matching Loss
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
    
        if count > 0 and i_iter > 0: # if any good predictions found for self-training loss
            loss_S = loss_ce + 0.01*loss_ce_aux + args.lambda_fm*loss_fm + args.lambda_st*loss_st 
        else:
            loss_S = loss_ce + 0.01*loss_ce_aux + args.lambda_fm*loss_fm

        loss_S.backward()
        loss_fm_value+= loss_fm.item() 
        loss_st_value += loss_st
        loss_ce_value += loss_ce.item()
        loss_S_value += loss_S.item()

        ###################################################### 5.train D  #################################################
        for param in model_D.parameters():
            param.requires_grad = True  #表示训练上方的某一些网络层

        # train with pred
        pred_cat = pred_cat.detach()  # detach does not allow the graddients to back propagate.
        
        D_out_z, _ = model_D(pred_cat)
        y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
        loss_D_fake = criterion(D_out_z, y_fake_) 

        # train with gt
        D_out_z_gt , _ = model_D(D_gt_v_cat)
        y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda()) 
        loss_D_real = criterion(D_out_z_gt, y_real_)
        
        loss_D = (loss_D_fake + loss_D_real)/2.0
        loss_D.backward()
        loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()
        
        if i_iter %20 ==0:
            print('iter={0:5d}, loss_ce={1:.3f}, loss_fm={2:.3f}, loss_S={3:.3f}, loss_D={4:.3f}, loss_st={5:.3f}'.format(i_iter,loss_ce_value, loss_fm_value, loss_S_value, loss_D_value, loss_st_value))

        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('saving checkpoint  ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
