import sys
import os
from optparse import OptionParser  #语法分析器，帮助文档
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim  #优化算法optimize

from .eval import eval_net_output_mask,eval_net
from .unet import UNet
from .utils import get_ids, split_ids , get_imgs_and_masks, batch
from sklearn.model_selection import StratifiedKFold

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

j = 0
f = open('./output_epi_ori_Adam_1e_5_deconv_se8_12_bilinear.txt', 'a')
print('output file: output_epi_ori_Adam_1e_5_deconv_se8_12_bilinear.txt')


def adjust_learning_rate(optimizer, decay_rate=.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    id = np.array(tpr - fpr)
    maxindex = np.argmax(id)
    print('最佳阈值:' , threshold[maxindex])
    print('最佳阈值:', threshold[maxindex],file=f)
    #计算最佳阈值
    #print(fpr.shape,tpr.shape,threshold.shape)
    #roc_auc = auc(fpr, tpr)  ###计算auc的值
    '''
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('1.jpg')
    '''
    return threshold[maxindex]
    #plt.show()


def train_net(iddataset,
              net,
              epochs=5,
              batch_size=4,
              lr=0.0001,
              #val_percent=0.2,  #Validation size
              save_cp=True,  #save checkpoints
              gpu=True,
              img_scale=1,
              cuda_device=0,
              ifold=1):



    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),  #.format
               len(iddataset['val']), str(save_cp), str(gpu)),file=f)

    N_train = len(iddataset['train'])   #训练集数

    dir_checkpoint = 'checkpoints_endo/'

    val_dice_data = np.zeros((epochs,))

    #mask_pred = np.zeros((580,80,80))
    #batch_num = math.ceil(N_train / batch_size)
    #first_epoch_loss = np.zeros((batch_num*5,))
    #k = 0
    #every_fold_epoch_loss = np.zeros((1,epochs),dtype=np.float32)
    for epoch in range(epochs):
        if epoch > 0:
            adjust_learning_rate(optimizer)
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs),file=f)
        net.train()      #train时和eval时Dropout等操作不同

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale,suffix)  #len=1860，93个病人
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale,suffix)    #len=460，23个病人

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):     #465个batch，当batch_size=4
            #每个batch中包含4个元组，每个元组4张图像及其对应的mask
            #i从1到465，b为其序号对应的batch的内容,b中包含4个元组，一个元组包含image和masks
            #先执行for i in b
            imgs = np.array([i[0] for i in b]).astype(np.float32)   #shape为(4,1,80,80)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda(cuda_device)
                true_masks = true_masks.cuda(cuda_device)

            masks_pred = net(imgs)  #网络最后是一个sigmoid方程

            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)      #计算loss，计算一个batch（batch_size=4）中每张图像的loss矩阵80*80，求平均
            epoch_loss += loss.item()
            #print('{} fold --- {} epoch --- {:.4f} --- loss: {:.6f}'.format(ifold, epoch+1, i * batch_size / N_train, loss.item()),file=f)
            optimizer.zero_grad()     #梯度置0
            loss.backward()         #根据loss计算梯度
            optimizer.step()       #更新所有参数

        print('{} fold: Epoch {} finished ! Loss: {}'.format(ifold, epoch+1, epoch_loss / i))           #所有batch的平均loss
        print('{} fold: Epoch {} finished ! Loss: {}'.format(ifold, epoch+1, epoch_loss / i),file=f)           #所有batch的平均loss
        #epoch_loss_data[epoch,] = epoch_loss / i

        if 1:
            val_dice = eval_net(net, val, gpu,cuda_device)          #把测试集做成val的格式，用这个函数即可
            #print('Validation Dice Coeff: {}'.format(val_dice),file=f)
            print('Validation Dice Coeff: {}'.format(val_dice))
            val_dice_data[epoch] = val_dice

    true_masks,pred_mask = eval_net_output_mask(net, list(get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale,suffix)), gpu=True, cuda_device=cuda_device)
            #every_fold_epoch_loss[0, epoch] = val_dice

    if save_cp:
            torch.save(net.state_dict(),          #保存参数
                       dir_checkpoint + '3rd_{}_flod_CP{}.pth'.format(ifold,epoch + 1))
            print('{} flod Checkpoint {} saved !'.format(ifold,epoch + 1),file=f)
    return true_masks,pred_mask,val_dice
'''

    plt.subplot(121)
    plt.plot(range(epochs, ), val_dice_data, label='epoch-val_dice')
    plt.xlabel('epoch')
    plt.ylabel('val_dice_data')
    plt.legend()  # legend 是在图区显示label，即上面 .plot()方法中label参数的值
    plt.subplot(122)
    plt.plot(range(epochs, ), epoch_loss_data, label='epoch-loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()  # legend 是在图区显示label，即上面 .plot()方法中label参数的值
    plt.subplots_adjust(wspace = 0.4)
    #plt.savefig("./eval_data/eval_endo.jpg")

    plt.savefig("./eval_data/eval_endo.jpg")
'''


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=32, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1E-2,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=1, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    i_fold = 0
    args = get_args()
    k_fold = 5
    cuda_device = 3
    net = UNet(n_channels=1, n_classes=1)  # 初始化net
    print(net,file=f)
    val_percent = 0.2
    val_outputmap = np.zeros((k_fold,580,80,80),dtype=np.float32)
    val_truemask = np.zeros((k_fold,580,80,80),dtype=np.float32)
    dice = np.zeros((k_fold),dtype=np.float32)
    dir_img = 'data/train/'
    dir_mask = 'data/train_masks_endo/'
    # dir_mask = 'data/train_masks_epi/'

    # dir_checkpoint = 'checkpoints_epi/'
    # suffix = '_masks_epi.gif'
    suffix = '_masks_endo.gif'

    ids = get_ids(dir_img)  # 不同于一般的函数会一次性返回包含了所有数值的数组，生成器一次只产生一个值，这样消耗的内存大大减少
    ids = list(ids)
    ids = np.array(ids, dtype=np.int32)
    ids.sort()  # 排序防止验证集病人图像混入训练集
    ids = ids[0:2900]
    y = np.ones(len(ids),)
    sFolder = StratifiedKFold(n_splits=5)




    criterion = nn.BCELoss()
   # fold_epoch_loss = np.zeros((k_fold,args.epochs),dtype=np.float32)
    k = 0
    for train,val in sFolder.split(ids,y):
        optimizer = optim.SGD(net.parameters(),
                              lr=1e-2,
                              momentum=0.9,
                              weight_decay=0.0005)
        '''
            optimizer = optim.Adam(net.parameters(),
                                   lr = 1e-3,
                                   weight_decay = 0.0005)
        '''
        k = k + 1
        train = train + 1
        val = val + 1
        train = split_ids(train)
        val = split_ids(val)
        data = {'train':list(train),'val':list(val)}
        #print(data)
        if args.load:
            net.load_state_dict(torch.load(args.load))        #用来加载模型参数

            print('Model loaded from {}'.format(args.load),file=f)

        if args.gpu:
            net.cuda(cuda_device)
        cudnn.benchmark = True # faster convolutions, but more memory

        try:
            true_masks,pred_mask,val_dice = train_net(iddataset=data,
                      net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      gpu=args.gpu,
                      img_scale=args.scale,
                      cuda_device=cuda_device,
                      ifold = k)
            #print(mask.shape,mask.dtype)
            val_outputmap[k-1,:,:,:] = pred_mask
            val_truemask[k-1,:,:,:] = true_masks
            dice[k-1] = val_dice
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')    #获取模型参数state_dict
            print('Saved interrupt',file=f)
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    #print('fold_epoch_loss：\n',fold_epoch_loss,file=f)
    #print('Avg_fold_loss： ', np.mean(fold_epoch_loss), file=f)
    #print('last_epoch_fold_loss: ',fold_epoch_loss[:,-1], file=f)



    truemask = val_truemask.flatten()
    predmask = val_outputmap.flatten()
    threshold = acu_curve(truemask,predmask)

    #计算每个病人的Dice  shape(5,580,80,80)
    val_predictmask = np.array(val_outputmap>0.5,dtype=np.float32)
    tot = 0
    eps = 0.0001
    tot_dice = 0
    #pretot_score = 0
    for i in range(k):
        for u in range(580):
            if ((u+1) % 20==0):
                tot_dice += tot/20
                tot = 0
            t_mask = val_truemask[i, u, :, :].flatten()
            p_mask = val_predictmask[i, u, :, :].flatten()
            #pre_score = precision_score(t_mask,p_mask)
            inter = np.dot(t_mask, p_mask)
            union = np.sum(t_mask) + np.sum(p_mask) + eps
            t = (2 * inter + eps) / union
            tot += t

    Avg_dice = tot_dice / 145
    print('Avg_dice: ',np.mean(dice))
    suffix = '_masks_endo.gif'
    # suffix = '_masks_epi.gif'

    savedir = './data/CV5_predict_endo/'
    savesuffix = '_CV5_masks_endo.gif'
    for i0_fold in range(5):
        for i in range(val_predictmask.shape[1]):
            mask_pre = Image.fromarray(val_predictmask[i0_fold, i, :, :] * 255)
        #  mask_pre
            mask_pre.save(savedir + str(i0_fold * 580 + i + 1) + savesuffix)

 #   print('Avg_dice: ', Avg_dice)
    f.close()
