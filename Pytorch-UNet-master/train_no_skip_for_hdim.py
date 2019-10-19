import sys
import os
from optparse import OptionParser  #语法分析器，帮助文档
import torch.backends.cudnn as cudnn
from torch import optim
from unet import No_skip_CNN
from metric import *
from dice_loss import *
from plot_curve import *
from output_data import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def adjust_learning_rate(optimizer, decay_rate=.85):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train_net(net,
              epochs=32,
              batch_size=16,
              lr=1e-4,
              save_cp=True,  #save checkpoints
              gpu=True,
              #high_dim = 64
              ):

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, 800,
               100, str(save_cp), str(gpu)))

    train, train_gt, val, val_gt = import_2Ddata_for_hdim()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    criterion_CE_Dice = CE_Dice_loss()

    epoch_loss_CE = np.zeros((epochs,),dtype=np.float32)
    epoch_loss_DSC = np.zeros((epochs,),dtype=np.float32)
    epoch_loss_Total = np.zeros((epochs,),dtype=np.float32)

    epochs_dice_ED_endo_train = np.zeros((epochs,train.shape[0]), dtype=np.float32)
    epochs_dice_ED_epi_train = np.zeros((epochs,train.shape[0]), dtype=np.float32)
    epochs_dice_ED_la_train = np.zeros((epochs,train.shape[0]), dtype=np.float32)
    epochs_dice_ES_endo_train = np.zeros((epochs,train.shape[0]), dtype=np.float32)
    epochs_dice_ES_epi_train = np.zeros((epochs,train.shape[0]), dtype=np.float32)
    epochs_dice_ES_la_train = np.zeros((epochs,train.shape[0]), dtype=np.float32)

    epochs_dice_ED_endo_val = np.zeros((epochs,val.shape[0]),dtype=np.float32)
    epochs_dice_ED_epi_val = np.zeros((epochs,val.shape[0]),dtype=np.float32)
    epochs_dice_ED_la_val = np.zeros((epochs,val.shape[0]),dtype=np.float32)
    epochs_dice_ES_endo_val = np.zeros((epochs,val.shape[0]),dtype=np.float32)
    epochs_dice_ES_epi_val = np.zeros((epochs,val.shape[0]),dtype=np.float32)
    epochs_dice_ES_la_val = np.zeros((epochs,val.shape[0]),dtype=np.float32)

    epochs_hd_ED_endo_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_hd_ED_epi_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_hd_ED_la_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_hd_ES_endo_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_hd_ES_epi_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_hd_ES_la_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)

    epochs_hd_ED_endo_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ED_epi_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ED_la_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ES_endo_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ES_epi_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ES_la_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)

    for epoch in range(epochs):
        if epoch > 10:
            adjust_learning_rate(optimizer)
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        epoch_loss = 0
        ce_loss = 0
        dsc_loss = 0
        for batch_i in range(train.shape[0] // batch_size): #
            imgs = train[batch_i * batch_size : (batch_i+1) * batch_size ,]
            gt = train_gt[batch_i * batch_size : (batch_i+1) * batch_size ,]
            imgs = torch.from_numpy(imgs).cuda()
            gt = torch.from_numpy(gt).cuda()
            ED_seg = net(imgs[:,0,].unsqueeze(dim=1)) # ED_seg.shape : batch_size * 4 * 256 * 256
            ES_seg = net(imgs[:,1,].unsqueeze(dim=1))
            # CE + Dice loss
            CEloss, DiceLoss = criterion_CE_Dice(ED_seg=ED_seg, ES_seg=ES_seg, gt=gt)
            loss = (CEloss + DiceLoss) / 2

            ED_endo, ED_epi, ED_LA, ES_endo, ES_epi, ES_LA = multi_class_dice(ED_seg=ED_seg, ES_seg=ES_seg, gt=gt)
            #print(batch_i,batch_size,ED_endo.shape)
            epochs_dice_ED_endo_train[epoch, batch_i * batch_size : (batch_i+1) * batch_size] = np.array(ED_endo)
            epochs_dice_ED_epi_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(ED_epi)
            epochs_dice_ED_la_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(ED_LA)
            epochs_dice_ES_endo_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(ES_endo)
            epochs_dice_ES_epi_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(ES_epi)
            epochs_dice_ES_la_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(ES_LA)
            hd_ED_endo, hd_ED_epi, hd_ED_LA, hd_ES_endo, hd_ES_epi, hd_ES_LA = multi_class_hd(ED_seg=ED_seg.cpu(), ES_seg=ES_seg.cpu(), gt=gt.cpu())
            epochs_hd_ED_endo_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(hd_ED_endo)
            epochs_hd_ED_epi_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(hd_ED_epi)
            epochs_hd_ED_la_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(hd_ED_LA)
            epochs_hd_ES_endo_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(hd_ES_endo)
            epochs_hd_ES_epi_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(hd_ES_epi)
            epochs_hd_ES_la_train[epoch, batch_i * batch_size: (batch_i + 1) * batch_size] = np.array(hd_ES_LA)
            print('Epoch : {} --- batch : {}/{} --- CE loss : {}'.format(epoch + 1, batch_i + 1, train.shape[0] // batch_size, CEloss.item()))
            print('Epoch : {} --- batch : {}/{} --- Dice loss : {}'.format(epoch + 1, batch_i + 1, train.shape[0] // batch_size, DiceLoss.item()))
            print('Epoch : {} --- batch : {}/{} --- Total loss : {}'.format(epoch + 1, batch_i + 1, train.shape[0] // batch_size, loss.item()))

            epoch_loss = epoch_loss + loss.item()
            ce_loss = ce_loss + CEloss.item()
            dsc_loss = dsc_loss + DiceLoss.item()
            optimizer.zero_grad()     #梯度置0
            loss.backward()         #根据loss计算梯度
            optimizer.step()       #更新所有参数

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (train.shape[0] // batch_size) ))           #所有batch的平均loss
        epoch_loss_CE[epoch,] = ce_loss / (train.shape[0] // batch_size)
        epoch_loss_DSC[epoch,] = dsc_loss / (train.shape[0] // batch_size)
        epoch_loss_Total[epoch,] = epoch_loss / (train.shape[0] // batch_size)

        if 1:
            net.eval()
            for batch_j in range(val.shape[0] // batch_size):
                imgs = val[batch_j * batch_size: (batch_j + 1) * batch_size, ]
                gt = val_gt[batch_j * batch_size: (batch_j + 1) * batch_size, ]
                imgs = torch.from_numpy(imgs).cuda()
                gt = torch.from_numpy(gt).cuda()
                val_ED = net(imgs[:, 0, ].unsqueeze(dim=1))
                val_ES = net(imgs[:, 1, ].unsqueeze(dim=1))
                ED_endo, ED_epi, ED_LA, ES_endo, ES_epi, ES_LA = multi_class_dice(ED_seg = val_ED, ES_seg = val_ES, gt = gt)
                epochs_dice_ED_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ED_endo)
                epochs_dice_ED_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ED_epi)
                epochs_dice_ED_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ED_LA)
                epochs_dice_ES_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ES_endo)
                epochs_dice_ES_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ES_epi)
                epochs_dice_ES_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ES_LA)

                hd_ED_endo, hd_ED_epi, hd_ED_LA, hd_ES_endo, hd_ES_epi, hd_ES_LA = multi_class_hd(ED_seg=val_ED.cpu(),ES_seg=val_ES.cpu(), gt=gt.cpu())

                epochs_hd_ED_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ED_endo)
                epochs_hd_ED_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ED_epi)
                epochs_hd_ED_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ED_LA)
                epochs_hd_ES_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ES_endo)
                epochs_hd_ES_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ES_epi)
                epochs_hd_ES_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ES_LA)

            print('Validation Dice Coeff ED_endo: {} ± {} '.format(np.mean(epochs_dice_ED_endo_val[epoch,]), np.std(epochs_dice_ED_endo_val[epoch,])))
            print('Validation Dice Coeff ED_epi: {} ± {} '.format(np.mean(epochs_dice_ED_epi_val[epoch,]), np.std(epochs_dice_ED_epi_val[epoch,])))
            print('Validation Dice Coeff ED_LA: {} ± {} '.format(np.mean(epochs_dice_ED_la_val[epoch,]), np.std(epochs_dice_ED_la_val[epoch,])))
            print('Validation Dice Coeff ES_endo: {} ± {} '.format(np.mean(epochs_dice_ES_endo_val[epoch,]), np.std(epochs_dice_ES_endo_val[epoch,])))
            print('Validation Dice Coeff ES_epi: {} ± {} '.format(np.mean(epochs_dice_ES_epi_val[epoch,]), np.std(epochs_dice_ES_epi_val[epoch,])))
            print('Validation Dice Coeff ES_LA: {} ± {} '.format(np.mean(epochs_dice_ES_la_val[epoch,]), np.std(epochs_dice_ES_la_val[epoch,])))

        torch.save(net.state_dict(), './CP_no_skip_for_hdim/' + 'CP{}.pth'.format(epoch + 1))
        print('Checkpoint saved !')

    output_dice_hd_pat(
        [epochs_dice_ED_endo_train, epochs_dice_ED_epi_train, epochs_dice_ED_la_train, epochs_dice_ES_endo_train,
         epochs_dice_ES_epi_train, epochs_dice_ES_la_train, epochs_hd_ED_endo_train, epochs_hd_ED_epi_train,
         epochs_hd_ED_la_train, epochs_hd_ES_endo_train, epochs_hd_ES_epi_train, epochs_hd_ES_la_train],
        [epochs_dice_ED_endo_val, epochs_dice_ED_epi_val, epochs_dice_ED_la_val, epochs_dice_ES_endo_val,
         epochs_dice_ES_epi_val, epochs_dice_ES_la_val, epochs_hd_ED_endo_val, epochs_hd_ED_epi_val,
         epochs_hd_ED_la_val,epochs_hd_ES_endo_val, epochs_hd_ES_epi_val, epochs_hd_ES_la_val],
            './output/data/no_skip_for_hdim128_cediceloss/')

    plot_loss_dsc_curve([epoch_loss_CE, epoch_loss_DSC, epoch_loss_Total],
                        [np.mean(epochs_dice_ED_endo_val, axis=1), np.mean(epochs_dice_ED_epi_val,axis=1),
                         np.mean(epochs_dice_ED_la_val,axis=1), np.mean(epochs_dice_ES_endo_val, axis=1),
                         np.mean(epochs_dice_ES_epi_val,axis=1), np.mean(epochs_dice_ES_la_val,axis=1)],
                        [np.mean(epochs_hd_ED_endo_val, axis=1), np.mean(epochs_hd_ED_epi_val,axis=1),
                         np.mean(epochs_hd_ED_la_val,axis=1), np.mean(epochs_hd_ES_endo_val, axis=1),
                         np.mean(epochs_hd_ES_epi_val,axis=1), np.mean(epochs_hd_ES_la_val,axis=1)],
                        [np.mean(epochs_dice_ED_endo_train, axis=1), np.mean(epochs_dice_ED_epi_train,axis=1),
                         np.mean(epochs_dice_ED_la_train,axis=1), np.mean(epochs_dice_ES_endo_train, axis=1),
                         np.mean(epochs_dice_ES_epi_train,axis=1), np.mean(epochs_dice_ES_la_train,axis=1)],
                        './output/data/no_skip_for_hdim128_cediceloss/')

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=32, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=5,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default= 1E-4,
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
    args = get_args()

    high_dim = 128
    load_model = False
    net = No_skip_CNN(n_channels=1, n_classes=4, high_represent_dim=high_dim)
    net.cuda()
    if load_model:
        net.load_state_dict(torch.load('./CP_no_skip_for_hdim/CP25.pth'))
        print('Model loaded from {}'.format(args.load))
    cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
              #    high_dim = high_dim
                  )
    except KeyboardInterrupt:
        #torch.save(encoder_net.state_dict(), 'INTERRUPTED.pth')    #获取模型参数state_dict
        print('Interrupt!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




