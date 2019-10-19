import torch.nn as nn
import torch
import os
import numpy as np
from unet import encoder_image_sequences, bi_GRU, decoder_image_sequences
from read_data import *
from biGRU_Unet import bi_unet_val
from plot_curve import *
from metric import *
from PIL import Image
from output_data import *
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

batch_size = 1
high_dim = 64
savedir = './output/data/skip_rnn_hdim64_celoss/plot/'
# Initial network
encoder_net = encoder_image_sequences(n_channels=1, high_represent_dim=high_dim)
rnn_net = bi_GRU(input_dim=high_dim, hidden_dim=high_dim, num_layers=1)
decoder_net = decoder_image_sequences(n_classes=4)

encoder_net.cuda()
rnn_net.cuda()
decoder_net.cuda()
net = [encoder_net, rnn_net, decoder_net]
net[0].eval()
net[1].eval()
net[2].eval()
encoder_net.load_state_dict(torch.load('./output/data/skip_rnn_hdim64_celoss/CP16_encoder.pth'))
rnn_net.load_state_dict(torch.load('./output/data/skip_rnn_hdim64_celoss/CP16_biGRU.pth'))
decoder_net.load_state_dict(torch.load('./output/data/skip_rnn_hdim64_celoss/CP16_dncoder.pth'))
# Read data
ch2, ch4 = import_data_for_val()
#training_set_result_tot = np.zeros((1,12),dtype=np.float32)
validation_set_result = np.zeros((50,12),dtype=np.float32)
Before_rnn_dim = np.zeros((1000, 64),dtype=np.float32)
After_rnn_dim = np.zeros((1000, 64),dtype=np.float32)
seq = np.arange(0,10) + 1
hdim = np.arange(0,64) + 1
for pat in range(50):
    print('Patient : {}'.format(pat + 1))
    imgs_ch2 = ch2[pat,]
    imgs_ch4 = ch4[pat,]
    imgs_ch2 = torch.from_numpy(imgs_ch2).unsqueeze(dim=0).cuda()
    imgs_ch4 = torch.from_numpy(imgs_ch4).unsqueeze(dim=0).cuda()
    seg_ED_ch2, seg_ES_ch2, before_rnn_dim_ch2, after_rnn_dim_ch2 = bi_unet_val(net, imgs_ch2, 0, batch_size, high_dim) #cuda : 0
    seg_ED_ch4, seg_ES_ch4, before_rnn_dim_ch4, after_rnn_dim_ch4 = bi_unet_val(net, imgs_ch4, 0, batch_size, high_dim)
    ch2_ED_mask = np.array((torch.max(seg_ED_ch2, dim=1)[1]).cpu())
    ch2_ES_mask = np.array((torch.max(seg_ES_ch2, dim=1)[1]).cpu())
    ch4_ED_mask = np.array((torch.max(seg_ED_ch4, dim=1)[1]).cpu())
    ch4_ES_mask = np.array((torch.max(seg_ES_ch4, dim=1)[1]).cpu())
    #print(np.max(before_rnn_dim_ch2), np.max(after_rnn_dim_ch2))
    before_rnn_dim = np.concatenate((before_rnn_dim_ch2, before_rnn_dim_ch4), axis=0)
    after_rnn_dim = np.concatenate((after_rnn_dim_ch2, after_rnn_dim_ch4), axis=0)
    Before_rnn_dim[pat*20 : (pat+1)*20,] = before_rnn_dim
    After_rnn_dim[pat*20 : (pat+1)*20,] = after_rnn_dim
    #plot_3d_hdim_surfaces(seq, hdim, before_rnn_dim, './output/data/skip_rnn_hdim64_celoss/3d_surfaces/before_rnn_dim_' + str(pat+1) + '.jpg')
    #plot_3d_hdim_surfaces(seq, hdim, after_rnn_dim, './output/data/skip_rnn_hdim64_celoss/3d_surfaces/after_rnn_dim_' + str(pat+1) + '.jpg')
    CH2_ED_gt, CH2_ES_gt, CH2_ED, CH2_ES, CH4_ED_gt, CH4_ES_gt, CH4_ED, CH4_ES = read_ori_EDES(pat)
    ch2_ori_shape = CH2_ED_gt.shape
    ch4_ori_shape = CH4_ED_gt.shape
    #print(ch2_ED_mask.shape,ch2_ori_shape,ch4_ED_mask.shape,ch4_ori_shape)
    '''
    # 256 * 256
    CH2_ED = np.squeeze(resize_ori_image(CH2_ED, [256, 256]))
    CH2_ES = np.squeeze(resize_ori_image(CH2_ES, [256, 256]))
    CH4_ED = np.squeeze(resize_ori_image(CH4_ED, [256, 256]))
    CH4_ES = np.squeeze(resize_ori_image(CH4_ES, [256, 256]))
    CH2_ED_gt = resize_gt(CH2_ED_gt, [256, 256])
    CH2_ES_gt = resize_gt(CH2_ES_gt, [256, 256])
    CH4_ED_gt = resize_gt(CH4_ED_gt, [256, 256])
    CH4_ES_gt = resize_gt(CH4_ES_gt, [256, 256])

    plot_mask_gt_line(CH2_ED, np.squeeze(ch2_ED_mask), np.squeeze(CH2_ED_gt), savedir + 'CH2_ED' + str(pat) + '.png')
    plot_mask_gt_line(CH2_ES, np.squeeze(ch4_ED_mask), np.squeeze(CH4_ED_gt), savedir + 'CH4_ED' + str(pat) + '.png')
    plot_mask_gt_line(CH4_ED, np.squeeze(ch2_ES_mask), np.squeeze(CH2_ES_gt), savedir + 'CH2_ES' + str(pat) + '.png')
    plot_mask_gt_line(CH4_ES, np.squeeze(ch4_ES_mask), np.squeeze(CH4_ES_gt), savedir + 'CH4_ES' + str(pat) + '.png')

    validation_set_result = cal_dice_hd([ch2_ED_mask, ch2_ES_mask, ch4_ED_mask, ch4_ES_mask],[CH2_ED_gt, CH2_ES_gt, CH4_ED_gt, CH4_ES_gt])
    #print(validation_set_result)
    validation_set_result_tot = np.array(validation_set_result) + validation_set_result_tot
    '''
    #print(ch2_ED_mask.shape)
    # ori size
    ch2_ED_mask = resize_gt(ch2_ED_mask, [ch2_ori_shape[1], ch2_ori_shape[2]])
    ch2_ES_mask = resize_gt(ch2_ES_mask, [ch2_ori_shape[1], ch2_ori_shape[2]])
    ch4_ED_mask = resize_gt(ch4_ED_mask, [ch4_ori_shape[1], ch4_ori_shape[2]])
    ch4_ES_mask = resize_gt(ch4_ES_mask, [ch4_ori_shape[1], ch4_ori_shape[2]])
    #print(ch2_ED_mask.shape)
    plot_mask_gt_line(np.squeeze(CH2_ED), np.squeeze(ch2_ED_mask), np.squeeze(CH2_ED_gt), savedir + 'CH2_ED'+ str(pat+1) +'.png')
    plot_mask_gt_line(np.squeeze(CH4_ED), np.squeeze(ch4_ED_mask), np.squeeze(CH4_ED_gt), savedir + 'CH4_ED'+ str(pat+1) +'.png')
    plot_mask_gt_line(np.squeeze(CH2_ES), np.squeeze(ch2_ES_mask), np.squeeze(CH2_ES_gt), savedir + 'CH2_ES'+ str(pat+1) +'.png')
    plot_mask_gt_line(np.squeeze(CH4_ES), np.squeeze(ch4_ES_mask), np.squeeze(CH4_ES_gt), savedir + 'CH4_ES'+ str(pat+1) +'.png')

    validation_set_result[pat,] = cal_dice_hd([ch2_ED_mask,ch2_ES_mask,ch4_ED_mask,ch4_ES_mask],[CH2_ED_gt,CH2_ES_gt,CH4_ED_gt,CH4_ES_gt])
    #validation_set_result_tot = validation_set_result + validation_set_result_tot
output_dice_hd_val(validation_set_result, './output/data/skip_rnn_hdim64_celoss/test/')
output_hdim_val_before_rnn(Before_rnn_dim, './output/data/skip_rnn_hdim64_celoss/test/')
output_hdim_val_after_rnn(After_rnn_dim, './output/data/skip_rnn_hdim64_celoss/test/')

print('Validation DSC ED_endo: {}'.format(np.mean(validation_set_result[:,0])))
print('Validation DSC ED_epi: {}'.format(np.mean(validation_set_result[:,1])))
print('Validation DSC ED_LA: {}'.format(np.mean(validation_set_result[:,2])))
print('Validation DSC ES_endo: {}'.format(np.mean(validation_set_result[:,3])))
print('Validation DSC ES_epi: {}'.format(np.mean(validation_set_result[:,4])))
print('Validation DSC ES_LA: {}'.format(np.mean(validation_set_result[:,5])))
print('Validation HD ED_endo: {}'.format(np.mean(validation_set_result[:,6])))
print('Validation HD ED_epi: {}'.format(np.mean(validation_set_result[:,7])))
print('Validation HD ED_LA: {}'.format(np.mean(validation_set_result[:,8])))
print('Validation HD ES_endo: {}'.format(np.mean(validation_set_result[:,9])))
print('Validation HD ES_epi: {}'.format(np.mean(validation_set_result[:,10])))
print('Validation HD ES_LA: {}'.format(np.mean(validation_set_result[:,11])))

'''
print('Training DSC ED_endo: {}'.format(training_set_result_tot[0,0] / 400))
print('Training DSC ED_epi: {}'.format(training_set_result_tot[0,1] / 400))
print('Training DSC ED_LA: {}'.format(training_set_result_tot[0,2] / 400))
print('Training DSC ES_endo: {}'.format(training_set_result_tot[0,3] / 400))
print('Training DSC ES_epi: {}'.format(training_set_result_tot[0,4] / 400))
print('Training DSC ES_LA: {}'.format(training_set_result_tot[0,5] / 400))
print('Training HD ED_endo: {}'.format(training_set_result_tot[0,6] / 400))
print('Training HD ED_epi: {}'.format(training_set_result_tot[0,7] / 400))
print('Training HD ED_LA: {}'.format(training_set_result_tot[0,8] / 400))
print('Training HD ES_endo: {}'.format(training_set_result_tot[0,9] / 400))
print('Training HD ES_epi: {}'.format(training_set_result_tot[0,10] / 400))
print('Training HD ES_LA: {}'.format(training_set_result_tot[0,11] / 400))
'''