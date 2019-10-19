import os
from unet import No_skip_CNN
from read_data import *
from plot_curve import *
from metric import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

batch_size = 1
high_dim = 64
savedir = './output/data/no_skip_for_hdim64_cediceloss/plot/'
# Initial network
net = No_skip_CNN(n_channels=1, n_classes=4, high_represent_dim=high_dim)
net.cuda()
net.eval()
net.load_state_dict(torch.load('./output/data/no_skip_for_hdim64_cediceloss/CP30.pth'))

# Read data
ch2, ch4 = import_2Ddata_for_hdim_val()
training_set_result_tot = np.zeros((1,12),dtype=np.float32)
validation_set_result_tot = np.zeros((1,12),dtype=np.float32)

for pat in range(50):
    print('Patient : {}'.format(pat + 1))
    imgs_ch2 = ch2[pat,]
    imgs_ch4 = ch4[pat,]
    imgs_ch2 = torch.from_numpy(imgs_ch2).unsqueeze(dim=0).cuda()
    imgs_ch4 = torch.from_numpy(imgs_ch4).unsqueeze(dim=0).cuda()
    seg_ED_ch2, hdim_ED_ch2 = net(imgs_ch2[:, 0, ].unsqueeze(dim=0))
    seg_ES_ch2, hdim_ES_ch2 = net(imgs_ch2[:, 1, ].unsqueeze(dim=0))
    seg_ED_ch4, hdim_ED_ch4 = net(imgs_ch4[:, 0, ].unsqueeze(dim=0))
    seg_ES_ch4, hdim_ES_ch4 = net(imgs_ch4[:, 1, ].unsqueeze(dim=0))
    print(hdim_ED_ch2, hdim_ES_ch4)
    ch2_ED_mask = np.array((torch.max(seg_ED_ch2, dim=1)[1]).cpu())
    ch2_ES_mask = np.array((torch.max(seg_ES_ch2, dim=1)[1]).cpu())
    ch4_ED_mask = np.array((torch.max(seg_ED_ch4, dim=1)[1]).cpu())
    ch4_ES_mask = np.array((torch.max(seg_ES_ch4, dim=1)[1]).cpu())
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

    validation_set_result = cal_dice_hd([ch2_ED_mask,ch2_ES_mask,ch4_ED_mask,ch4_ES_mask],[CH2_ED_gt,CH2_ES_gt,CH4_ED_gt,CH4_ES_gt])
    validation_set_result_tot = validation_set_result + validation_set_result_tot

print('Validation DSC ED_endo: {}'.format(validation_set_result_tot[0,0] / 50))
print('Validation DSC ED_epi: {}'.format(validation_set_result_tot[0,1] / 50))
print('Validation DSC ED_LA: {}'.format(validation_set_result_tot[0,2] / 50))
print('Validation DSC ES_endo: {}'.format(validation_set_result_tot[0,3] / 50))
print('Validation DSC ES_epi: {}'.format(validation_set_result_tot[0,4] / 50))
print('Validation DSC ES_LA: {}'.format(validation_set_result_tot[0,5] / 50))
print('Validation HD ED_endo: {}'.format(validation_set_result_tot[0,6] / 50))
print('Validation HD ED_epi: {}'.format(validation_set_result_tot[0,7] / 50))
print('Validation HD ED_LA: {}'.format(validation_set_result_tot[0,8] / 50))
print('Validation HD ES_endo: {}'.format(validation_set_result_tot[0,9] / 50))
print('Validation HD ES_epi: {}'.format(validation_set_result_tot[0,10] / 50))
print('Validation HD ES_LA: {}'.format(validation_set_result_tot[0,11] / 50))
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