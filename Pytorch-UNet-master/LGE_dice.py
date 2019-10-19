import read_data as rd
import numpy as np
import torch
import SimpleITK as sitk
import os
import cv2
from scipy.spatial.distance import directed_hausdorff
from PIL import Image
cuda_device = 1

def hausdorff_distance(output, segment):
    hd1 = directed_hausdorff(output, segment)[0]
    hd2 = directed_hausdorff(segment, output)[0]
    return np.max([hd1, hd2])

def dice(input,target):
    eps = 1e-8
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    if torch.sum(input) and torch.sum(target):
        t = (2 * inter.float() + eps) / union.float()
    elif torch.sum(input) or torch.sum(target):
        t = 0
    else: t = 1
    return t

os.makedirs("output/Test_output_PIL", exist_ok=True)
os.makedirs("output/Test_output_nii", exist_ok=True)
os.makedirs("output/Test_output_PIL_gt", exist_ok=True)

pred_path = './output/Test_output_nii/patient'
image_path = './data/C0LET2_nii45_for_challenge19/c0t2lge/patient'
label_path = './data/C0LET2_nii45_for_challenge19/lgegt/patient'
pred_noshape_path = './output/Test_output_nii_noshape/patient'
pred_unet_path = './output/Test_output_nii_unet/patient'

image_suffix = '_LGE.nii.gz'
lge_suffix = '_testLGE_predict.nii.gz'
label_suffix = '_LGE_manual.nii.gz'

num_patient = 5

pat_tot_mask1 = 0
pat_tot_mask2 = 0
pat_tot_mask3 = 0
pat_tot_mask = 0

Avg_dice_200_std = np.zeros((1, 5), dtype=np.float32)
Avg_dice_500_std = np.zeros((1, 5), dtype=np.float32)
Avg_dice_600_std = np.zeros((1, 5), dtype=np.float32)

#pat_tot_hd_mask1 = 0
#pat_tot_hd_mask2 = 0
#pat_tot_hd_mask3 = 0
#pat_tot_hd_mask = 0

for pat in range(5):
    image_dir = image_path + str(pat + 1) + image_suffix
    img = sitk.ReadImage(image_dir)
    img = sitk.GetArrayFromImage(img).astype(np.float32)
    lge_predict_dir = pred_path + str(pat + 1) + lge_suffix
    lge_predict = sitk.ReadImage(lge_predict_dir)
    lge_predict_mask = sitk.GetArrayFromImage(lge_predict).astype(np.float32)
    lge_predict_dir_noshape = pred_noshape_path + str(pat + 1) + lge_suffix
    lge_noshape_predict = sitk.ReadImage(lge_predict_dir_noshape)
    lge_predict_noshape_mask = sitk.GetArrayFromImage(lge_noshape_predict).astype(np.float32)
    lge_predict_unet_dir = pred_unet_path + str(pat + 1) + lge_suffix
    lge_predict_unet = sitk.ReadImage(lge_predict_unet_dir)
    lge_predict_unet_mask = sitk.GetArrayFromImage(lge_predict_unet).astype(np.float32)
    lge_gt_dir = label_path + str(pat + 1) + label_suffix
    lge_gt = sitk.ReadImage(lge_gt_dir)
    lge_gt_mask = sitk.GetArrayFromImage(lge_gt).astype(np.float32)
    mask1 = lge_gt_mask == 200
    mask2 = lge_gt_mask == 500
    mask3 = lge_gt_mask == 600
    lge_gt_mask[mask1] = 1
    lge_gt_mask[mask2] = 2
    lge_gt_mask[mask3] = 3
    mask1 = lge_predict_unet_mask == 200
    mask2 = lge_predict_unet_mask == 500
    mask3 = lge_predict_unet_mask == 600
    lge_predict_unet_mask[mask1] = 1
    lge_predict_unet_mask[mask2] = 2
    lge_predict_unet_mask[mask3] = 3
    mask1 = lge_predict_noshape_mask == 200
    mask2 = lge_predict_noshape_mask == 500
    mask3 = lge_predict_noshape_mask == 600
    lge_predict_noshape_mask[mask1] = 1
    lge_predict_noshape_mask[mask2] = 2
    lge_predict_noshape_mask[mask3] = 3
    mask1 = lge_predict_mask == 200
    mask2 = lge_predict_mask == 500
    mask3 = lge_predict_mask == 600
    lge_predict_mask[mask1] = 1
    lge_predict_mask[mask2] = 2
    lge_predict_mask[mask3] = 3




    for i in range(lge_predict_mask.shape[0]):
        ori_img = Image.fromarray(img[i,...])
        ori_img.save('./data/image/' + str(pat + 1) + 'patient_' + str(i + 1) + 'slice_ori_image.tif')
        pred = Image.fromarray(lge_predict_mask[i, ...])
        pred.save('./data/image/' + str(pat + 1) + 'patient_' + str(i + 1) + 'slice_pred.gif')
        pred_noshape = Image.fromarray(lge_predict_noshape_mask[i, ...])
        pred_noshape.save('./data/image/' + str(pat + 1) + 'patient_' + str(i + 1) + 'slice_pred_noshape.gif')
        pred_unet = Image.fromarray(lge_predict_unet_mask[i, ...])
        pred_unet.save('./data/image/' + str(pat + 1) + 'patient_' + str(i + 1) + 'slice_pred_unet.gif')
        gt = Image.fromarray(lge_gt_mask[i, ...])
        gt.save('./data/image/' + str(pat + 1) + 'patient_' + str(i + 1) + 'slice_gt.gif')


    gt_mask1 = (lge_gt_mask == 200).astype(np.float32)
    gt_mask2 = (lge_gt_mask == 500).astype(np.float32)
    gt_mask3 = (lge_gt_mask == 600).astype(np.float32)
    predict_mask1 = (lge_predict_mask == 200).astype(np.float32)
    predict_mask2 = (lge_predict_mask == 500).astype(np.float32)
    predict_mask3 = (lge_predict_mask == 600).astype(np.float32)

    # tot_hd_mask1 = 0
    # tot_hd_mask2 = 0
    # tot_hd_mask3 = 0

    tot_mask1 = dice(torch.from_numpy(gt_mask1), torch.from_numpy(predict_mask1))#.numpy()
    tot_mask2 = dice(torch.from_numpy(gt_mask2), torch.from_numpy(predict_mask2))#.numpy()
    tot_mask3 = dice(torch.from_numpy(gt_mask3), torch.from_numpy(predict_mask3))#.numpy()
    # tot_hd_mask1 += hausdorff_distance(gt_mask1[slice,...] , predict_mask1[slice,...])
    # tot_hd_mask2 += hausdorff_distance(gt_mask2[slice,...] , predict_mask2[slice,...])
    # tot_hd_mask3 += hausdorff_distance(gt_mask3[slice,...] , predict_mask3[slice,...])
    Avg_dice_200_std[0, pat] = tot_mask1
    Avg_dice_500_std[0, pat] = tot_mask2
    Avg_dice_600_std[0, pat] = tot_mask3
    pat_Avg_dice_mask = (tot_mask1 + tot_mask2 + tot_mask3) / 3

    print('Patinet {0:} : Avg validation Dice Coeff: {1:.4f}'.format(pat + 1, pat_Avg_dice_mask))
    print('Patinet {0:} : MYO validation Dice Coeff: {1:.4f}'.format(pat + 1, tot_mask1))
    print('Patinet {0:} : LV validation Dice Coeff: {1:.4f}'.format(pat + 1, tot_mask2))
    print('Patinet {0:} : RV validation Dice Coeff: {1:.4f}'.format(pat + 1, tot_mask3))

    # Avg_hd_mask1 = tot_hd_mask1 / len(l_imgs)
    # Avg_hd_mask2 = tot_hd_mask2 / len(l_imgs)
    # Avg_hd_mask3 = tot_hd_mask3 / len(l_imgs)
    # pat_Avg_hd_mask = (Avg_hd_mask1 + Avg_hd_mask2 + Avg_hd_mask3) / 3
    # print('Patinet {} : Avg validation HD: {}'.format(pat + 1, pat_Avg_hd_mask))
    # print('Patinet {} : MYO validation HD: {}'.format(pat + 1, Avg_hd_mask1))
    # print('Patinet {} : LV validation HD: {}'.format(pat + 1, Avg_hd_mask2))
    # print('Patinet {} : RV validation HD: {}'.format(pat + 1, Avg_hd_mask3))

    pat_tot_mask1 += tot_mask1
    pat_tot_mask2 += tot_mask2
    pat_tot_mask3 += tot_mask3
    # pat_tot_hd_mask1 += Avg_hd_mask1
    # pat_tot_hd_mask2 += Avg_hd_mask2
    # pat_tot_hd_mask3 += Avg_hd_mask3
    pat_tot_mask += pat_Avg_dice_mask
    # pat_tot_hd_mask += pat_Avg_hd_mask


AVG_dice_mask1 = (pat_tot_mask1 / 5)
AVG_dice_mask2 = (pat_tot_mask2 / 5)
AVG_dice_mask3 = (pat_tot_mask3 / 5)
#AVG_hd_mask1 = pat_tot_hd_mask1 / 5
#AVG_hd_mask2 = pat_tot_hd_mask2 / 5
#AVG_hd_mask3 = pat_tot_hd_mask3 / 5
AVG_dice_mask = (pat_tot_mask / 5)
#AVG_hd_mask = pat_tot_hd_mask / 5
print('AVG validation dice myo: {0:.4f}' .format(AVG_dice_mask1) )
print('AVG validation dice lv: {0:.4f} '.format(AVG_dice_mask2))
print('AVG validation dice rv: {0:.4f} '.format( AVG_dice_mask3))
#print('AVG validation hd myo: {}'.format(AVG_hd_mask1))
#print('AVG validation hd lv: {}'.format(AVG_hd_mask2))
#print('AVG validation hd rv: {}'.format(AVG_hd_mask3))
print('Avg validation Dice Coeff: {0:.4f} '.format( AVG_dice_mask))
#print('Avg validation HD: {}'.format(AVG_hd_mask))
print('Validation Dice Coeff_200_std: {}'.format(np.std(Avg_dice_200_std)))
print('Validation Dice Coeff_500_std: {}'.format(np.std(Avg_dice_500_std)))
print('Validation Dice Coeff_600_std: {}'.format(np.std(Avg_dice_600_std)))