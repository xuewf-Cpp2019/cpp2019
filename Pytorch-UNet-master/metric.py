import torch
from dice_loss import dice,dice_coeff
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from biGRU_Unet import bi_unet
from read_data import *
def hausdorff_distance(output, segment):
    hd1 = directed_hausdorff(output, segment)[0]
    hd2 = directed_hausdorff(segment, output)[0]
    return np.max([hd1, hd2])

def multi_class_dice(ED_seg, ES_seg, gt): # 返回每个病人的各项指标的dice
    ED_mask = torch.max(ED_seg, dim=1)[1]
    ES_mask = torch.max(ES_seg, dim=1)[1]
    ED_gt = gt[:, 0, ]
    ES_gt = gt[:, 1, ]
    dice_ED_endo = dice((ED_mask == 1).float(), (ED_gt == 1).float())
    dice_ED_epi = dice((ED_mask == 1).float()+(ED_mask == 2).float(), (ED_gt == 1).float()+(ED_gt == 2).float())
    dice_ED_LA = dice((ED_mask == 3).float(), (ED_gt == 3).float())
    dice_ES_endo = dice((ES_mask == 1).float(), (ES_gt == 1).float())
    dice_ES_epi = dice((ES_mask == 1).float()+(ES_mask == 2).float(), (ES_gt == 1).float()+(ES_gt == 2).float())
    dice_ES_LA = dice((ES_mask == 3).float(), (ES_gt == 3).float())
    return  dice_ED_endo, dice_ED_epi, dice_ED_LA, dice_ES_endo, dice_ES_epi, dice_ES_LA

def multi_class_hd(ED_seg, ES_seg, gt):
    ED_mask = torch.max(ED_seg, dim=1)[1]
    ES_mask = torch.max(ES_seg, dim=1)[1]
    ED_gt = gt[:, 0, ]
    ES_gt = gt[:, 1, ]
    #print(ED_mask.device,ED_gt.device)
    hd_ED_endo = torch.zeros(ED_seg.shape[0],)
    hd_ED_epi = torch.zeros(ED_seg.shape[0],)
    hd_ED_LA = torch.zeros(ED_seg.shape[0],)
    hd_ES_endo = torch.zeros(ED_seg.shape[0],)
    hd_ES_epi = torch.zeros(ED_seg.shape[0],)
    hd_ES_LA = torch.zeros(ED_seg.shape[0],)
    for i in range(ED_seg.shape[0]):
        hd_ED_endo[i,] = hausdorff_distance((ED_mask[i,] == 1).float(), (ED_gt[i,] == 1).float()).item()
        hd_ED_epi[i,] = hausdorff_distance((ED_mask[i,] == 1).float()+(ED_mask[i,] == 2).float(), (ED_gt[i,] == 1).float()+(ED_gt[i,] == 2).float()).item()
        hd_ED_LA[i,] = hausdorff_distance((ED_mask[i,] == 3).float(), (ED_gt[i,] == 3).float()).item()
        hd_ES_endo[i,] = hausdorff_distance((ES_mask[i,] == 1).float(), (ES_gt[i,] == 1).float()).item()
        hd_ES_epi[i,] = hausdorff_distance((ES_mask[i,] == 1).float()+(ES_mask[i,] == 2).float(), (ES_gt[i,] == 1).float()+(ES_gt[i,] == 2).float()).item()
        hd_ES_LA[i,] = hausdorff_distance((ES_mask[i,] == 3).float(), (ES_gt[i,] == 3).float()).item()
    return  hd_ED_endo, hd_ED_epi, hd_ED_LA, hd_ES_endo, hd_ES_epi, hd_ES_LA

def multi_class_dice_test(ED_mask, ES_mask, ED_gt, ES_gt):
    dice_ED_endo = dice((ED_mask == 1).float(), (ED_gt == 1).float())
    dice_ED_epi = dice((ED_mask == 1).float() + (ED_mask == 2).float(), (ED_gt == 1).float() + (ED_gt == 2).float())
    dice_ED_LA = dice((ED_mask == 3).float(), (ED_gt == 3).float())
    dice_ES_endo = dice((ES_mask == 1).float(), (ES_gt == 1).float())
    dice_ES_epi = dice((ES_mask == 1).float() + (ES_mask == 2).float(), (ES_gt == 1).float() + (ES_gt == 2).float())
    dice_ES_LA = dice((ES_mask == 3).float(), (ES_gt == 3).float())
    return  dice_ED_endo, dice_ED_epi, dice_ED_LA, dice_ES_endo, dice_ES_epi, dice_ES_LA

def multi_class_hd_test(ED_mask, ES_mask, ED_gt, ES_gt):
    hd_ED_endo = hausdorff_distance((ED_mask == 1).float(), (ED_gt == 1).float()).item()
    hd_ED_epi = hausdorff_distance((ED_mask == 1).float()+(ED_mask == 2).float(), (ED_gt == 1).float()+(ED_gt == 2).float()).item()
    hd_ED_LA = hausdorff_distance((ED_mask == 3).float(), (ED_gt == 3).float()).item()
    hd_ES_endo = hausdorff_distance((ES_mask == 1).float(), (ES_gt == 1).float()).item()
    hd_ES_epi = hausdorff_distance((ES_mask == 1).float()+(ES_mask == 2).float(), (ES_gt == 1).float()+(ES_gt == 2).float()).item()
    hd_ES_LA = hausdorff_distance((ES_mask == 3).float(), (ES_gt == 3).float()).item()
    return  hd_ED_endo, hd_ED_epi, hd_ED_LA, hd_ES_endo, hd_ES_epi, hd_ES_LA

def cal_dice_hd(mask,gt):
    # resize pred_mask to ori shape
    # calculate dice of ori_size_mask
    DSC_ED_endo_ch2, DSC_ED_epi_ch2, DSC_ED_LA_ch2, DSC_ES_endo_ch2, DSC_ES_epi_ch2, DSC_ES_LA_ch2 = multi_class_dice_test(
        ED_mask=torch.from_numpy(mask[0]), ES_mask=torch.from_numpy(mask[1]),
        ED_gt=torch.from_numpy(gt[0]), ES_gt=torch.from_numpy(gt[1]))
    DSC_ED_endo_ch4, DSC_ED_epi_ch4, DSC_ED_LA_ch4, DSC_ES_endo_ch4, DSC_ES_epi_ch4, DSC_ES_LA_ch4 = multi_class_dice_test(
        ED_mask=torch.from_numpy(mask[2]), ES_mask=torch.from_numpy(mask[3]),
        ED_gt=torch.from_numpy(gt[2]), ES_gt=torch.from_numpy(gt[3]))
    HD_ED_endo_ch2, HD_ED_epi_ch2, HD_ED_LA_ch2, HD_ES_endo_ch2, HD_ES_epi_ch2, HD_ES_LA_ch2 = multi_class_hd_test(
        ED_mask=torch.from_numpy(np.squeeze(mask[0])), ES_mask=torch.from_numpy(np.squeeze(mask[1])),
        ED_gt=torch.from_numpy(np.squeeze(gt[0])), ES_gt=torch.from_numpy(np.squeeze(gt[1])))
    HD_ED_endo_ch4, HD_ED_epi_ch4, HD_ED_LA_ch4, HD_ES_endo_ch4, HD_ES_epi_ch4, HD_ES_LA_ch4 = multi_class_hd_test(
        ED_mask=torch.from_numpy(np.squeeze(mask[2])), ES_mask=torch.from_numpy(np.squeeze(mask[3])),
        ED_gt=torch.from_numpy(np.squeeze(gt[2])), ES_gt=torch.from_numpy(np.squeeze(gt[3])))

    dice_ED_endo = (DSC_ED_endo_ch2 + DSC_ED_endo_ch4) / 2
    dice_ED_epi = (DSC_ED_epi_ch2 + DSC_ED_epi_ch4) / 2
    dice_ED_LA = (DSC_ED_LA_ch2 + DSC_ED_LA_ch4) / 2
    dice_ES_endo = (DSC_ES_endo_ch2 + DSC_ES_endo_ch4) / 2
    dice_ES_epi = (DSC_ES_epi_ch2 + DSC_ES_epi_ch4) / 2
    dice_ES_LA = (DSC_ES_LA_ch2 + DSC_ES_LA_ch4) / 2
    hd_ED_endo = (HD_ED_endo_ch2 + HD_ED_endo_ch4) / 2
    hd_ED_epi = (HD_ED_epi_ch2 + HD_ED_epi_ch4) / 2
    hd_ED_LA = (HD_ED_LA_ch2 + HD_ED_LA_ch4) / 2
    hd_ES_endo = (HD_ES_endo_ch2 + HD_ES_endo_ch4) / 2
    hd_ES_epi = (HD_ES_epi_ch2 + HD_ES_epi_ch4) / 2
    hd_ES_LA = (HD_ES_LA_ch2 + HD_ES_LA_ch4) / 2
    return [dice_ED_endo.item(),dice_ED_epi.item(),dice_ED_LA.item(),dice_ES_endo.item(),dice_ES_epi.item(),
             dice_ES_LA.item(),hd_ED_endo,hd_ED_epi,hd_ED_LA,hd_ES_endo,hd_ES_epi,hd_ES_LA]
