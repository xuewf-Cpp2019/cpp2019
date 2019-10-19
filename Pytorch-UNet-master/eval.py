import torch
import numpy as np
import torch.nn.functional as F

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False, cuda_device=0):
    """Evaluation without the densecrf with the dice coefficient"""
    #epi阈值：        endo阈值：
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):    #无需用batch()
        img = b[0]          #shape:1*80*80
        true_mask = b[1]    #80*80

        img = torch.from_numpy(img).unsqueeze(0)          #shape:1*1*80*80
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)    #shape:1*80*80

        if gpu:
            img = img.cuda(cuda_device)
            true_mask = true_mask.cuda(cuda_device)

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()
        tot += dice_coeff(mask_pred, true_mask,cuda_device).item()

    return tot / (i + 1)

#def eval_
def eval_net_output_mask(net, dataset, gpu=False, cuda_device=0):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    #tot = 0
    mask_predict = np.zeros((len(dataset),80,80),dtype=np.float32)
    true_masks = np.zeros((len(dataset),80,80),dtype=np.int)

    SA1  = np.zeros((len(dataset),64, 80,80),dtype=np.float32)
    SA1_SA = np.zeros((len(dataset),64,80,80),dtype=np.float32)
    SA1_no_SA= np.zeros((len(dataset),64,80,80),dtype=np.float32)

    SA2 = np.zeros((len(dataset), 64, 80, 80), dtype=np.float32)
    SA2_SA = np.zeros((len(dataset), 64, 80, 80), dtype=np.float32)
    SA2_no_SA = np.zeros((len(dataset), 64, 80, 80), dtype=np.float32)



    for i, b in enumerate(dataset):    #无需用batch()
        img = b[0]          #shape:1*80*80
        true_mask = b[1]    #80*80

        img = torch.from_numpy(img).unsqueeze(0)          #shape:1*1*80*80
        #true_mask = torch.from_numpy(true_mask).unsqueeze(0)    #shape:1*80*80

        if gpu:
            img = img.cuda(cuda_device)

        true_masks[i, :, :] = true_mask
        mask_pred,SA1_, SA1_SA_, SA1_no_SA_, SA2_, SA2_SA_,SA2_no_SA_  = net(img)#[0]
        #mask_pred = (mask_pred > 0.5).float()
        mask_predict[i,:,:] = mask_pred.detach().cpu()
        SA1[i, :,:, :] = SA1_
        SA1_SA[i,:, :, :] = SA1_SA_
        SA1_no_SA[i,:, :, :] = SA1_no_SA_
        SA2[i, :,:, :] = SA2_
        SA2_SA[i,:, :, :] = SA2_SA_
        SA2_no_SA[i,:, :, :] = SA2_no_SA_




        #tot += dice_coeff(mask_pred, true_mask,cuda_device).item()

    return true_masks,mask_predict,SA1, SA1_SA, SA1_no_SA, SA2, SA2_SA,SA2_no_SA

def eval_net_BSSFP(net, dataset,cuda_device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot_1 = 0
    tot_2 = 0
    tot_3 = 0
    for i in range(len(dataset)):    #无需用batch()
        img = dataset[i][0]          #shape:80*80
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)       #shape:1*1*80*80
        gt = dataset[i][1]
        gt = torch.from_numpy(gt)
        img = img.cuda(cuda_device)
        gt = gt.cuda(cuda_device)
        predict_mask = net(img)[0]

        mask1 = (predict_mask[1,:,:] > predict_mask[0,:,:]).int() + (predict_mask[1,:,:] > predict_mask[2,:,:]).int() + (predict_mask[1,:,:] > predict_mask[3,:,:]).int()
        mask2 = (predict_mask[2,:,:] > predict_mask[0,:,:]).int() + (predict_mask[2,:,:] > predict_mask[1,:,:]).int() + (predict_mask[2,:,:] > predict_mask[3,:,:]).int()
        mask3 = (predict_mask[3,:,:] > predict_mask[0,:,:]).int() + (predict_mask[3,:,:] > predict_mask[1,:,:]).int() + (predict_mask[3,:,:] > predict_mask[2,:,:]).int()
        mask1 = mask1 == 3
        mask2 = mask2 == 3
        mask3 = mask3 == 3

        tot_1 += dice_coeff(mask1.float(), (gt == 1).float(), cuda_device).item()
        tot_2 += dice_coeff(mask2.float()+mask1.float(), (gt == 2).float()+(gt == 1).float(), cuda_device).item()
        tot_3 += dice_coeff(mask3.float(), (gt == 3).float(), cuda_device).item()

    return tot_1/200 , tot_2/200 , tot_3/200

def output_mask_BSSFP(net, dataset,gpu=False, cuda_device=3):
    net.eval()
    tot_200 = 0
    tot_500 = 0
    tot_600 = 0
    mask_pred_output = np.zeros((dataset.shape[0],dataset.shape[1],dataset.shape[2]),dtype=np.float32)
    for i in range(len(dataset)):    #无需用batch()
        img = dataset[i,:,:]          #shape:1*80*80
        #true_mask = gt[i, :, :]
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)       #shape:1*1*80*80

        #if gpu:
        img = img.cuda(cuda_device)
        mask_pred = net(img)  # [0]
        predict_mask = torch.from_numpy(np.zeros_like(img.cpu().squeeze(0).squeeze(0)))

        #true_mask = torch.from_numpy(true_mask)  # .cuda(cuda_device)
        mask1 = (mask_pred[0, 1, :, :] > mask_pred[0, 0, :, :]).int() + (
                    mask_pred[0, 1, :, :] > mask_pred[0, 2, :, :]).int() + (
                            mask_pred[0, 1, :, :] > mask_pred[0, 3, :, :]).int()
        mask2 = (mask_pred[0, 2, :, :] > mask_pred[0, 0, :, :]).int() + (
                    mask_pred[0, 2, :, :] > mask_pred[0, 1, :, :]).int() + (
                            mask_pred[0, 2, :, :] > mask_pred[0, 3, :, :]).int()
        mask3 = (mask_pred[0, 3, :, :] > mask_pred[0, 0, :, :]).int() + (
                    mask_pred[0, 3, :, :] > mask_pred[0, 1, :, :]).int() + (
                            mask_pred[0, 3, :, :] > mask_pred[0, 2, :, :]).int()
        mask1 = mask1 == 3
        mask2 = mask2 == 3
        mask3 = mask3 == 3
        predict_mask[mask1] = 1
        predict_mask[mask2] = 2
        predict_mask[mask3] = 3
        # predict_mask.cuda(cuda_device)
        # mask_pred = (mask_pred > 0.5).float()
        # print(mask_pred.shape,mask_pred.type)
        '''
        predict_mask_200 = (predict_mask == 1).float()
        gt_mask_200 = (true_mask == 1).float()

        predict_mask_500 = (predict_mask == 2).float()
        gt_mask_500 = (true_mask == 2).float()

        predict_mask_600 = (predict_mask == 3).float()
        gt_mask_600 = (true_mask == 3).float()
        tot_200 += dice_coeff(predict_mask_200, gt_mask_200, cuda_device).item()
        tot_500 += dice_coeff(predict_mask_500, gt_mask_500, cuda_device).item()
        tot_600 += dice_coeff(predict_mask_600, gt_mask_600, cuda_device).item()
        '''
        mask_pred_output[i,:,:] = predict_mask
        # tot += dice_coeff(predict_mask, true_mask,cuda_device).item()

    return  mask_pred_output #tot_200 / (i + 1), tot_500 / (i + 1), tot_600 / (i + 1),

def eval_multiseg(net, dataset, cuda_device=3):
    net.eval()
    mask_pred_output = np.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2]), dtype=np.float32)
    for i in range(dataset.shape[0]):    #无需用batch()
        img = dataset[i,...]          #shape:1*80*80
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)       #shape:1*1*80*80
        img = img.cuda(cuda_device)
        mask_pred = net(img)  # [0]
        predict_mask = torch.from_numpy(np.zeros_like(img.cpu()).squeeze())

        mask1 = (mask_pred[:, 1, :, :] > mask_pred[:, 0, :, :]).int() + (
                    mask_pred[:, 1, :, :] > mask_pred[:, 2, :, :]).int() + (
                            mask_pred[:, 1, :, :] > mask_pred[:, 3, :, :]).int()
        mask2 = (mask_pred[:, 2, :, :] > mask_pred[:, 0, :, :]).int() + (
                    mask_pred[:, 2, :, :] > mask_pred[:, 1, :, :]).int() + (
                            mask_pred[:, 2, :, :] > mask_pred[:, 3, :, :]).int()
        mask3 = (mask_pred[:, 3, :, :] > mask_pred[:, 0, :, :]).int() + (
                    mask_pred[:, 3, :, :] > mask_pred[:, 1, :, :]).int() + (
                            mask_pred[:, 3, :, :] > mask_pred[:, 2, :, :]).int()
        mask1 = (mask1 == 3).squeeze()
        mask2 = (mask2 == 3).squeeze()
        mask3 = (mask3 == 3).squeeze()
        #print(mask1.shape)
        #print(predict_mask.shape)
        predict_mask[mask1] = 200
        predict_mask[mask2] = 500
        predict_mask[mask3] = 600
        mask_pred_output[i, :, :] = predict_mask

    return mask_pred_output