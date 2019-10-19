import cv2
import SimpleITK as sitk
import numpy as np
save_dir = './data/whr/CAMUS_project/data/images/train'
img_dir = './CAMUS_project/data/training/training/patient'

pat_num = 450
# save raw data to .png image
for i in range(pat_num):
    if i < 9:
        pat = '000' + str(i + 1)
    elif i < 99:
        pat = '00' + str(i + 1)
    else:
        pat = '0' + str(i + 1)
    # 2CH_ED
    CH2_ED_dir = img_dir + pat + '/patient' + pat + '_2CH_ED.mhd'
    CH2_ED_gt_dir = img_dir + pat + '/patient' + pat + '_2CH_ED_gt.mhd'
    CH2_ED_image = sitk.ReadImage(CH2_ED_dir)
    CH2_ED_gt = sitk.ReadImage(CH2_ED_gt_dir)
    image = sitk.GetArrayFromImage(CH2_ED_image)
    gt = sitk.GetArrayFromImage(CH2_ED_gt)
    image = np.squeeze(image)
    gt = np.squeeze(gt)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt = cv2.resize(gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    gt_1 = (gt == 1)#.astype(np.uint8)
    gt_2 = (gt == 2)#.astype(np.uint8)
    gt_3 = (gt == 3)#.astype(np.uint8)
    gt[gt_1] = 80
    gt[gt_2] = 160
    gt[gt_3] = 255
    cv2.imwrite('./CAMUS_project/data/images/train/' + pat + '_2CH_ED' + '.png', image)
    cv2.imwrite('./CAMUS_project/data/images/gt_train/' + pat + '_2CH_ED' + '.png', gt)
    # 2CH_ES
    CH2_ES_dir = img_dir + pat + '/patient' + pat + '_2CH_ES.mhd'
    CH2_ES_gt_dir = img_dir + pat + '/patient' + pat + '_2CH_ES_gt.mhd'
    CH2_ES_image = sitk.ReadImage(CH2_ES_dir)
    CH2_ES_gt = sitk.ReadImage(CH2_ES_gt_dir)
    image = sitk.GetArrayFromImage(CH2_ES_image)
    gt = sitk.GetArrayFromImage(CH2_ES_gt)
    image = np.squeeze(image)
    gt = np.squeeze(gt)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt = cv2.resize(gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    gt_1 = (gt == 1)  # .astype(np.uint8)
    gt_2 = (gt == 2)  # .astype(np.uint8)
    gt_3 = (gt == 3)  # .astype(np.uint8)
    gt[gt_1] = 80
    gt[gt_2] = 160
    gt[gt_3] = 255
    cv2.imwrite('./CAMUS_project/data/images/train/' + pat + '_2CH_ES' + '.png', image)
    cv2.imwrite('./CAMUS_project/data/images/gt_train/' + pat + '_2CH_ES' + '.png', gt)
    # 4CH_ED
    CH4_ED_dir = img_dir + pat + '/patient' + pat + '_4CH_ED.mhd'
    CH4_ED_gt_dir = img_dir + pat + '/patient' + pat + '_4CH_ED_gt.mhd'
    CH4_ED_image = sitk.ReadImage(CH4_ED_dir)
    CH4_ED_gt = sitk.ReadImage(CH4_ED_gt_dir)
    image = sitk.GetArrayFromImage(CH4_ED_image)
    gt = sitk.GetArrayFromImage(CH4_ED_gt)
    image = np.squeeze(image)
    gt = np.squeeze(gt)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt = cv2.resize(gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    gt_1 = (gt == 1)  # .astype(np.uint8)
    gt_2 = (gt == 2)  # .astype(np.uint8)
    gt_3 = (gt == 3)  # .astype(np.uint8)
    gt[gt_1] = 80
    gt[gt_2] = 160
    gt[gt_3] = 255
    cv2.imwrite('./CAMUS_project/data/images/train/' + pat + '_4CH_ED' + '.png', image)
    cv2.imwrite('./CAMUS_project/data/images/gt_train/' + pat + '_4CH_ED' + '.png', gt)
    # 4CH_ES
    CH4_ES_dir = img_dir + pat + '/patient' + pat + '_4CH_ES.mhd'
    CH4_ES_gt_dir = img_dir + pat + '/patient' + pat + '_4CH_ES_gt.mhd'
    CH4_ES_image = sitk.ReadImage(CH4_ES_dir)
    CH4_ES_gt = sitk.ReadImage(CH4_ES_gt_dir)
    image = sitk.GetArrayFromImage(CH4_ES_image)
    gt = sitk.GetArrayFromImage(CH4_ES_gt)
    image = np.squeeze(image)
    gt = np.squeeze(gt)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt = cv2.resize(gt, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    gt_1 = (gt == 1)  # .astype(np.uint8)
    gt_2 = (gt == 2)  # .astype(np.uint8)
    gt_3 = (gt == 3)  # .astype(np.uint8)
    gt[gt_1] = 80
    gt[gt_2] = 160
    gt[gt_3] = 255
    cv2.imwrite('./CAMUS_project/data/images/train/' + pat + '_4CH_ES' + '.png', image)
    cv2.imwrite('./CAMUS_project/data/images/gt_train/' + pat + '_4CH_ES' + '.png', gt)



    # contour gt on images
    '''
    h_gt1 = cv2.findContours(gt_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = h_gt1[0]
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    h_gt2 = cv2.findContours(gt_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = h_gt2[0]
    image = cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    h_gt3 = cv2.findContours(gt_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = h_gt3[0]
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    '''

