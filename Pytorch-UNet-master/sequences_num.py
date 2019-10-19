import cv2
import SimpleITK as sitk
import numpy as np

img_dir = './data/training/training/patient'

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
    CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_sequence.mhd'
    CH2_image = sitk.ReadImage(CH2_dir)
    s2 = sitk.GetArrayFromImage(CH2_image)

    CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_sequence.mhd'
    CH4_image = sitk.ReadImage(CH4_dir)
    s4 = sitk.GetArrayFromImage(CH4_image)
    print(s2.shape[0],s4.shape[0])

