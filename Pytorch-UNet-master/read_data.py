import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
#from PIL import Image

def import_data_for_val():
    ch2_val, ch4_val = read_select_resize_sequences_val()
    ch2_val, ch4_val = normalize_val(ch2_val, ch4_val)
    return ch2_val, ch4_val

def import_2Ddata_for_hdim_val():
    ch2_val, ch4_val = read_resize_2D_data_val()
    ch2_val, ch4_val = normalize_val(ch2_val, ch4_val)
    return ch2_val, ch4_val

def import_2Ddata_for_hdim():
    ch2_train, ch2_val, ch4_train, ch4_val = read_resize_2D_data()
    ch2_train, ch2_val, ch4_train, ch4_val = normalize(ch2_train, ch2_val, ch4_train, ch4_val)
    train = np.concatenate((ch2_train, ch4_train), axis=0)
    val = np.concatenate((ch2_val, ch4_val), axis=0)
    ch2_train_gt, ch2_val_gt, ch4_train_gt, ch4_val_gt = read_resize_EDES_GT()
    train_gt = np.concatenate((ch2_train_gt, ch4_train_gt), axis=0)
    val_gt = np.concatenate((ch2_val_gt, ch4_val_gt), axis=0)

    state_train = np.random.get_state()
    np.random.shuffle(train)
    np.random.set_state(state_train)
    np.random.shuffle(train_gt)

    state_val = np.random.get_state()
    np.random.shuffle(val)
    np.random.set_state(state_val)
    np.random.shuffle(val_gt)
    return train, train_gt, val, val_gt


def import_process_data():
    ch2_train, ch2_val, ch4_train, ch4_val = read_select_resize_sequences()
    ch2_train, ch2_val, ch4_train, ch4_val = normalize(ch2_train, ch2_val, ch4_train, ch4_val)
    train = np.concatenate((ch2_train, ch4_train), axis=0)
    val = np.concatenate((ch2_val, ch4_val), axis=0)
    ch2_train_gt, ch2_val_gt, ch4_train_gt, ch4_val_gt = read_resize_EDES_GT()
    train_gt = np.concatenate((ch2_train_gt, ch4_train_gt), axis=0)
    val_gt = np.concatenate((ch2_val_gt, ch4_val_gt), axis=0)

    state_train = np.random.get_state()
    np.random.shuffle(train)
    np.random.set_state(state_train)
    np.random.shuffle(train_gt)

    state_val = np.random.get_state()
    np.random.shuffle(val)
    np.random.set_state(state_val)
    np.random.shuffle(val_gt)
    return train, train_gt, val, val_gt

def read_resize_2D_data():
    img_dir = './data/training/training/patient'
    pat_num = 450
    ch2 = np.zeros((pat_num, 2, 256, 256), dtype=np.float32)
    ch4 = np.zeros((pat_num, 2, 256, 256), dtype=np.float32)
    for i in range(pat_num):
        if i < 9:
            pat = '000' + str(i + 1)
        elif i < 99:
            pat = '00' + str(i + 1)
        else:
            pat = '0' + str(i + 1)
        # Chamber 2 ED
        CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_ED.mhd'
        CH2_image = sitk.ReadImage(CH2_dir)
        s2 = sitk.GetArrayFromImage(CH2_image)
        ch2[i,0] = resize_ori_image(s2, [256, 256])
        # Chamber 2 ES
        CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_ES.mhd'
        CH2_image = sitk.ReadImage(CH2_dir)
        s2 = sitk.GetArrayFromImage(CH2_image)
        ch2[i, 1] = resize_ori_image(s2, [256, 256])
        # Chamber 4 ED
        CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_ED.mhd'
        CH4_image = sitk.ReadImage(CH4_dir)
        s4 = sitk.GetArrayFromImage(CH4_image)
        ch4[i, 0] = resize_ori_image(s4, [256, 256])
        # Chamber 4 ES
        CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_ED.mhd'
        CH4_image = sitk.ReadImage(CH4_dir)
        s4 = sitk.GetArrayFromImage(CH4_image)
        ch4[i, 1] = resize_ori_image(s4, [256, 256])
    return ch2[0:400, ], ch2[400:450, ], ch4[0:400, ], ch4[400:450, ]

def read_resize_2D_data_val():
    img_dir = './data/training/training/patient'
    pat_num = 50
    ch2 = np.zeros((pat_num, 2, 256, 256), dtype=np.float32)
    ch4 = np.zeros((pat_num, 2, 256, 256), dtype=np.float32)
    for i in range(pat_num):
        pat = '0' + str(i + 401)
        # Chamber 2 ED
        CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_ED.mhd'
        CH2_image = sitk.ReadImage(CH2_dir)
        s2 = sitk.GetArrayFromImage(CH2_image)
        ch2[i,0] = resize_ori_image(s2, [256, 256])
        # Chamber 2 ES
        CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_ES.mhd'
        CH2_image = sitk.ReadImage(CH2_dir)
        s2 = sitk.GetArrayFromImage(CH2_image)
        ch2[i, 1] = resize_ori_image(s2, [256, 256])
        # Chamber 4 ED
        CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_ED.mhd'
        CH4_image = sitk.ReadImage(CH4_dir)
        s4 = sitk.GetArrayFromImage(CH4_image)
        ch4[i, 0] = resize_ori_image(s4, [256, 256])
        # Chamber 4 ES
        CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_ED.mhd'
        CH4_image = sitk.ReadImage(CH4_dir)
        s4 = sitk.GetArrayFromImage(CH4_image)
        ch4[i, 1] = resize_ori_image(s4, [256, 256])
    return ch2, ch4

def read_select_resize_sequences():
    img_dir = './data/training/training/patient'

    pat_num = 450
    ch2 = np.zeros((pat_num,10,256,256),dtype=np.float32)
    ch4 = np.zeros((pat_num,10,256,256),dtype=np.float32)
    # select 10 sequences and resize image to 256 * 256
    for i in range(pat_num):
        if i < 9:
            pat = '000' + str(i + 1)
        elif i < 99:
            pat = '00' + str(i + 1)
        else:
            pat = '0' + str(i + 1)
        # Chamber 2
        CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_sequence.mhd'
        CH2_image = sitk.ReadImage(CH2_dir)
        s2 = sitk.GetArrayFromImage(CH2_image)
        s2_ori_cnum = s2.shape[0]
        s2_gap = (s2_ori_cnum - 4) / 6
        select_s2_id = np.zeros(s2_ori_cnum)
        select_s2_id[0:2] = 1
        select_s2_id[s2_ori_cnum-2:] = 1

        if s2_ori_cnum > 10:
            for j in range(6):
                s2_i = np.int(np.round(s2_gap * (j+1)) + 1)
                select_s2_id[s2_i] = 1
        else: select_s2_id[:] = 1

        ch2_s = s2[select_s2_id>0,]
        ch2[i,] = resize_ori_image(ch2_s, [256, 256])
        # Chamber 4
        CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_sequence.mhd'
        CH4_image = sitk.ReadImage(CH4_dir)
        s4 = sitk.GetArrayFromImage(CH4_image)
        s4_ori_cnum = s4.shape[0]
        s4_gap = (s4_ori_cnum - 4) / 6
        select_s4_id = np.zeros(s4_ori_cnum)
        select_s4_id[0:2] = 1
        select_s4_id[s4_ori_cnum - 2:] = 1
        if s4_ori_cnum > 10:
            for j in range(6):
                s4_i = np.int(np.round(s4_gap * (j + 1)) + 1)
                select_s4_id[s4_i] = 1
        else: select_s4_id[:] = 1

        ch4_s = s4[select_s4_id > 0,]
        ch4[i,] = resize_ori_image(ch4_s, [256, 256])
    return ch2[0:400, ], ch2[400:450, ], ch4[0:400, ], ch4[400:450, ]

def read_select_resize_sequences_val():
    img_dir = './data/training/training/patient'

    pat_num = 50
    ch2 = np.zeros((pat_num,10,256,256),dtype=np.float32)
    ch4 = np.zeros((pat_num,10,256,256),dtype=np.float32)
    # select 10 sequences and resize image to 256 * 256
    for i in range(pat_num):
        j = i + 400

        pat = '0' + str(j + 1)
        # Chamber 2
        CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_sequence.mhd'
        CH2_image = sitk.ReadImage(CH2_dir)
        s2 = sitk.GetArrayFromImage(CH2_image)
        s2_ori_cnum = s2.shape[0]
        s2_gap = (s2_ori_cnum - 4) / 6
        select_s2_id = np.zeros(s2_ori_cnum)
        select_s2_id[0:2] = 1
        select_s2_id[s2_ori_cnum-2:] = 1

        if s2_ori_cnum > 10:
            for j in range(6):
                s2_i = np.int(np.round(s2_gap * (j+1)) + 1)
                select_s2_id[s2_i] = 1
        else: select_s2_id[:] = 1

        ch2_s = s2[select_s2_id>0,]
        ch2[i,] = resize_ori_image(ch2_s, [256, 256])
        # Chamber 4
        CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_sequence.mhd'
        CH4_image = sitk.ReadImage(CH4_dir)
        s4 = sitk.GetArrayFromImage(CH4_image)
        s4_ori_cnum = s4.shape[0]
        s4_gap = (s4_ori_cnum - 4) / 6
        select_s4_id = np.zeros(s4_ori_cnum)
        select_s4_id[0:2] = 1
        select_s4_id[s4_ori_cnum - 2:] = 1
        if s4_ori_cnum > 10:
            for j in range(6):
                s4_i = np.int(np.round(s4_gap * (j + 1)) + 1)
                select_s4_id[s4_i] = 1
        else: select_s4_id[:] = 1

        ch4_s = s4[select_s4_id > 0,]
        ch4[i,] = resize_ori_image(ch4_s, [256, 256])
    return ch2, ch4

def read_ori_EDES(i):
    img_dir = './data/training/training/patient'

    pat = '0' + str(i + 401)
    # Chamber 2
    CH2_ED_dir = img_dir + pat + '/patient' + pat + '_2CH_ED_gt.mhd'
    CH2_ED_gt = sitk.ReadImage(CH2_ED_dir)
    CH2_ED_gt = sitk.GetArrayFromImage(CH2_ED_gt)
    #CH2_ED_gt = np.squeeze(CH2_ED_gt)

    CH2_ES_dir = img_dir + pat + '/patient' + pat + '_2CH_ES_gt.mhd'
    CH2_ES_gt = sitk.ReadImage(CH2_ES_dir)
    CH2_ES_gt = sitk.GetArrayFromImage(CH2_ES_gt)
    #CH2_ES_gt = np.squeeze(CH2_ES_gt)

    CH2_ED_dir = img_dir + pat + '/patient' + pat + '_2CH_ED.mhd'
    CH2_ED = sitk.ReadImage(CH2_ED_dir)
    CH2_ED = sitk.GetArrayFromImage(CH2_ED)
    #CH2_ED = np.squeeze(CH2_ED)

    CH2_ES_dir = img_dir + pat + '/patient' + pat + '_2CH_ES.mhd'
    CH2_ES = sitk.ReadImage(CH2_ES_dir)
    CH2_ES = sitk.GetArrayFromImage(CH2_ES)
    #CH2_ES = np.squeeze(CH2_ES)

    # Chamber 4
    CH4_ED_dir = img_dir + pat + '/patient' + pat + '_4CH_ED_gt.mhd'
    CH4_ED_gt = sitk.ReadImage(CH4_ED_dir)
    CH4_ED_gt = sitk.GetArrayFromImage(CH4_ED_gt)
    #CH4_ED_gt = np.squeeze(CH4_ED_gt)

    CH4_ES_dir = img_dir + pat + '/patient' + pat + '_4CH_ES_gt.mhd'
    CH4_ES_gt = sitk.ReadImage(CH4_ES_dir)
    CH4_ES_gt = sitk.GetArrayFromImage(CH4_ES_gt)
    #CH4_ES_gt = np.squeeze(CH4_ES_gt)

    CH4_ED_dir = img_dir + pat + '/patient' + pat + '_4CH_ED.mhd'
    CH4_ED = sitk.ReadImage(CH4_ED_dir)
    CH4_ED = sitk.GetArrayFromImage(CH4_ED)
    #CH4_ED = np.squeeze(CH4_ED)

    CH4_ES_dir = img_dir + pat + '/patient' + pat + '_4CH_ES.mhd'
    CH4_ES = sitk.ReadImage(CH4_ES_dir)
    CH4_ES = sitk.GetArrayFromImage(CH4_ES)
    #CH4_ES = np.squeeze(CH4_ES)

    return CH2_ED_gt, CH2_ES_gt, CH2_ED, CH2_ES, CH4_ED_gt, CH4_ES_gt, CH4_ED, CH4_ES

def read_resize_EDES_GT():
    img_dir = './data/training/training/patient'
    pat_num = 450
    ch2_GT = np.zeros((pat_num,2,256,256),dtype=np.long)
    ch4_GT = np.zeros((pat_num,2,256,256),dtype=np.long)

    for i in range(pat_num):
        if i < 9:
            pat = '000' + str(i + 1)
        elif i < 99:
            pat = '00' + str(i + 1)
        else:
            pat = '0' + str(i + 1)
        # Chamber 2
        CH2_ED_dir = img_dir + pat + '/patient' + pat + '_2CH_ED_gt.mhd'
        CH2_ED_image = sitk.ReadImage(CH2_ED_dir)
        s2_ED = sitk.GetArrayFromImage(CH2_ED_image)
        #s2_ED = np.squeeze(s2_ED)
        ch2_GT[i, 0,] = np.squeeze(resize_gt(s2_ED, [256,256]))

        CH2_ES_dir = img_dir + pat + '/patient' + pat + '_2CH_ES_gt.mhd'
        CH2_ES_image = sitk.ReadImage(CH2_ES_dir)
        s2_ES = sitk.GetArrayFromImage(CH2_ES_image)
        #s2_ES = np.squeeze(s2_ES)
        ch2_GT[i, 1,] = np.squeeze(resize_gt(s2_ES, [256,256]))

        # Chamber 4
        CH4_ED_dir = img_dir + pat + '/patient' + pat + '_4CH_ED_gt.mhd'
        CH4_ED_image = sitk.ReadImage(CH4_ED_dir)
        s4_ED = sitk.GetArrayFromImage(CH4_ED_image)
        #s4_ED = np.squeeze(s4_ED)
        ch4_GT[i, 0,] = np.squeeze(resize_gt(s4_ED, [256,256]))

        CH4_ES_dir = img_dir + pat + '/patient' + pat + '_4CH_ES_gt.mhd'
        CH4_ES_image = sitk.ReadImage(CH4_ES_dir)
        s4_ES = sitk.GetArrayFromImage(CH4_ES_image)
        #s4_ES = np.squeeze(s4_ES)
        ch4_GT[i, 1,] = np.squeeze(resize_gt(s4_ES, [256,256]))
    return ch2_GT[0:400, ], ch2_GT[400:450, ], ch4_GT[0:400, ], ch4_GT[400:450, ]
'''
def read_imgs():
    dir_imgs = './data/images/train/'
    dir_gts = './data/images/gt_train/'
    images = []
    ground_trues = []
    imgs = os.listdir(dir_imgs)
    imgs.sort()
    gts = os.listdir(dir_gts)
    gts.sort()
    for i in range(len(imgs)):
        image = (cv2.imread(dir_imgs + imgs[i]))[:,:,0]
        image = normalize(image)
        gt = (cv2.imread(dir_gts + gts[i]))[:,:,0]
        gt_1 = (gt == 80)  # .astype(np.uint8)
        gt_2 = (gt == 160)  # .astype(np.uint8)
        gt_3 = (gt == 255)  # .astype(np.uint8)
        gt[gt_1] = 1
        gt[gt_2] = 2
        gt[gt_3] = 3
        images.append(image.astype(np.float32))
        ground_trues.append(gt)
        img_gt = list(zip(images,ground_trues))
        train_set = img_gt[0:1600]
        val_set = img_gt[1600:1800]
    return train_set,val_set
'''

def normalize(ch2_train, ch2_val, ch4_train, ch4_val):
    for i in range(400):
        min = np.min(ch2_train[i,])
        max = np.max(ch2_train[i,])
        ch2_train[i,] = (ch2_train[i,] - min) / (max - min)
        ch2_train[i,] = (ch2_train[i,] - 0.5) / 0.5
        min = np.min(ch4_train[i,])
        max = np.max(ch4_train[i,])
        ch4_train[i,] = (ch4_train[i,] - min) / (max - min)
        ch4_train[i,] = (ch4_train[i,] - 0.5) / 0.5
    for i in range(50):
        min = np.min(ch2_val[i,])
        max = np.max(ch2_val[i,])
        ch2_val[i,] = (ch2_val[i,] - min) / (max - min)
        ch2_val[i,] = (ch2_val[i,] - 0.5) / 0.5
        min = np.min(ch4_val[i,])
        max = np.max(ch4_val[i,])
        ch4_val[i,] = (ch4_val[i,] - min) / (max - min)
        ch4_val[i,] = (ch4_val[i,] - 0.5) / 0.5
    return ch2_train, ch2_val, ch4_train, ch4_val

def normalize_val(ch2_val, ch4_val):
    for i in range(50):
        min = np.min(ch2_val[i,])
        max = np.max(ch2_val[i,])
        ch2_val[i,] = (ch2_val[i,] - min) / (max - min)
        ch2_val[i,] = (ch2_val[i,] - 0.5) / 0.5
        min = np.min(ch4_val[i,])
        max = np.max(ch4_val[i,])
        ch4_val[i,] = (ch4_val[i,] - min) / (max - min)
        ch4_val[i,] = (ch4_val[i,] - 0.5) / 0.5
    return ch2_val, ch4_val

def resize_ori_image(imgs, target_shape):
    C = imgs.shape[0]
    W,H = target_shape[0],target_shape[1]
    re_imgs = np.zeros((C, W, H),dtype=np.float32)
    for i in range(C):
        re_imgs[i, :, :] = resize(imgs[i, :, :], output_shape = (W, H), order=1, mode='constant', preserve_range=True, anti_aliasing=True)
    return re_imgs

def resize_gt(imgs, target_shape):
    C = imgs.shape[0]
    W,H = target_shape[0],target_shape[1]
    re_imgs = np.zeros((C, W, H),dtype=np.long)
    for i in range(C):
        re_imgs[i, :, :] = resize(imgs[i, :, :], output_shape = (W, H), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
    return re_imgs

'''
    # script : " save gif of the select_resize sequences "

    for i in range(ch2.shape[0]):
        frames = []
        for j in range(ch2.shape[1]):
            frames.append(Image.fromarray(ch2[i,j,]))
        imageio.mimsave('./data/sequences_select_and_resize/ch2/patient_'+ str(i+1) + '.gif', frames, 'GIF', duration=0.1)
    for i in range(ch4.shape[0]):
        frames = []
        for j in range(ch4.shape[1]):
            frames.append(Image.fromarray(ch4[i,j,]))
        imageio.mimsave('./data/sequences_select_and_resize/ch4/patient_'+ str(i+1) + '.gif', frames, 'GIF', duration=0.1)
    '''