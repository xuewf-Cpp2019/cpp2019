import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D



def plot_loss_dsc_curve(loss, dice, hd, dice_train, path):
    f = open(path+'best_result.txt',"w+")

    epochs = loss.shape[0]
    epoch = np.arange(epochs) + 1
    plt.figure(figsize=[20,16],dpi=450)
    plt.subplot(221)
    plt.plot(epoch, loss, color='green', label='Training CE Loss')
    #plt.plot(epoch, loss[1], color='red', label='Training Dice Loss')
    #plt.plot(epoch, loss[2], color='purple', label='Training Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()  # legend 是在图区显示label，即上面 .plot()方法中label参数的值

    plt.subplot(222)
    plt.plot(epoch, dice_train[1], color='red', label='ED_epi_dsc')
    plt.plot(epoch, dice_train[4], color='yellow', label='ES_epi_dsc')
    plt.plot(epoch, dice_train[0], color='green', label='ED_endo_dsc')
    plt.plot(epoch, dice_train[3], color='black', label='ES_endo_dsc')
    plt.plot(epoch, dice_train[5], color='blue', label='ES_la_dsc')
    plt.plot(epoch, dice_train[2], color='orange', label='ED_la_dsc')
    mean_dice_train = np.mean(np.array(dice_train), axis=0)
    plt.plot(epoch, mean_dice_train, color='purple', label='mean_all_dsc')
    plt.xlabel('Epoch')
    plt.ylabel('Training DSC')
    plt.legend()  # legend 是在图区显示label，即上面 .plot()方法中label参数的值

    plt.subplot(223)
    plt.plot(epoch, hd[1], color='red', label='ED_epi_hd')
    plt.plot(epoch, hd[4], color='yellow', label='ES_epi_hd')
    plt.plot(epoch, hd[0], color='green', label='ED_endo_hd')
    plt.plot(epoch, hd[3], color='black', label='ES_endo_hd')
    plt.plot(epoch, hd[5], color='blue', label='ES_la_hd')
    plt.plot(epoch, hd[2], color='orange', label='ED_la_hd')
    mean_hd = np.mean(np.array(hd), axis=0)
    plt.plot(epoch, mean_hd, color='purple', label='mean_all_hd')
    plt.xlabel('Epoch')
    plt.ylabel('Validation hd')
    plt.legend()  # legend 是在图区显示label，即上面 .plot()方法中label参数的值

    plt.subplot(224)
    plt.plot(epoch, dice[1], color='red', label='ED_epi_dsc')
    plt.plot(epoch, dice[4], color='yellow', label='ES_epi_dsc')
    plt.plot(epoch, dice[0], color='green', label='ED_endo_dsc')
    plt.plot(epoch, dice[3], color='black', label='ES_endo_dsc')
    plt.plot(epoch, dice[5], color='blue', label='ES_la_dsc')
    plt.plot(epoch, dice[2], color='orange', label='ED_la_dsc')
    mean_dice = np.mean(np.array(dice), axis=0)
    plt.plot(epoch ,mean_dice, color='purple', label='mean_all_dsc')
    plt.xlabel('Epoch')
    plt.ylabel('Validation DSC')
    plt.legend()  # legend 是在图区显示label，即上面 .plot()方法中label参数的值

    #plt.subplots_adjust(wspace = 0.4)
    plt.savefig(path+'Epoch_Trainloss_Valdice.jpg')
    best_epoch = np.argmax(mean_dice)
    print('Best Checkpoint : {}'.format(best_epoch + 1))
    print('Best Checkpoint : {}'.format(best_epoch + 1),file=f)
    print('ED_endo_dsc: {0:.4f} , ED_epi_dsc: {1:.4f} , ED_la_dsc: {2:.4f} , ES_endo_dsc: {3:.4f} , ES_epi_dsc: {4:.4f} , ES_la_dsc: {5:.4f}'.
    format(dice[0][best_epoch],dice[1][best_epoch],dice[2][best_epoch],dice[3][best_epoch],dice[4][best_epoch],dice[5][best_epoch]),file=f)
    print('ED_endo_hd: {0:.4f} , ED_epi_hd: {1:.4f} , ED_la_hd: {2:.4f} , ES_endo_hd: {3:.4f} , ES_epi_hd: {4:.4f} , ES_la_hd: {5:.4f}'.
        format(hd[0][best_epoch],hd[1][best_epoch],hd[2][best_epoch],hd[3][best_epoch],hd[4][best_epoch],hd[5][best_epoch]), file=f)
    f.close()

def plot_3d_hdim_surfaces(seq, hdim, z, path):

    plt.figure(figsize=[10,6], dpi=300)
    ax = plt.gca()
    ax.plot_surface(seq, hdim, z, c= 'b', marker='^')
    ax.set_xlabel('Sequences')
    ax.set_ylabel('High Representation')
    ax.set_zlabel('Z')
    ax.savefig(path)

def plot_mask_gt_line(image, mask, gt, path):
    # gt curve
    plt.figure(figsize=[12, 9],dpi=450)
    image_Gray = Image.fromarray(image)
    #print(image.mode)
    image_RGB = image_Gray.convert('RGB')
    plt.imshow(image_RGB,aspect='auto')
    #print(mask.shape)
    # predict
    mask1 = (mask == 1).astype(np.int16)
    mask2 = (mask == 2).astype(np.int16) + (mask == 1).astype(np.int16)
    mask3 = (mask == 3).astype(np.int16)
    contours1 = measure.find_contours(mask1, 0.5)
    contours2 = measure.find_contours(mask2, 0.5)
    contours3 = measure.find_contours(mask3, 0.5)
    for n, contour in enumerate(contours2):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2.5, color='yellow')
    for n, contour in enumerate(contours1):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2.5, color='cyan')
    for n, contour in enumerate(contours3):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2.5, color='green')
    # gt
    gt1 = (gt == 1).astype(np.int16)
    gt2 = (gt == 2).astype(np.int16) + (gt == 1).astype(np.int16)
    gt3 = (gt == 3).astype(np.int16)
    contours1 = measure.find_contours(gt1, 0.5)
    contours2 = measure.find_contours(gt2, 0.5)
    contours3 = measure.find_contours(gt3, 0.5)
    for n, contour in enumerate(contours2):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2.5, color='blue', linestyle='--')
    for n, contour in enumerate(contours1):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2.5, color='orange', linestyle='--')
    for n, contour in enumerate(contours3):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2.5, color='purple', linestyle='--')
    plt.savefig(path)
    plt.close()
    '''
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    c = cv2.findContours((gt==1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = c[0]
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    c = cv2.findContours((gt==1).astype(np.uint8)+(gt==2).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = c[0]
    image = cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    c = cv2.findContours((gt==3).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = c[0]
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    # pred curve
    c = cv2.findContours((mask == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = c[0]
    image = cv2.drawContours(image, contours, -1, (0, 100, 0), 2)
    c = cv2.findContours((mask==1).astype(np.uint8)+(mask==2).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = c[0]
    image = cv2.drawContours(image, contours, -1, (255, 192, 203), 2)
    c = cv2.findContours((mask == 3).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = c[0]
    image = cv2.drawContours(image, contours, -1, (0, 191, 255), 2)
    cv2.imwrite(path, image)
    '''