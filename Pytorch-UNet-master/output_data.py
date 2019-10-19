import numpy as np
import pandas as pd

def output_dice_hd_pat(train_metric,val_metric,path):
    # train data
    writer = pd.ExcelWriter(path + 'train.xlsx')
    data = pd.DataFrame(np.transpose(train_metric[0]))
    data.to_excel(writer, 'ED_endo_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[1]))
    data.to_excel(writer, 'ED_epi_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[2]))
    data.to_excel(writer, 'ED_la_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[3]))
    data.to_excel(writer, 'ES_endo_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[4]))
    data.to_excel(writer, 'ES_epi_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[5]))
    data.to_excel(writer, 'ES_la_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[6]))
    data.to_excel(writer, 'ED_endo_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[7]))
    data.to_excel(writer, 'ED_epi_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[8]))
    data.to_excel(writer, 'ED_la_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[9]))
    data.to_excel(writer, 'ES_endo_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[10]))
    data.to_excel(writer, 'ES_epi_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(train_metric[11]))
    data.to_excel(writer, 'ES_la_hd', float_format='%.4f', header=False, index=False)
    writer.save()
    writer.close()
    # val data
    writer = pd.ExcelWriter(path + 'val.xlsx')
    data = pd.DataFrame(np.transpose(val_metric[0]))
    data.to_excel(writer, 'ED_endo_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[1]))
    data.to_excel(writer, 'ED_epi_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[2]))
    data.to_excel(writer, 'ED_la_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[3]))
    data.to_excel(writer, 'ES_endo_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[4]))
    data.to_excel(writer, 'ES_epi_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[5]))
    data.to_excel(writer, 'ES_la_dice', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[6]))
    data.to_excel(writer, 'ED_endo_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[7]))
    data.to_excel(writer, 'ED_epi_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[8]))
    data.to_excel(writer, 'ED_la_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[9]))
    data.to_excel(writer, 'ES_endo_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[10]))
    data.to_excel(writer, 'ES_epi_hd', float_format='%.4f', header=False, index=False)
    data = pd.DataFrame(np.transpose(val_metric[11]))
    data.to_excel(writer, 'ES_la_hd', float_format='%.4f', header=False, index=False)
    writer.save()
    writer.close()

def output_dice_hd_val(val_metric,path):
    writer = pd.ExcelWriter(path + 'test_output_metric12.xlsx')
    data = pd.DataFrame(val_metric)
    data.to_excel(writer, 'test_50_patients', float_format='%.4f', header=['ED_endo_dice','ED_epi_dice','ED_la_dice',
                                              'ES_endo_dice','ES_epi_dice','ES_la_dice','ED_endo_hd','ED_epi_hd',
                                              'ED_la_hd','ES_endo_hd','ES_epi_hd','ES_la_hd'], index=False)
    writer.save()
    writer.close()
def output_hdim_val_before_rnn(hdim, path):
    writer = pd.ExcelWriter(path + 'test_hdim_before_rnn.xlsx')
    data = pd.DataFrame(hdim)
    data.to_excel(writer, 'test_hdim_before_rnn', float_format='%.4f', header=False, index=False)
    writer.save()
    writer.close()

def output_hdim_val_after_rnn(hdim, path):
    writer = pd.ExcelWriter(path + 'test_hdim_after_rnn.xlsx')
    data = pd.DataFrame(hdim)
    data.to_excel(writer, 'test_hdim_after_rnn', float_format='%.4f', header=False, index=False)
    writer.save()
    writer.close()