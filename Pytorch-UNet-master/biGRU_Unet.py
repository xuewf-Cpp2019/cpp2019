import torch
import numpy as np
def bi_unet(net, imgs, cuda_device, batch_size = 1, high_dim=64):

    high_representation = torch.zeros(batch_size, 10, high_dim).cuda(cuda_device)
    f1 = torch.zeros(batch_size, 2, 32, imgs.shape[2], imgs.shape[3]).cuda(cuda_device)
    f2 = torch.zeros(batch_size, 2, 64, imgs.shape[2] // 2, imgs.shape[3] // 2).cuda(cuda_device)
    f3 = torch.zeros(batch_size, 2, 128, imgs.shape[2] // 4, imgs.shape[3] // 4).cuda(cuda_device)
    f4 = torch.zeros(batch_size, 2, 256, imgs.shape[2] // 8, imgs.shape[3] // 8).cuda(cuda_device)
    f5 = torch.zeros(batch_size, 2, 256, imgs.shape[2] // 16, imgs.shape[3] // 16).cuda(cuda_device)

    for frame_i in range(10):
        if frame_i == 0:
            high_representation[:, frame_i, ], f1[:, 0, ], f2[:, 0, ], f3[:, 0, ], f4[:, 0, ], f5[:, 0, ] = net[0](
                imgs[:, frame_i, ].unsqueeze(dim=1))
        elif frame_i == 9:
            high_representation[:, frame_i, ], f1[:, 1, ], f2[:, 1, ], f3[:, 1, ], f4[:, 1, ], f5[:, 1, ] = net[0](
                imgs[:, frame_i, ].unsqueeze(dim=1))
        else:
            high_representation[:, frame_i, ], _, _, _, _, _ = net[0](imgs[:, frame_i, ].unsqueeze(dim=1))

    high_representation_biGRU = net[1](x=high_representation,cuda_device=cuda_device)
    for frame_EDorES in range(2):
        if frame_EDorES == 0:
            ED_seg = net[2](high_representation_biGRU[:, 0, ], f1[:, 0, ], f2[:, 0, ], f3[:, 0, ], f4[:, 0, ],
                            f5[:, 0, ])
        if frame_EDorES == 1:
            ES_seg = net[2](high_representation_biGRU[:, 9, ], f1[:, 1, ], f2[:, 1, ], f3[:, 1, ], f4[:, 1, ],
                            f5[:, 1, ])
    return ED_seg, ES_seg

def bi_unet_val(net, imgs, cuda_device, batch_size = 1, high_dim=64):

    high_representation = torch.zeros(batch_size, 10, high_dim).cuda(cuda_device)
    f1 = torch.zeros(batch_size, 2, 32, imgs.shape[2], imgs.shape[3]).cuda(cuda_device)
    f2 = torch.zeros(batch_size, 2, 64, imgs.shape[2] // 2, imgs.shape[3] // 2).cuda(cuda_device)
    f3 = torch.zeros(batch_size, 2, 128, imgs.shape[2] // 4, imgs.shape[3] // 4).cuda(cuda_device)
    f4 = torch.zeros(batch_size, 2, 256, imgs.shape[2] // 8, imgs.shape[3] // 8).cuda(cuda_device)
    f5 = torch.zeros(batch_size, 2, 256, imgs.shape[2] // 16, imgs.shape[3] // 16).cuda(cuda_device)

    for frame_i in range(10):
        if frame_i == 0:
            high_representation[:, frame_i, ], f1[:, 0, ], f2[:, 0, ], f3[:, 0, ], f4[:, 0, ], f5[:, 0, ] = net[0](
                imgs[:, frame_i, ].unsqueeze(dim=1))
        elif frame_i == 9:
            high_representation[:, frame_i, ], f1[:, 1, ], f2[:, 1, ], f3[:, 1, ], f4[:, 1, ], f5[:, 1, ] = net[0](
                imgs[:, frame_i, ].unsqueeze(dim=1))
        else:
            high_representation[:, frame_i, ], _, _, _, _, _ = net[0](imgs[:, frame_i, ].unsqueeze(dim=1))
    print(high_representation[0,])
    #high_representation = torch.zeros(batch_size, 10, high_dim).cuda(cuda_device)
    high_representation_biGRU = net[1](x=high_representation,cuda_device=cuda_device)
    print(high_representation[0,])
    for frame_EDorES in range(2):
        if frame_EDorES == 0:
            ED_seg = net[2](high_representation_biGRU[:, 0, ], f1[:, 0, ], f2[:, 0, ], f3[:, 0, ], f4[:, 0, ],
                            f5[:, 0, ])
        if frame_EDorES == 1:
            ES_seg = net[2](high_representation_biGRU[:, 9, ], f1[:, 1, ], f2[:, 1, ], f3[:, 1, ], f4[:, 1, ],
                            f5[:, 1, ])
    return ED_seg, ES_seg, np.array(high_representation[0].cpu().detach()), np.array(high_representation_biGRU[0].cpu().detach())