
def main(mymodel, args, config):

    epoch = args.epoch
    learning_rate = args.learning_rate
    bsz = args.batch_size

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = RealESRGANDataset(config, bsz)
    degrader = RealESRGANDegrader(config, device)
    dataloader = DataLoader(dataset, batch_size=bsz)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    milestones=[200,400,600,800]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    model_dir = "./%s" % (args.model_dir,)
    log_path = "./%s/%s.txt" % (args.log_dir,args.log_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)


    fft_loss = FFTLoss().to(device)

    print("start training...")
    for epoch_i in range(1, epoch + 1):
        start_time = time.time()
        loss_avg = 0.0
        loss_L1 = 0.0
        loss_WAV = 0.0
        iter_num = 0
        for batch in tqdm(dataloader):
            with torch.cuda.amp.autocast(enabled=False):
                LR, HR = degrader.degrade(batch)
                LR, HR = LR * 2 - 1, HR * 2 - 1
                LR = LR.to(device)
                HR = HR.to(device)
                optimizer.zero_grad()
                SR = mymodel(LR)
                # 计算损失
                loss_l1 = l1_loss(HR, SR)
                # loss_lpips = lpips_loss(HR, SR,device=device)
                loss_wav = wavelet_loss(HR, SR)   # 权重0.5
                # loss_fft = fft_loss(HR, SR)     # 权重0.05
                loss =  loss_l1 + 0.5*loss_wav # + 1e-6 * tv_loss(SR)# + 1e-6 * tv_loss(output) + 0.5*loss_wav

                loss_avg+=loss
                # loss_L1 += loss_l1
                # loss_WAV+=loss_wav

                loss.backward()
                optimizer.step()
            iter_num+=1
        scheduler.step()
        loss_avg /= iter_num
        loss_L1 /= iter_num
        loss_WAV /= iter_num
        log_data = "[%d/%d] Average loss: %f, loss_L1: %f, loss_WAV: %f, time cost: %.2fs, cur lr is %f." % (epoch_i, epoch, loss_avg, loss_L1, loss_WAV, time.time() - start_time, scheduler.get_last_lr()[0])

        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")
        if epoch_i % args.save_interval == 0:
            torch.save(mymodel.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))


import torch, os, glob, random, copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
import numpy as np
from argparse import ArgumentParser
import time
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset import RealESRGANDataset, RealESRGANDegrader
from torchvision import transforms
from loss_utils import l1_loss,lpips_loss,wavelet_loss, tv_loss
from model.MSLRSR.losses.losses import FFTLoss # 权重0.05

if __name__ == '__main__':
    # 定义模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 首先设置GPU设备
    cudnn.benchmark = True  # 对卷积进行加速

    model_name = "LDFF_Net"                        # *************** 修改 *************** #
    from model.LDFF_Net import MYMODEL            # *************** 修改 *************** #
    mymodel = MYMODEL(up_scale=4)
    mymodel = mymodel.to(device)

    config = OmegaConf.load("config.yml")
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--milestones", type=int, nargs='+', default=[200,400,600,800])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--model_dir", type=str, default=f"weight/{model_name}")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--log_name", type=str, default=f"{model_name}")
    parser.add_argument("--save_interval", type=int, default=200)
    args = parser.parse_args()
    print(f'开始处理:{model_name}')
    main(mymodel, args, config)
