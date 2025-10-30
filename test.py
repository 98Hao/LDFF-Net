# ************************************************************************************ #
import torch, os, glob, copy
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from tqdm import tqdm
from torch.backends import cudnn



# 定义模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 首先设置GPU设备
cudnn.benchmark = True  # 对卷积进行加速
model_name = "LDFF_Net"                              # *************** 修改 *************** #
from model.LDFF_Net import MYMODEL                  # *************** 修改 *************** #
mymodel = MYMODEL(up_scale=4)
mymodel = mymodel.to(device)


LR_path = f'/media/ubuntu/58bd05a9-8d46-4354-9917-bf7d02de6f68/超分_红外2_小目标/2-原始的真实退化/yolov5/data/sirst/images/val_original'
SR_path = f'/media/ubuntu/58bd05a9-8d46-4354-9917-bf7d02de6f68/超分_红外2_小目标/2-原始的真实退化/yolov5/data/sirst/images/val_imdn'
device = torch.device("cuda")
parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--model_dir", type=str, default=f"weight/{model_name}")
parser.add_argument("--LR_dir", type=str, default=LR_path)
parser.add_argument("--SR_dir", type=str, default=SR_path)
args = parser.parse_args()


mymodel.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (args.model_dir, args.epoch), weights_only=False))
mymodel = mymodel.to(device)
mymodel.eval()

test_LR_paths = list(sorted(glob.glob(os.path.join(args.LR_dir, "*.png"))))
test_HR_paths = list(sorted(glob.glob(os.path.join(args.HR_dir, "*.png"))))

os.makedirs(args.SR_dir, exist_ok=True)

total_time = 0
# 预热GPU(避免初始化的时间影响测量)
with torch.no_grad():
    dummy_input = torch.rand(1, 3, 224, 224).cuda()
    for _ in range(10):
        _ = mymodel(dummy_input)

# 创建事件
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    mypad=2
    for i, path in enumerate(tqdm(test_LR_paths)):
        # 加载图像并转换为RGB
        LR_img = Image.open(path).convert("RGB")
        H, W = LR_img.size  # 获取原始图像尺寸

        # 计算需要填充的像素数，使尺寸成为4的倍数
        pad_h = (mypad - H % mypad) % mypad  # 高度方向填充
        pad_w = (mypad - W % mypad) % mypad  # 宽度方向填充

        # 如果需要填充，则进行填充操作
        if pad_h > 0 or pad_w > 0:
            # 转换为numpy数组进行填充（右下角填充）
            lr_np = np.array(LR_img)
            # 使用边缘填充，只在右下角填充
            lr_padded = np.pad(lr_np, ((0, pad_w), (0, pad_h), (0, 0)), mode='edge')
            LR_img = Image.fromarray(lr_padded)

        # 转换为张量并进行预处理
        LR = transforms.ToTensor()(LR_img).to(device).unsqueeze(0) * 2 - 1

        # 同步并记录时间
        torch.cuda.synchronize()
        start_event.record()

        # 进行超分辨率处理
        SR = mymodel(LR)

        end_event.record()
        torch.cuda.synchronize()

        # 累加时间
        total_time += start_event.elapsed_time(end_event)


        # 标准化处理
        SR = (SR - SR.mean(dim=[2, 3], keepdim=True)) / SR.std(dim=[2, 3], keepdim=True) \
             * LR.std(dim=[2, 3], keepdim=True) + LR.mean(dim=[2, 3], keepdim=True)

        # 转换回PIL图像
        SR_img = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())

        # 如果之前进行了填充，现在裁剪掉多余部分
        if pad_h > 0 or pad_w > 0:
            # 计算目标尺寸（原始尺寸的4倍）
            target_width = W * mypad
            target_height = H * mypad
            # 裁剪图像，只保留左上角的目标区域
            SR_img = SR_img.crop((0, 0, target_height, target_width))

        # 保存处理后的图像
        SR_img.save(os.path.join(args.SR_dir, os.path.basename(path)))

    print(f"处理 {len(test_LR_paths)} 张图像的平均时间: {total_time/len(test_LR_paths):.2f} 毫秒")