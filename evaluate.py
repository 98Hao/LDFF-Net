# ************************************************************************************ #
import torch, os, glob, pyiqa
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

LR_path = f'/media/ubuntu/58bd05a9-8d46-4354-9917-bf7d02de6f68/超分_红外2_小目标/2-原始的真实退化/yolov5/data/sirst/images/val_ours/original'
SR_path = f'/media/ubuntu/58bd05a9-8d46-4354-9917-bf7d02de6f68/超分_红外2_小目标/2-原始的真实退化/师兄图像/无参考/锉刀/After' # Before   After

parser = ArgumentParser()
parser.add_argument("--SR_dir", type=str, default=SR_path)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化评估指标
niqe = pyiqa.create_metric("niqe", device=device)
print('完成niqe初始化')
clipiqa = pyiqa.create_metric("clipiqa", device=device)
print('完成clipiqa初始化')
musiq = pyiqa.create_metric("musiq", device=device)
print('完成musiq初始化')

# 获取所有图像路径，确保只获取图像文件
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
test_SR_paths = []
for ext in image_extensions:
    test_SR_paths.extend(glob.glob(os.path.join(args.SR_dir, ext)))
test_SR_paths = sorted(test_SR_paths)

# 确保有图像文件
if not test_SR_paths:
    raise ValueError(f"在目录 {args.SR_dir} 中未找到任何图像文件")

# 初始化指标字典（只包含实际使用的指标）
metrics = {"niqe": [], "musiq": [], "clipiqa": []}

# 处理每个图像
for i, sr_path in tqdm(enumerate(test_SR_paths), total=len(test_SR_paths)):
    try:
        # 打开图像并转换为RGB
        sr_img = Image.open(sr_path).convert("RGB")

        # 转换为张量并添加批次维度
        sr_tensor = transforms.ToTensor()(sr_img).to(device).unsqueeze(0)

        # 计算指标
        metrics["niqe"].append(niqe(sr_tensor).item())
        metrics["clipiqa"].append(clipiqa(sr_tensor).item())
        metrics["musiq"].append(musiq(sr_tensor).item())
    except Exception as e:
        print(f"处理图像 {sr_path} 时出错: {str(e)}")
        continue

# 计算平均值
for k in metrics.keys():
    if metrics[k]:  # 确保列表非空
        metrics[k] = np.mean(metrics[k])
    else:
        metrics[k] = None

# 打印结果
for k, v in metrics.items():
    if v is not None:
        if k == "niqe":
            print(f"{k}: {v:.3g}")
        else:
            print(f"{k}: {v:.4g}")
    else:
        print(f"{k}: 无有效数据")