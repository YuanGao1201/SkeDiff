import torch
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# env CUDA_VISIBLE_DEVICES=0
torch.cuda.set_device(0)
device = 'cuda:0'
sys.path.append('../')
from vq_gan_3d.model.vqgan import VQGAN
from train.get_dataset import get_dataset
import matplotlib.pyplot as plt
import SimpleITK as sitk
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.default import DEFAULTDataset


DDPM_CHECKPOINT = '/home/fi/GY/medicaldiffusion/checkpoints/ddpm/DEFAULT/gy/model-270.pt'
VQGAN_CHECKPOINT = '/home/fi/GY/medicaldiffusion/checkpoints/vq_gan/DEFAULT/gy/lightning_logs/version_0/checkpoints/epoch=298-step=300000-10000-train/recon_loss=0.07.ckpt'


# vqgan = VQGAN.load_from_checkpoint('/data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints_generation/knee_mri_gen/lightning_logs/version_0/checkpoints/epoch=228-step=207000-train/recon_loss=0.91.ckpt', map_location=device)
vqgan = VQGAN.load_from_checkpoint(VQGAN_CHECKPOINT)
vqgan = vqgan.to(device)
vqgan.eval()

val_dataset = DEFAULTDataset(root_dir='/home/fi/lyh/CT1Kdata/dataSets/')
for i in range(3029):
    input_ = torch.tensor(val_dataset[i]['data'][None]).to(device)
    with torch.no_grad():
        output_ = vqgan(input_)
        output_ = (sitk.GetImageFromArray(output_[1][0][0].cpu()) + 1.0) * 127.5
        input_ = (sitk.GetImageFromArray(input_[0][0].cpu()) + 1.0) * 127.5
        sitk.WriteImage(output_,
                        '/home/fi/GY/medicaldiffusion/evaluation/result_vq/' + val_dataset[i]['name'])
        sitk.WriteImage(input_,
                        '/home/fi/GY/medicaldiffusion/evaluation/result_vq/input/' + val_dataset[i]['name'])

