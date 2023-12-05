import torch
import os
import sys
from re import I
os.environ["CUDA_VISIBLE_DEVICES"]=f"{0}"
#%env CUDA_VISIBLE_DEVICES=3
#torch.cuda.set_device(GPU_NUM)
device = f'cuda:0'
sys.path.append('../')
from vq_gan_3d.model.vqgan import VQGAN
from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset
import matplotlib.pyplot as plt
import SimpleITK as sitk
from ddpm import Unet3D, GaussianDiffusion, Trainer
import pytorch_ssim
from train.get_dataset import get_dataset
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import SimpleITK as sitk
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms as T
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# gy FLOPs
from thop import profile


DDPM_CHECKPOINT = '/home/fi/GY/medicaldiffusion/checkpoints/ddpm/DEFAULT/gy/model-149.pt'
VQGAN_CHECKPOINT = '/home/fi/GY/medicaldiffusion/checkpoints/vq_gan/DEFAULT/gy/lightning_logs/version_0/checkpoints/epoch=298-step=300000-10000-train/recon_loss=0.07.ckpt'

# with initialize(config_path="../config/"):
#     cfg=compose(config_name="base_cfg.yaml", overrides=[
#         "model=ddpm",
#         "dataset=default",
#         f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
#         "model.diffusion_img_size=32",
#         "model.diffusion_depth_size=32",
#         "model.diffusion_num_channels=8",
#         "model.dim_mults=[1,2,4,8]",
#         "model.batch_size=10 ",
#         "model.gpus=0 ",
#         ])
@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    train_dataset, _, _ = get_dataset(cfg)

    #cat
    # model = Unet3D(
    #     dim=cfg.model.diffusion_img_size,
    #     dim_mults=cfg.model.dim_mults,
    #     # channels=cfg.model.diffusion_num_channels,
    #     channels=10,  # gy cat condition
    #     out_dim=8,  # gy cat condition
    #     has_cond_gy=True,  # gy
    #     conformer_crossattn=False  # gy
    # ).cuda()
    #cross-attn
    model = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
        # channels=10,#gy cat condition
        out_dim=8,  # gy cat condition
        has_cond_gy=True,  # gy
        conformer_crossattn=False,  # gy
        x_crossattn=True  # gy
    ).cuda()

    # input_cond1 = torch.randn(1, 1, 128, 128).cuda()
    # input_cond2 = torch.randn(1, 1, 128, 128).cuda()
    # time1 = torch.tensor(299).cuda()
    # macs, params = profile(model, inputs=[torch.randn(1, 8, 32, 32, 32).cuda(), time1, None, [input_cond1, input_cond2]])

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).cuda()

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        # train_batch_size=cfg.model.batch_size,
        train_batch_size=1,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        # results_folder='/home/fi/GY/medicaldiffusion/evaluation/result-hip',
        results_folder='/home/fi/GY/medicaldiffusion/evaluation/result-crossattn',
        num_workers=cfg.model.num_workers,
        # logger=cfg.model.logger
    )

    trainer.load(DDPM_CHECKPOINT, map_location='cuda:0')

    # sample = trainer.ema_model.sample(batch_size=1)

    def num_to_groups(num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr


    def tensor_to_ct(tensor, path,affine,spacing):
        import numpy
        import nibabel as nib
        import SimpleITK as sitk
        tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
        tensor = tensor.to('cpu')
        tensor = tensor.squeeze().numpy()
        # spacing = (spacing[0].item(),spacing[1].item(),spacing[2].item())
        spacing = (1, 1, 1)
        # affine = np.eye(4)
        itkimage = sitk.GetImageFromArray(tensor, isVector=False)
        itkimage.SetSpacing(spacing)
        itkimage.SetOrigin((0,0,0))
        itkimage = (itkimage + 1.0) * 127.5
        # sitk.WriteImage(itkimage, path, True)
        sitk.WriteImage(itkimage, path)


    def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
        tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
        images = map(T.ToPILImage(), tensor.unbind(dim=1))
        first_img, *rest_imgs = images
        first_img.save(path, save_all=True, append_images=rest_imgs,
                       duration=duration, loop=loop, optimize=optimize)
        return images


    for i in range(291):#291 for spine. 1151 for hip
        datas = next(trainer.dl)
        affine = datas['affine']
        spacing = datas['spacing']
        drr1 = datas['drr1'].cuda()
        drr2 = datas['drr2'].cuda()
        name = datas['name']
        # if name[0] == '1.3.6.1.4.1.9328.50.4.0010_l.nii.gz':# only for test
        with torch.no_grad():
            num_samples = trainer.num_sample_rows ** 2
            batches = num_to_groups(num_samples, trainer.batch_size)  # [1,1,1]
            # print(drr1[0,:,:,:].shape)
            all_videos_list = list(
                map(lambda n: trainer.ema_model.sample(batch_size=n, condtion=[drr1[0].unsqueeze(0), drr2[0].unsqueeze(0)]),
                    batches))  # inference condi
            all_videos_list = torch.cat(all_videos_list, dim=0)

        ct_path = str(trainer.results_folder / str(f'{name[0]}.nii.gz'))
        tensor_to_ct(all_videos_list, ct_path, affine[0], [spacing[0][0], spacing[1][0], spacing[2][0]])

        # all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
        # one_gif = rearrange(
        #     all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=trainer.num_sample_rows)
        # video_path = str(trainer.results_folder / str(f'{name[0]}.gif'))
        # video_tensor_to_gif(one_gif, video_path)


if __name__ == '__main__':
    run()




# sum_ssim = 0
# for k in range(20):
#     for i,dat in enumerate(train_loader):
#         dat = dat['data']
#         if len(dat)!=2:
#             print("Length: ", len(dat))
#             break
#         img1 = dat[0]
#         img2 = dat[1]
#
#         msssim = pytorch_ssim.msssim_3d(img1,img2)
#         sum_ssim = sum_ssim+msssim
#     print(sum_ssim/((k+1)*(i+1)))