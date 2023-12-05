from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse
from PIL import Image
import numpy as np
import torch
import nibabel as nib
import numpy as np
import SimpleITK as sitk


from skimage import transform


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    # tio.CropOrPad(target_shape=(256, 256, 32))
    tio.CropOrPad(target_shape=(128, 128, 128))
    # tio.CropOrPad(target_shape=(200, 200, 200))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])


class DEFAULTDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        nifti_file_names = os.listdir(self.root_dir)
        folder_names = [os.path.join(
            self.root_dir, nifti_file_name) for nifti_file_name in nifti_file_names if nifti_file_name.endswith('.nii.gz')]
        return folder_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        # ct_data = SimpleITK.GetArrayFromImage(img_itk)
        img = tio.ScalarImage(self.file_paths[idx])
        # print(self.file_paths[idx].replace('spineCT-128', 'DRR').replace('.nii.gz', '_1.png'))
        # drr1 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('spineCT-128', 'DRR-128').replace('.nii.gz', '_1.png')))[np.newaxis, :, :],dtype=torch.float32)
        # drr2 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('spineCT-128', 'DRR-128').replace('.nii.gz', '_2.png')))[np.newaxis, :, :],dtype=torch.float32)
        # drr1 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('dataSets-supine-preprocess3', 'dataSetsDRR-p2-128').replace('.nii.gz', '_1.png')))[np.newaxis, :, :],dtype=torch.float32)
        # drr2 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('dataSets-supine-preprocess3', 'dataSetsDRR-p2-128').replace('.nii.gz', '_2.png')))[np.newaxis, :, :],dtype=torch.float32)
        drr1 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('dataSets-supine-preprocess3-test', 'dataSetsDRR-p2-128-test').replace('.nii.gz', '_1.png')))[np.newaxis, :, :],dtype=torch.float32)
        drr2 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('dataSets-supine-preprocess3-test', 'dataSetsDRR-p2-128-test').replace('.nii.gz', '_2.png')))[np.newaxis, :, :],dtype=torch.float32)
        # drr1 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('hipCT-220-p1', 'hipCTDRR-220-128-nor').replace('.nii.gz', '_1.png')))[np.newaxis, :, :],dtype=torch.float32)
        # drr2 = torch.tensor(np.array(Image.open(self.file_paths[idx].replace('hipCT-220-p1', 'hipCTDRR-220-128-nor').replace('.nii.gz', '_2.png')))[np.newaxis, :, :],dtype=torch.float32)
        # drr1 = torch.tensor(np.array(Image.open('/home/fi/lyh/CT1Kdata/DRR/liver-12-2.png'))[np.newaxis, :, :], dtype=torch.float32)
        # drr2 = torch.tensor(np.array(Image.open('/home/fi/lyh/CT1Kdata/DRR/liver-12-2-pro.png'))[np.newaxis, :, :], dtype=torch.float32)
        name = self.file_paths[idx].split('/')[-1]
        #lyh
        affine = img.affine
        spacing = img.spacing

        img = self.preprocessing(img)

        # img = img.data
        # img = transform.resize(img,(128,128,128))
        # img = tio.ScalarImage(tensor=img)

        img = self.transforms(img)

        # gy, 创建 NIfTI 图像对象
        # itkimage = sitk.GetImageFromArray(img.data.numpy()[0], isVector=False)
        # itkimage.SetSpacing((1.0, 1.0, 1.0))
        # itkimage.SetOrigin((0,0,0))
        # sitk.WriteImage(itkimage, str(idx)+'.nii.gz', True)

        # return {'data': img.data.permute(0, -1, 1, 2)}
        return {'data': img.data.permute(0, -1, 1, 2), 'drr1': drr1, 'drr2': drr2,'affine': affine,'spacing': spacing, 'name': name}
