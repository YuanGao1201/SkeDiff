# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os.path

import SimpleITK
from xlib.dataset.baseDataSet import Base_DataSet
from xlib.dataset.utils import *
import h5py
import numpy as np
from PIL import Image

class AlignDataSet(Base_DataSet):
  '''
  DataSet For unaligned data
  '''
  def __init__(self, opt):
    super(AlignDataSet, self).__init__()
    self.opt = opt
    # self.ext = '.h5'
    self.ext = '.nii.gz'
    self.dataset_paths = get_dataset_from_txt_file(self.opt.datasetfile)
    self.dataset_paths = sorted(self.dataset_paths)
    self.dataset_size = len(self.dataset_paths)
    self.dir_root = self.get_data_path
    self.data_augmentation = self.opt.data_augmentation(opt)

  @property
  def name(self):
    return 'AlignDataSet'

  @property
  def get_data_path(self):
    path = os.path.join(self.opt.dataroot)
    return path

  @property
  def num_samples(self):
    return self.dataset_size

  def get_image_path(self, root, index_name):
    # img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
    # ct_path = os.path.join(root, 'spineCT-128', index_name + self.ext)
    # xray1_path = os.path.join(root, 'DRR-128', index_name + '_1.png')
    # xray2_path = os.path.join(root, 'DRR-128', index_name + '_2.png')
    # ct_path = os.path.join(root, 'dataSets', index_name + self.ext)
    # xray1_path = os.path.join(root, 'dataSetsDRR-128', index_name + '_1.png')
    # xray2_path = os.path.join(root, 'dataSetsDRR-128', index_name + '_2.png')
    # ct_path = os.path.join(root, 'dataSets-supine', index_name + self.ext)
    # xray1_path = os.path.join(root, 'dataSetsDRR-tmp', index_name + '_1.png')
    # xray2_path = os.path.join(root, 'dataSetsDRR-tmp', index_name + '_2.png')
    # ct_path = os.path.join(root, 'dataSets-supine-preprocess3', index_name + self.ext)
    ct_path = os.path.join(root, 'hipCT-220-p1', index_name + self.ext)
    xray1_path = os.path.join(root, 'hipCTDRR-220-128-nor', index_name + '_1.png')
    xray2_path = os.path.join(root, 'hipCTDRR-220-128-nor', index_name + '_2.png')
    assert os.path.exists(ct_path), 'Path do not exist: {}'.format(ct_path)
    assert os.path.exists(xray1_path), 'Path do not exist: {}'.format(xray1_path)
    assert os.path.exists(xray2_path), 'Path do not exist: {}'.format(xray2_path)
    # return img_path
    return ct_path, xray1_path, xray2_path

  # def load_file(self, file_path):
  def load_file(self, ct_path, xray1_path, xray2_path):
    # hdf5 = h5py.File(file_path, 'r')
    # ct_data = np.asarray(hdf5['ct'])
    # x_ray1 = np.asarray(hdf5['xray1'])
    # x_ray2 = np.asarray(hdf5['xray2'])
    # x_ray1 = np.expand_dims(x_ray1, 0)
    # x_ray2 = np.expand_dims(x_ray2, 0)
    # hdf5.close()
    img_itk = SimpleITK.ReadImage(ct_path)
    ct_data = SimpleITK.GetArrayFromImage(img_itk)
    # aa = ct_data.min()
    # bb = ct_data.max()
    x_ray1 = Image.open(xray1_path)
    x_ray2 = Image.open(xray2_path)
    if not x_ray1.mode=='L':
      x_ray1 = x_ray1.convert('L')
    if not x_ray2.mode=='L':
      x_ray2 = x_ray2.convert('L')
    x_ray1 = np.expand_dims(x_ray1, 0)
    x_ray2 = np.expand_dims(x_ray2, 0)
    return ct_data, x_ray1, x_ray2

  '''
  generate batch
  '''
  def pull_item(self, item):
    file_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
    # ct_data, x_ray1, x_ray2 = self.load_file(file_path)
    ct_path, xray1_path, xray2_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
    ct_data, x_ray1, x_ray2 = self.load_file(ct_path, xray1_path, xray2_path)

    # Data Augmentation
    ct, xray1, xray2 = self.data_augmentation([ct_data, x_ray1, x_ray2])

    return ct, xray1, xray2, file_path






