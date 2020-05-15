# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
import os, glob
import numpy as np
import tqdm
    

flags.DEFINE_list(
    'image_base_paths', [None],
    'Path to directory where text files are to be written.')
flags.DEFINE_list(
    'silhouette_paths', [None],
    'Indoor hdr file list.')
flags.DEFINE_list(
    'file_types', [None],
    'file types in each dir.')
flags.DEFINE_string(
    'output_dir', None,
    'Path to directory where split text files are to be written.')
FLAGS = flags.FLAGS


def get_paths_for_dirs(dir_path, sstr='.jpg'):
  image_paths = []
  for full_dir, _, filenames in os.walk(dir_path):
    image_paths.extend([
        os.path.join(full_dir, filename)
        for filename in filenames
        if sstr in filename
    ])
  return image_paths, len(image_paths) * [dir_path]


def read_bbox(txt_path):
  """Read face bbox into a list for a given .txt path."""
  bboxs = []
  with open(txt_path, 'r') as f:
    lines = f.readlines()
    for bbox in lines:
      bboxs.append([int(float(x)) for x in bbox.strip().split(',')])
  return bboxs


def prepare_train_list():
  all_paths = []
  all_base_paths = []
  all_silhouette = []
  for base_dir, file_type in zip(FLAGS.image_base_paths, FLAGS.file_types):
    try:
      path_i, base_i = get_paths_for_dirs(base_dir, sstr=file_type)
    except Exception as e:
      print('Not able to get paths from %s'%(base_dir))
    all_paths.extend(path_i)
    all_base_paths.extend(base_i)
  seed = np.random.randint(100000)
  np.random.seed(seed)
  all_paths_permute = np.random.permutation(all_paths)
  np.random.seed(seed)
  all_base_paths_permute = np.random.permutation(all_base_paths)
  num_train = len(all_base_paths_permute)
  
  for base_dir in FLAGS.silhouette_paths:
    path_s_i, _ = get_paths_for_dirs(base_dir, sstr='.png')
    all_silhouette.extend(path_s_i)
  all_silhouette_permute = list(
      np.random.permutation(all_silhouette))
  train_ct = 0
  ind_ct = 0
  pbar = tqdm.tqdm(total=num_train)

  with open(FLAGS.output_dir, 'w') as out_file:
    while train_ct < num_train:
      img_path = all_paths_permute[np.mod(ind_ct, num_train)]
      img_ext = os.path.basename(img_path).split('.')[1]
      img_base_path = all_base_paths_permute[np.mod(ind_ct, num_train)]
      image_mask_base_path = img_path.replace(img_base_path, img_base_path+'_mask_ind')
      image_mask_paths = glob.glob(image_mask_base_path.replace('.'+img_ext, '*.png'))  # in case there are more faces in one image
      mask_file_ind = np.random.choice(range(len(image_mask_paths)))
      image_mask_path = image_mask_paths[mask_file_ind]
      bboxs_path = image_mask_base_path.replace('.'+img_ext, '.txt')
      if not os.path.exists(bboxs_path):
        print('bbox not found', bboxs_path)
        ind_ct += 1
        continue
      bboxs = read_bbox(bboxs_path)[0]
      sil_path = np.random.choice(all_silhouette_permute)
      out_file.write('%s,%s,%s,%s,%s,%s,%s\n' %(img_path, image_mask_path, sil_path, bboxs[0], bboxs[1], bboxs[2], bboxs[3]))
      train_ct += 1
      ind_ct += 1
      pbar.update(1)
  pbar.close()


def main(_):
  prepare_train_list()


if __name__ == '__main__':
  app.run(main)
