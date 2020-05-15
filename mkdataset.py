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
import os
import tensorflow as tf
import utils as utils
import datasets
from PIL import Image


flags.DEFINE_string('mode', 'train', 'Mode to run, train or eval')
flags.DEFINE_integer('batchsize', 1, 'Batch size')
flags.DEFINE_integer('size', 256, 'Training image size')
flags.DEFINE_integer('num_in_channels', 3, 'Input channel #')
flags.DEFINE_string('gttype', 'before', 'Name of the experiment')
flags.DEFINE_integer('max_step', 10, 'Max steps, here == the number of images to generate')
flags.DEFINE_list('trainwildpaths',
                  ['example.txt'],
                  'list of training dataset txt paths')
flags.DEFINE_list('evalwildpaths',
                  ['example.txt'],
                  'list of training dataset txt paths')
flags.DEFINE_string('out_dir', './example_out', 'Output dir')

FLAGS = flags.FLAGS

def main(_):
    train_paths = {}
    eval_paths = {}
    maxstep = 1
    train_paths['wild'] = FLAGS.trainwildpaths
    eval_paths['wild'] = FLAGS.evalwildpaths
    
    if FLAGS.mode == 'train':
      next_batch = datasets.input_fn(
        dataset_name='wild',
        train_txt_paths=train_paths['wild'],
        eval_txt_paths=eval_paths['wild'],
        is_train=FLAGS.mode == 'train',
        image_size=[FLAGS.size, FLAGS.size],
        batch_size=FLAGS.batchsize,
        seed=None,
        n_interleave_workers=1,
        n_map_workers=12,
        shuffle_buffer_size=4)
      maxstep = FLAGS.max_step
    
    input_image = next_batch['shadowed_hard']
    input_image.set_shape([FLAGS.batchsize, FLAGS.size, FLAGS.size, FLAGS.num_in_channels])
    gt_image_l1 = next_batch['shadowed_before']
    gt_image_l1.set_shape([FLAGS.batchsize, FLAGS.size, FLAGS.size, 3])
    image_mask = next_batch['segmentation']
    image_mask.set_shape([FLAGS.batchsize, FLAGS.size, FLAGS.size, 1])
    image_path = next_batch['image_path']
    image_mask = tf.image.convert_image_dtype(image_mask, dtype=tf.uint8, saturate=True)
    gt_image_l1 = tf.image.convert_image_dtype(utils.rgb_to_srgb((gt_image_l1 + 1)/2), dtype=tf.uint8, saturate=True)
    input_image = tf.image.convert_image_dtype(utils.rgb_to_srgb((input_image + 1)/2), dtype=tf.uint8, saturate=True)
    
    gt_path = '%s/shadow_gt'%(FLAGS.out_dir)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    in_path = '%s/shadow_input'%(FLAGS.out_dir)
    if not os.path.exists(in_path):
        os.makedirs(in_path)
    mask_path = '%s/shadow_input_mask_ind'%(FLAGS.out_dir)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    
    sess = tf.compat.v1.Session()
    for iter in range(maxstep):
        fetch_list = [input_image, gt_image_l1, image_path, image_mask]
        shadow_image, gt_image, input_image_path, image_mask_out = sess.run(fetch_list)
        input_image_path = input_image_path[0].decode(encoding="utf-8")
        print('step %d'%(iter), input_image_path)
        
        Image.fromarray(gt_image[0]).save('%s/%05d.png'%(gt_path, iter))
        Image.fromarray(shadow_image[0]).save('%s/%05d.png'%(in_path, iter))
        Image.fromarray(image_mask_out[0,...,0]).save('%s/%05d.png'%(mask_path, iter))

if __name__ == '__main__':
    app.run(main)
