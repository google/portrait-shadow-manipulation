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

import utils as utils
import tensorflow as tf


"""
  Free parameters to control the synthesis
"""
MAX_SIGMA = 50  # convert shadow from hard --> soft
INT_SIGMA = 0.5  # how mnuch intensity variation: 1 --> no variation; 0 --> intense variation
TONE_SIGMA = 0.1  # tone curve jitter: if set to None --> no color jitter
SS_SIGMA = 0.5  # probability to apply ss approximation: 1 --> not apply; 0 --> always apply


"""
  Data loader
"""
def decode_line_foreign_wild(line):
  """Decode the line to tensor. Foreign shadow generation using images in the wild. Each line has 7 fields."""
  data_dict = {}
  record_defaults = [''] * 7
  items = tf.io.decode_csv(line, record_defaults, field_delim=',')
  data_dict['image_path'] = items[0]
  data_dict['silhouette_path'] = items[2]
  data_dict['bbox'] = [tf.strings.to_number(items[3], tf.float32),
                       tf.strings.to_number(items[4], tf.float32),
                       tf.strings.to_number(items[5], tf.float32),
                       tf.strings.to_number(items[6], tf.float32)]

  data_dict['shadowed_before'] = utils.read_float(items[0], channel=3, itype='jpg', is_linear=True)
  data_dict['segmentation'] = utils.read_float(items[1], channel=1, itype='png', is_linear=False)
  data_dict['silhouette'] = utils.read_float(items[2], channel=1, itype='png', is_linear=False)

  return data_dict


def read_bbox(txt_path):
  """Read face bbox into a list for a given .txt path."""
  bboxs = []
  with open(txt_path, 'r') as f:
    lines = f.readlines()
    for bbox in lines:
      bboxs.append([int(float(x)) for x in bbox.strip().split(',')])
  return bboxs


def prepare_train_foreign_wild(
    data_dict, size, is_train=True):
  """Prepare training data from images in the wild."""
  out_dict = {}
  out_dict['image_path'] = data_dict['image_path']
  
  if is_train:  # random resize the image during training
    rsz_ratio = tf.random.uniform(shape=(), minval=0.8, maxval=1.2, dtype=tf.float32)
  else:
    rsz_ratio = tf.constant(1., dtype=tf.float32)
  image_concat = tf.concat([data_dict['shadowed_before'], data_dict['segmentation']], axis=2)

  bbox = data_dict['bbox']
  processed_images, process_params = align_images_and_segmentation(
  	image_concat, size=size, bbox=bbox, rsz=rsz_ratio, param_save=True, is_train=True)

  shadow_lit_image = processed_images[..., :3]
  segmentation = processed_images[..., -1:]

  curve_gain = 0.5 + tf.random.uniform([3], -TONE_SIGMA, TONE_SIGMA, tf.float32)
  shadow_occl_image = utils.apply_tone_curve(shadow_lit_image, gain=curve_gain, is_rgb=True)
  color_matrix = utils.get_ctm_ls(shadow_lit_image, shadow_occl_image)
  shadow_occl_image = utils.apply_ctm(shadow_occl_image, color_matrix)

  # Create a shadow map, randomly choosing between using perlin noise or a random silhouette image
  shadow_mask_hard = 1- data_dict['silhouette']
  shadow_mask_hard = tf.cond(
      tf.greater(tf.random.uniform([]), tf.constant(0.5)),
      lambda: utils.render_perlin_mask(size=size),
      lambda: utils.render_silhouette_mask(
          silhouette=shadow_mask_hard,
          size=size,
          segmentation=segmentation))
  shadow_mask_hard_inv = 1 - shadow_mask_hard

  # Randomly apply ss approximation
  prob_apply_ss = tf.random.uniform([])
  shadow_mask_ss = tf.cond(
    tf.greater(prob_apply_ss, tf.constant(SS_SIGMA)),
    lambda: utils.apply_ss_shadow_map(shadow_mask_hard),
    lambda: tf.image.grayscale_to_rgb(shadow_mask_hard))
  shadow_mask_ss_inv = 1 - shadow_mask_ss
  
  # Apply intensity variation
  intensity_mask = utils.get_brightness_mask(size=size, min_val=INT_SIGMA)
  shadow_mask_sv = shadow_mask_ss_inv * tf.expand_dims(intensity_mask, 2)

  # Apply shadow mask to image
  shadow_hard_image = shadow_mask_sv * shadow_occl_image + shadow_mask_ss * shadow_lit_image
  input_shadowed_hard = ((shadow_hard_image * 2) - 1) * segmentation
  image_concat = tf.concat(
      [input_shadowed_hard, shadow_lit_image, shadow_occl_image, shadow_mask_hard, segmentation], axis=2)
  
  if is_train:
    image_concat = utils.apply_geometric_augmentation(image_concat)
  
  segmentation = image_concat[..., -1:]
  out_dict['shadowed_hard'] = image_concat[..., :3]  # input_shadowed_hard
  out_dict['shadowed_before'] = ((image_concat[..., 3:6] * 2) - 1) * segmentation  # shadow_lit_image
  out_dict['shadowed_after'] = ((image_concat[..., -7:-4] * 2) - 1) * segmentation  # shadow_occl_image
  out_dict['shadow_mask'] = image_concat[..., -4:-1]  # shadow_mask_hard
  out_dict['segmentation'] = segmentation  # segmentation
  return out_dict


def align_images_and_segmentation(images, size, bbox, rsz=1., param_save=False, is_train=True):
  """Apply the same resize / crop to a batch of images with face alignment using the input face bbox."""
  with tf.name_scope('align_img_seg'):
    out_params = dict()
    if is_train:  # insert some uncertainty during training
      golden_ratio = tf.random.uniform(
          shape=(), minval=0.75, maxval=0.85, dtype=tf.float32)
    else:
      golden_ratio = 0.8  # roughly align faces to the center, heuristic number
    w = bbox[2]
    h = bbox[3]
    target_h = size[0]
    target_w = size[1]
    resize_ratio = (target_w * golden_ratio) / (w) * rsz
    out_params['resize_ratio'] = resize_ratio
    image_resize = utils.resize_image(images, rsz=resize_ratio)
    out_params['resize_height'] = tf.shape(image_resize)[0]
    out_params['resize_width'] = tf.shape(image_resize)[1]
    yhstart = tf.cast(((bbox[1] + h/2) * resize_ratio) - target_h/2, tf.int32)
    xwstart = tf.cast(((bbox[0] + w/2) * resize_ratio) - target_w/2, tf.int32)
    out_params['pad_offset_height'] = tf.maximum(0, -yhstart+1)
    out_params['pad_offset_width'] = tf.maximum(0, -xwstart+1)
    out_params['pad_target_height'] = tf.maximum(target_h, tf.shape(image_resize)[0] + tf.maximum(yhstart, 0)) + tf.maximum(0, -yhstart+1)
    out_params['pad_target_width'] = tf.maximum(target_w, tf.shape(image_resize)[1] + tf.maximum(xwstart, 0)) + tf.maximum(0, -xwstart+1)
    image_resize = tf.image.pad_to_bounding_box(
        image_resize,
        tf.maximum(0, -yhstart+1),
        tf.maximum(0, -xwstart+1),
        tf.maximum(target_h, tf.shape(image_resize)[0] + tf.maximum(yhstart, 0)) + tf.maximum(0, -yhstart+1),
        tf.maximum(target_w, tf.shape(image_resize)[1] + tf.maximum(xwstart, 0)) + tf.maximum(0, -xwstart+1))
    out_params['crop_offset_height'] = tf.maximum(yhstart, 0)
    out_params['crop_offset_width'] = tf.maximum(xwstart, 0)
    out_params['crop_target_height'] = target_h
    out_params['crop_target_width'] = target_w
    processed_images = tf.image.crop_to_bounding_box(
        image_resize,
        tf.maximum(yhstart, 0),
        tf.maximum(xwstart, 0),
        target_h,
        target_w)
  if param_save:
    return processed_images, out_params
  return processed_images


def input_fn(
    dataset_name,
    train_txt_paths,
    eval_txt_paths,
    is_train=True,
    image_size=(512, 512),
    batch_size=4,
    seed=None,
    n_interleave_workers=12,
    n_map_workers=12,
    shuffle_buffer_size=64):
  """Input function for training and eval parallel I/O.
  """

  with tf.name_scope('data-prep-%s'%(dataset_name)):
    if dataset_name == 'wild':
      decode_fn = decode_line_foreign_wild
      data_process_fn = prepare_train_foreign_wild
    else:
      raise NotImplementedError('Unknown dataset ', dataset_name)

  if is_train:
    with tf.name_scope('data-train-%s'%(dataset_name)):
      next_batch = _input_fn(
          decode_fn=decode_fn,
          is_train=True,
          dataset_txt_paths=train_txt_paths,
          image_size=image_size,
          batch_size=batch_size,
          seed=seed,
          process_fn=data_process_fn,
          n_interleave_workers=n_interleave_workers,
          n_map_workers=n_map_workers,
          shuffle_buffer_size=shuffle_buffer_size,
          )
  else:
    with tf.name_scope('data-eval-%s'%(dataset_name)):
      next_batch = _input_fn(
          decode_fn=decode_fn,
          is_train=False,
          dataset_txt_paths=eval_txt_paths,
          image_size=image_size,
          batch_size=batch_size,
          seed=seed,
          process_fn=data_process_fn,
          n_interleave_workers=n_interleave_workers,
          n_map_workers=n_map_workers,
          shuffle_buffer_size=shuffle_buffer_size,
          )
  return next_batch


def _input_fn(
    decode_fn,
    process_fn,
    is_train=True,
    dataset_txt_paths=[''],
    image_size=(512, 512),
    batch_size=4,
    seed=None,
    n_interleave_workers=12,
    n_map_workers=12,
    shuffle_buffer_size=64):
  
  """Customized input func with different decoding and processing functions.
  """

  dataset = (tf.data.Dataset.from_tensor_slices(dataset_txt_paths)
             .interleave(
                 lambda x: tf.data.TextLineDataset(x).map(
                     decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
                 cycle_length=tf.data.experimental.AUTOTUNE))

  # Parallelize processing and ignore errors
  dataset = dataset.map(
      lambda v: process_fn(v, image_size, is_train),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.apply(tf.data.experimental.ignore_errors())

  # Shuffle
  if is_train:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
  dataset = dataset.repeat()

  # Batching
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

  # Preprocessing on CPU while computing a training step
  dataset = dataset.prefetch(buffer_size=8)
  return dataset.make_one_shot_iterator().get_next()