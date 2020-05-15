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

import numpy as np
import scipy.spatial
import tensorflow as tf


"""
  Free parameters to control the synthesis
"""
_MAX_SS_SIGMA = 5  # control subsurface scattering strength
_MAX_BLUR_SIGMA = 10  # control spatially varying blur strength
_SV_SIGMA = 0.5  # 1. --> not sv blur on boudary; 0. -> always sv blur


"""
  Common I/O Utils
"""
def read_float(path, channel=3, itype='jpg', is_linear=False):
  """Decode an image from string. Return 3 channels.

  Args:
    path: a tf string of the image path.
    channel: scalar, number of channels of the input image.
    itype: string, the type of the input image.
    is_linear: a bool, indicates whether or not to convert to linear color space.
    (undo gamma correction)
  
  Returns:
    A 3D tensor of the read-in image.
  """
  image_string = tf.io.read_file(path)
  if itype == 'jpg':
    image = tf.image.decode_jpeg(image_string, channels=channel)
  elif itype == 'png':
    image = tf.image.decode_png(image_string, channels=channel)
  image = tf.image.convert_image_dtype(image, tf.float32)
  if is_linear:
    image = srgb_to_rgb(image)
  return image


def srgb_to_rgb(srgb, name='srgb_to_rgb'):
  """Converts sRGB to linear RGB."""
  with tf.name_scope(name):
    mask = tf.cast(tf.greater(srgb, 0.04045), dtype=srgb.dtype)
  return (srgb / 12.92 * (1.0 - mask) + tf.pow(
      (srgb + 0.055) / 1.055, 2.4) * mask)


def rgb_to_srgb(rgb, name='rgb_to_srgb'):
  """Converts linear RGB to sRGB."""
  with tf.name_scope(name):
    mask = tf.cast(tf.greater(rgb, 0.0031308), dtype=tf.float32)
  return (rgb * 12.92 * (1.0 - mask) +
          (tf.pow(rgb, 1.0 / 2.4) * 1.055 - 0.055) * mask)


def resize_image(image, new_sizeh=None, new_sizew=None, rsz=None):
  """Customized image resizing op."""
  with tf.name_scope('resize_image'):
    if new_sizeh is None:
      height = tf.cast(tf.shape(image)[0], tf.float32)
      width = tf.cast(tf.shape(image)[1], tf.float32)
      new_sizeh = tf.cast(height * rsz, tf.int32)
      new_sizew = tf.cast(width * rsz, tf.int32)
  return tf.compat.v1.image.resize(
      image, [new_sizeh, new_sizew],
      method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)

"""
  Subsurface scattering approxmiation
"""
def apply_ss_shadow_map(mask):
  """Apply subsurface scattering approximation to the shadow mask.

  Args:
    mask: A Tensor of shape [H, W, 1].

  Returns:
    A Tensor of shape [H, W, 3] that is applied with wavelength-dependent blur.
  """
  r = tf.random.uniform(
      shape=(), minval=0.5, maxval=_MAX_SS_SIGMA, dtype=tf.float32)  # a global scalar to scale all the blur size
  shadow_map = wavelength_filter(mask, num_lv=6, scale=r, is_rgb=False)
  shadow_map = tf.minimum(1., shadow_map/0.6)  # a heuristic scalar for more stable normalization
  return shadow_map


def wavelength_filter(input_img, num_lv=6, scale=5, is_rgb=False, name='wavelength_filter'):
  """Image-based subsurface scattering approximation

  Parameters from the NVIDIA screen-space subsurface scattering (SS) slide 98.

  http://developer.download.nvidia.com/presentations/2007/gdc/Advanced_Skin.pdf

  Args:
    input_img: a 3D tensor [H, W, C].
    num_lv: a scalar that specifies the number of Gaussian filter levels in the SS model.
    scale: a scalar that is the scale used to calibrate the kernel size into # pixels based on the size of the face in the image.
    is_rgb: a bool that indicates whether input is grayscale(c=1) or rgb(c=3).
    name: string, name of the graph.

  Returns:
    A 3D tensor after approximated with subsurface scattering.
  """
  with tf.name_scope(name):
    scale = tf.cast(scale, tf.float32)
    ss_weights = np.array([[0.042, 0.22, 0.437, 0.635],
                           [0.220, 0.101, 0.355, 0.365],
                           [0.433, 0.119, 0.208, 0],
                           [0.753, 0.114, 0, 0],
                           [1.412, 0.364, 0, 0],
                           [2.722, 0.080, 0, 0]])
    ss_weights_norm = np.sum(ss_weights, 0)
    img_blur_rgb = 0.
    for lv in range(num_lv):
      if lv != 0:
        blur_kernel = ss_weights[lv, 0] * scale
      else:
        blur_kernel = ss_weights[lv, 0] * scale
      rgb_weights = ss_weights[lv, 1:]
      if not is_rgb:
        blur_img = gaussian_filter(tf.expand_dims(input_img, 0), blur_kernel)[0]
        blur_r = blur_img * rgb_weights[0] * 1.2
        blur_g = blur_img * rgb_weights[1]
        blur_b = blur_img * rgb_weights[2]
      else:
        blur_r = gaussian_filter(
            tf.expand_dims(input_img[..., 0, tf.newaxis], 0),
            blur_kernel)[0] * rgb_weights[0] * 1. / ss_weights_norm[1]
        blur_g = gaussian_filter(
            tf.expand_dims(input_img[..., 1, tf.newaxis], 0),
            blur_kernel)[0] * rgb_weights[1] * 1. / ss_weights_norm[2]
        blur_b = gaussian_filter(
            tf.expand_dims(input_img[..., 2, tf.newaxis], 0),
            blur_kernel)[0] * rgb_weights[2] * 1. / ss_weights_norm[3]
      img_blur = tf.concat([blur_r, blur_g, blur_b], 2)
      img_blur_rgb += img_blur
  return img_blur_rgb


def gaussian_filter(image, sigma, pad_mode='REFLECT', name='gaussian_filter'):
  """Applies Gaussian filter to an image using depthwise conv.

  Args:
    image: 4-D Tensor with float32 dtype and shape [N, H, W, C].
    sigma: Positive float or 0-D Tensor.
    pad_mode: String, mode argument for tf.pad. Default is 'REFLECT' for
      whole-sample symmetric padding.
    name: A string to name this part of the graph.

  Returns:
    Filtered image, has the same shape with the input.
  """
  with tf.name_scope(name):
    image.shape.assert_has_rank(4)
    sigma = tf.cast(sigma, tf.float32)
    sigma.shape.assert_has_rank(0)  # sigma is a scalar.

    channels = tf.shape(image)[3]
    r = tf.cast(tf.math.ceil(2.0 * sigma), tf.int32)
    n = tf.range(-tf.cast(r, tf.float32), tf.cast(r, tf.float32) + 1)
    coeffs = tf.exp(-0.5 * (n / sigma)**2)
    coeffs /= tf.reduce_sum(coeffs)
    coeffs_x = tf.tile(tf.reshape(coeffs, (1, -1, 1, 1)), (1, 1, channels, 1))
    coeffs_y = tf.reshape(coeffs_x, (2 * r + 1, 1, channels, 1))

    padded = tf.pad(image, ((0, 0), (r, r), (r, r), (0, 0)), pad_mode)
    with tf.device('/cpu:0'):  # seems necessary for depthwise_conv2d
      filtered = tf.nn.depthwise_conv2d(
          padded, coeffs_x, (1, 1, 1, 1), 'VALID', name='filter_x')
      filtered = tf.nn.depthwise_conv2d(
          filtered, coeffs_y, (1, 1, 1, 1), 'VALID', name='filter_y')
      filtered.set_shape(image.shape)
  return filtered


"""
  Spatially varying utils
"""
def apply_disc_filter(input_img, kernel_sz, is_rgb=True):
  """Apply disc filtering to the input image with a specified kernel size.
  To handle large kernel sizes, this is operated (and thus approximated) in 
  frequency domain (fft).

  Args:
    input_img: a 2D or 3D tensor. [H, W, 1] or [H, W].
    kernel_sz: a scalar tensor that specifies the disc kernel size.
    is_rgb: a bool that indicates whether FFT is grayscale(c=1) or rgb(c=3).

  Returns:
    A Tensor after applied disc filter, has the same size as the input tensor.
  """
  if kernel_sz == 0:
    raise Warning('Input kenrel size is 0.')
    return input_img
  
  disc = create_disc_filter(kernel_sz)
  offset = kernel_sz - 1
  # if len(tf.shape(input_img)) == 2:
  #   padding_img = [[0, kernel_sz], [0, kernel_sz]]
  # elif len(tf.shape(input_img)) == 3:
  padding_img = [[0, kernel_sz], [0, kernel_sz], [0, 0]]
  img_padded = tf.pad(input_img, padding_img, 'constant')
  paddings = [[0, tf.shape(img_padded)[0] - tf.shape(disc)[0]],
              [0, tf.shape(img_padded)[1] - tf.shape(disc)[1]]]
  disc_padded = tf.pad(disc, paddings)
  # if len(tf.shape(input_img)) == 2:
  #   img_blurred = fft_filter(
  #       img_padded, disc_padded)[offset:offset + tf.shape(input_img)[0],
  #                                offset:offset + tf.shape(input_img)[1]]
  # else:
  img_blurred = fft3_filter(
      img_padded, disc_padded,
      is_rgb=is_rgb)[offset:offset + tf.shape(input_img)[0],
                     offset:offset + tf.shape(input_img)[1]]
  return img_blurred


def create_disc_filter(r):
  """Create a disc filter of radius r.
  Args:
    r: an int of the kernel radius.

  Returns:
    disk filter: A 2D Tensor
  """
  x, y = tf.meshgrid(tf.range(-r, r + 1), tf.range(-r, r + 1))
  mask = tf.less_equal(tf.pow(x, 2) + tf.pow(y, 2), tf.pow(r, 2))
  mask = tf.cast(mask, tf.float32)
  mask /= tf.reduce_sum(mask)
  return mask


def get_brightness_mask(size, min_val=0.5):
  """Render per-pixel intensity variation mask within [min_val, 1.].

  Args:
    size: A 2D tensor of target mask size.
  
  Returns:
    A Tensor of shape [H, W, 1] that is generated with perlin noise pattern.
  """
  perlin_map = perlin_collection((size[0], size[1]), [2, 2], 2,
                                 tf.random.uniform([], 0.05, 0.25))
  perlin_map = perlin_map / (1. / (min_val + 1e-6)) + min_val
  perlin_map = tf.minimum(perlin_map, 1.)
  return perlin_map


def fft_filter(img, kernel):
  """Apply FFT to a 2D tensor.

  Args:
    img: a 2D tensor of the input image [H, W].
    kernel: a 2D tensor of the kernel.
  
  Returns:
    a 2D tensor applied with a filter using FFT.
  """
  with tf.name_scope('fft2d_gray'):
    img = tf.cast(img, tf.complex64)
    kernel = tf.cast(kernel, tf.complex64)
    img_filtered = tf.cast(
        tf.abs(tf.signal.ifft2d(tf.multiply(tf.signal.fft2d(img), tf.signal.fft2d(kernel)))),
        tf.float32)
  return img_filtered


def fft3_filter(img, kernel, is_rgb=True):
  """Apply FFT to a 3D tensor.

  Args:
    img: a 3D tensor of the input image [H, W, C].
    kernel: a 2D tensor of the kernel.
    is_rgb: a bool that indicates whether input is rgb or not.
  
  Returns:
    a filtered 3D tensor, has the same size as input.
  """
  with tf.name_scope('fft2d_rgb'):
    img = tf.cast(img, tf.complex64)
    kernel = tf.cast(kernel, tf.complex64)
  if not is_rgb:
    img_r = fft_filter(img[..., 0], kernel)
    img_r = tf.expand_dims(img_r, 2)
    return img_r
  else:
    img_r = fft_filter(img[..., 0], kernel)
    img_g = fft_filter(img[..., 1], kernel)
    img_b = fft_filter(img[..., 2], kernel)
    img_filtered = tf.stack([img_r, img_g, img_b], 2)
  return img_filtered


def perlin_collection(size, reso, octaves, persistence):
  """Generate perlin patterns of varying frequencies.

  Args:
    size: a tuple of the target noise pattern size.
    reso: a tuple that specifies the resolution along lateral and longitudinal.
    octaves: int, number of octaves to use in the perlin model.
    persistence: int, persistence applied to every iteration of the generation.
  
  Returns:
    a 2D tensor of the perlin noise pattern.
  """
  noise = tf.zeros(size)
  amplitude = 1.0

  for _ in range(octaves):
    noise += amplitude * perlin(size, reso)
    amplitude *= persistence
    reso[0] *= 2
    reso[1] *= 2

  return noise


def perlin(size, reso):
  """Generate a perlin noise pattern, with specified frequency along x and y.

  Theory: https://flafla2.github.io/2014/08/09/perlinnoise.html

  Args:
    size: a tuple of integers of the target shape of the noise pattern.
    reso: reso: a tuple that specifies the resolution along lateral and longitudinal (x and y).
  
  Returns:
    a 2D tensor of the target size.
  """
  ysample = tf.linspace(0.0, reso[0], size[0])
  xsample = tf.linspace(0.0, reso[1], size[1])
  xygrid = tf.stack(tf.meshgrid(ysample, xsample), 2)
  xygrid = tf.math.mod(tf.transpose(xygrid, [1, 0, 2]), 1.0)

  xyfade = (6.0 * xygrid**5) - (15.0 * xygrid**4) + (10.0 * xygrid**3)
  angles = 2.0 * np.pi * tf.random.uniform([reso[0] + 1, reso[1] + 1])
  grads = tf.stack([tf.cos(angles), tf.sin(angles)], 2)

  gradone = tf.compat.v1.image.resize(grads[0:-1, 0:-1], [size[0], size[1]], 'nearest')
  gradtwo = tf.compat.v1.image.resize(grads[1:, 0:-1], [size[0], size[1]], 'nearest')
  gradthr = tf.compat.v1.image.resize(grads[0:-1, 1:], [size[0], size[1]], 'nearest')
  gradfou = tf.compat.v1.image.resize(grads[1:, 1:], [size[0], size[1]], 'nearest')

  gradone = tf.reduce_sum(gradone * tf.stack([xygrid[:, :, 0], xygrid[:, :, 1]], 2), 2)
  gradtwo = tf.reduce_sum(gradtwo * tf.stack([xygrid[:, :, 0] - 1, xygrid[:, :, 1]], 2), 2)
  gradthr = tf.reduce_sum(gradthr * tf.stack([xygrid[:, :, 0], xygrid[:, :, 1] - 1], 2), 2)
  gradfou = tf.reduce_sum(gradfou * tf.stack([xygrid[:, :, 0] - 1, xygrid[:, :, 1] - 1], 2), 2)

  inteone = (gradone * (1.0 - xyfade[:, :, 0])) + (gradtwo * xyfade[:, :, 0])
  intetwo = (gradthr * (1.0 - xyfade[:, :, 0])) + (gradfou * xyfade[:, :, 0])
  intethr = (inteone * (1.0 - xyfade[:, :, 1])) + (intetwo * xyfade[:, :, 1])

  return tf.sqrt(2.0) * intethr


def apply_spatially_varying_blur(image, blur_size=2, blurtype='disk'):
  """Apply spatially-varying blur to an image.
  Using pyramid to approximate for efficiency
  
  Args:
    image: a 3D image tensor [H, W, C].
    blur_size: base value for the blur size in the pyramic.
    blurtype: type of blur, either 'disk' or 'gaussian'.
  
  Returns:
    a 2D tensor of the target size.
  """
  pyramid = create_pyramid(image, blur_size=blur_size, blurtype=blurtype)
  image_blurred = apply_pyramid_blend(pyramid)
  return image_blurred


def apply_pyramid_blend(pyramid):
  """Reconstruct an image using bilinear interpolation between pyramid levels.

  Args:
    pyramid: a list of tensors applied with different blur levels.
  
  Returns:
    A reconstructed 3D tensor that is collapsed from the input pyramid.
  """
  num_levels = 3
  guidance_perlin_base = perlin_collection(
      (tf.shape(pyramid[0])[0], tf.shape(pyramid[0])[1]), [2, 2], 1,
      tf.random.uniform([], 0.05, 0.25))
  guidance_perlin_base -= tf.reduce_min(guidance_perlin_base)
  guidance_perlin_base /= tf.reduce_max(guidance_perlin_base)
  guidance_blur = tf.clip_by_value(guidance_perlin_base / (1. / num_levels),
                                   0.0, num_levels)
  image_reconst = pyramid
  for i in range(int(num_levels) - 2, -1, -1):
    alpha = tf.clip_by_value(guidance_blur - i, 0., 1.)
    alpha = tf.expand_dims(alpha, 2)
    image_reconst[i] = lerp(pyramid[i], image_reconst[i + 1], alpha)
  return image_reconst[0]


def create_pyramid(image, blur_size=2, blurtype='disk'):
  """Create a pyramid of different levels of disk blur.

  Args:
    image: a 2D or 3D tensor of the input image.
    blur_size: base value for the blur size in the pyramic.
    blurtype: a string that specifies the kind of blur, either disk or gaussian.
  
  Returns:
    Pyramid: a list of tensors applied with different blur kernels.
  """
  image_pyramid = []
  for i in range(3):
    rsz = np.power(2, i) * blur_size
    if blurtype == 'disk':
      input_lv = apply_disc_filter(image, rsz, is_rgb=False)
    elif blurtype == 'gaussian':
      input_lv = gaussian_filter(tf.expand_dims(input_lv, 0), blur_size)[0, ...]
    else:
      raise ValueError('Unknown blur type.')
    image_pyramid.append(input_lv)
  return image_pyramid


def lerp(a, b, x):
  """Linear interpolation between a and b using weight x."""
  return a + x * (b - a)


def render_shadow_from_mask(mask, segmentation=None):
  """Render a shadow mask by applying spatially-varying blur.

  Args:
    mask: A Tensor of shape [H, W, 1].
    segmentation: face segmentation, apply to the generated shadow mask if provided.

  Returns:
    A Tensor of shape [H, W, 1] containing the shadow mask.
  """
  mask = tf.expand_dims(mask, 2)
  disc_filter_sz = tf.random.uniform(
      shape=(), minval=1, maxval=_MAX_BLUR_SIGMA, dtype=tf.int32)
  mask_blurred = tf.cond(
      tf.greater(tf.random.uniform([]),
                 tf.constant(_SV_SIGMA)), lambda: apply_spatially_varying_blur(
                     mask,
                     blur_size=tf.random.uniform(
                         shape=(), minval=1, maxval=3, dtype=tf.int32)),
      lambda: apply_disc_filter(mask, disc_filter_sz, is_rgb=False))
  mask_blurred_norm = tf.math.divide(mask_blurred, tf.reduce_max(mask_blurred))
  if segmentation is not None:
    mask_blurred_seg = mask_blurred_norm * segmentation
  else:
    mask_blurred_seg = mask_blurred_norm
  tf.compat.v1.debugging.assert_greater_equal(
      tf.reduce_sum(mask_blurred_seg),
      0.1,
      message='Rendered silhouette mask values too small.')  # sample drops if this happens
  return mask_blurred_norm


def render_perlin_mask(size, segmentation=None):
  """Render a shadow mask using perlin noise pattern.

  Args:
    size: A 2D tensor of target mask size.
    segmentation: face segmentation, apply to the generated shadow mask if provided.

  Returns:
    A Tensor of shape [H, W, 1] containing the shadow mask.
  """
  with tf.name_scope('render_perlin'):
    size = tf.cast(size, tf.int32)
    perlin_map = perlin_collection((size[0], size[1]), [4, 4], 4,
                                   tf.random.uniform([], 0.05, 0.85))
    perlin_map_thre = tf.cast(tf.greater(perlin_map, 0.15), tf.float32)
    perlin_shadow_map = render_shadow_from_mask(
        perlin_map_thre, segmentation=segmentation)
  return perlin_shadow_map


def render_silhouette_mask(silhouette, size, segmentation=None):
  """Render a shadow mask using silhouette image.

  The sihouette image is first augmented by applying random rotation and tiling.
  Then used to render a shadow mask by applying spatially-varying blur.
  Args:
    silhouette: Rotation matrices of shape [H, W, 1].
    size: A 2D tensor of target mask size.
    segmentation: face segmentation, apply to the generated shadow mask if provided.

  Returns:
    A Tensor of shape [H, W, 1] containing the shadow mask.
  """
  with tf.name_scope('render_silhouette'):
    silhouette.shape.assert_has_rank(3)
    tf.compat.v1.assert_equal(silhouette.shape[2], 1)
    degree = tf.random.uniform(shape=(), minval=0, maxval=360, dtype=tf.float32)
    silhouette_rot = tf.contrib.image.rotate(
        silhouette, degree * np.pi / 180., interpolation='BILINEAR')
    rand_rz_ratio = tf.random.uniform(
        shape=(), minval=0.3, maxval=0.6, dtype=tf.float32)
    silhouette_rsz = resize_image(silhouette_rot, rsz=rand_rz_ratio)
    num_rep_h = tf.math.floordiv(
        tf.cast(size[0], tf.float32),
        tf.cast(tf.shape(silhouette_rsz)[0], tf.float32)) + 2
    num_rep_h = tf.cast(num_rep_h, tf.int32)
    num_rep_w = tf.math.floordiv(
        tf.cast(size[1], tf.float32),
        tf.cast(tf.shape(silhouette_rsz)[1], tf.float32)) + 2
    num_rep_w = tf.cast(num_rep_w, tf.int32)
    silhouette_solid_tile = tf.tile(silhouette_rsz, [num_rep_h, num_rep_w, 1])
    silhouette_solid_tile = silhouette_solid_tile[:size[0], :size[1], 0]
    silhouette_solid_mask = render_shadow_from_mask(
        silhouette_solid_tile, segmentation=segmentation)
  return silhouette_solid_mask



"""
  Color jitter
"""
def apply_tone_curve(image, gain=(0.5, 0.5, 0.5), is_rgb=False):
  """Apply tone perturbation to images.

  Tone curve jitter comes from Schlick's bias and gain.
  Schlick, Christophe. "Fast alternatives to Perlin’s bias and gain functions." Graphics Gems IV 4 (1994).
  Args:
    image: a 3D image tensor [H, W, C].
    gain: a tuple of length 3 that specifies the strength of the jitter per color channel.
    is_rgb: a bool that indicates whether input is grayscale (C=1) or rgb (C=3).
  
  Returns:
    3D tensor applied with a tone curve jitter, has the same size as input.
  """
  image_max = tf.reduce_max(image)
  image /= image_max
  if not is_rgb:
    mask = tf.cast(tf.greater_equal(image, 0.5), image.dtype)
    image = getbias(image * 2.0, gain[0]) / 2.0 * (1.0 - mask) + (
        getbias(image * 2.0 - 1.0, 1.0 - gain[0]) / 2.0 + 0.5) * mask
  else:
    image_r = image[..., 0, tf.newaxis]
    image_r_mask = tf.cast(tf.greater_equal(image_r, 0.5), image.dtype)
    image_r = getbias(image_r * 2.0, gain[0]) / 2.0 * (1.0 - image_r_mask) + (
        getbias(image_r * 2.0 - 1.0, 1.0 - gain[0]) / 2.0 + 0.5) * image_r_mask

    image_g = image[..., 1, tf.newaxis]
    image_g_mask = tf.cast(tf.greater_equal(image_r, 0.5), image.dtype)
    image_g = getbias(image_g * 2.0, gain[1]) / 2.0 * (1.0 - image_g_mask) + (
        getbias(image_g * 2.0 - 1.0, 1.0 - gain[1]) / 2.0 + 0.5) * image_g_mask

    image_b = image[..., 2, tf.newaxis]
    image_b_mask = tf.cast(tf.greater_equal(image_r, 0.5), image.dtype)
    image_b = getbias(image_b * 2.0, gain[2]) / 2.0 * (1.0 - image_b_mask) + (
        getbias(image_b * 2.0 - 1.0, 1.0 - gain[2]) / 2.0 + 0.5) * image_b_mask

    image = tf.concat([image_r, image_g, image_b], 2)
  return image * image_max


def getbias(x, bias):
  """Bias in Ken Perlin’s bias and gain functions."""
  return x / ((1.0 / bias - 2.0) * (1.0 - x) + 1.0 + 1e-6)


def get_ctm_ls(image, target):
  """Use least square to obtain color transfer matrix.

  Args:
    image: the source tensor of shape [H, W, 3].
    target: target tensor with the same shape as input.
  
  Returns:
    tensor of size 3 by 3 that minimizes |C x image - target|_2.
  """
  image = tf.reshape(image, [-1, 3])
  target = tf.reshape(target, [-1, 3])
  ctm = tf.linalg.lstsq(image, target, l2_regularizer=0.0, fast=True)
  return tf.transpose(ctm)


def apply_ctm(image, ctm):
  """Apply a color transfer matrix.

  Args:
    image: a tensor that contains the source image of shape [H, W, 3].
    ctm: a tensor that contains a 3 by 3 color matrix.
  Returns:
    a tensor of the same shape as image.
  """
  shape = tf.shape(image)
  image = tf.reshape(image, [-1, 3])
  image = tf.tensordot(image, ctm, axes=[[-1], [-1]])
  return tf.reshape(image, shape)


def apply_geometric_augmentation(image):
  """Randomly apply geometric augmentation."""
  processed_images = tf.image.random_flip_left_right(image)
  return processed_images
