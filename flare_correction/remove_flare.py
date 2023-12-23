import os.path
from typing import Optional

import numpy as np

from absl import app
from absl import flags
import tensorflow as tf
import tqdm

from . import models
from . import utils

FLAGS = flags.FLAGS

_DEFAULT_CKPT = None
flags.DEFINE_string(
    'ckpt', _DEFAULT_CKPT,
    'Location of the model checkpoint. May be a SavedModel dir, in which case '
    'the model architecture & weights are both loaded, and "--model" is '
    'ignored. May also be a TF checkpoint path, in which case only the latest '
    'model weights are loaded (this is much faster), and "--model" is '
    'required. To load a specific checkpoint, use the checkpoint prefix '
    'instead of the checkpoint directory for this argument.')
flags.DEFINE_string(
    'model', None,
    'Only required when "--ckpt" points to a TF checkpoint or checkpoint dir. '
    'Must be one of "Uformer", "unet" or "can".')
flags.DEFINE_integer(
    'batch_size', 1,
    'Number of images in each batch. Some networks (e.g., the rain removal '
    'network) can only accept predefined batch sizes.')
flags.DEFINE_string('input_dir', None,
                    'The directory contains all input images.')
flags.DEFINE_string('out_dir', None, 'Output directory.')
flags.DEFINE_boolean(
    'separate_out_dirs', True,
    'Whether the results are saved in separate folders under different names '
    '(True), or the same folder under different names (False).')


def center_crop(image, width, height):
  """Returns the center crop of a given image."""
  old_height, old_width, _ = image.shape
  x_offset = (old_width - width) // 2
  y_offset = (old_height - height) // 2
  if x_offset < 0 or y_offset < 0:
    raise ValueError('The specified output size is bigger than the image size.')
  return image[y_offset:(y_offset + height), x_offset:(x_offset + width), :]


def write_outputs_same_dir(out_dir,
                           name_prefix,
                           input_image = None,
                           pred_scene = None,
                           pred_flare = None,
                           pred_blend = None):
  """Writes various outputs to the same directory on disk."""
  if not tf.io.gfile.isdir(out_dir):
    raise ValueError(f'{out_dir} is not a directory.')
  path_prefix = os.path.join(out_dir, name_prefix)
  if input_image is not None:
    utils.write_image(input_image, path_prefix + '_input.png')
  if pred_scene is not None:
    utils.write_image(pred_scene, path_prefix + '_output.png')
  if pred_flare is not None:
    utils.write_image(pred_flare, path_prefix + '_output_flare.png')
  if pred_blend is not None:
    utils.write_image(pred_blend, path_prefix + '_output_blend.png')


def write_outputs_separate_dir(out_dir,
                               file_name,
                               input_image = None,
                               pred_scene = None,
                               pred_flare = None,
                               pred_blend = None):
  """Writes various outputs to separate subdirectories on disk."""
  if not tf.io.gfile.isdir(out_dir):
    raise ValueError(f'{out_dir} is not a directory.')
  if input_image is not None:
    utils.write_image(input_image, os.path.join(out_dir, 'input', file_name))
  if pred_scene is not None:
    utils.write_image(pred_scene, os.path.join(out_dir, 'output', file_name))
  if pred_flare is not None:
    utils.write_image(pred_flare,
                      os.path.join(out_dir, 'output_flare', file_name))
  if pred_blend is not None:
    utils.write_image(pred_blend,
                      os.path.join(out_dir, 'output_blend', file_name))


def process_one_image(model, image_path, out_dir, separate_out_dirs):
  """Reads one image and writes inference results to disk."""
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    blob = f.read()
  input_u8 = tf.image.decode_image(blob)[Ellipsis, :3]
  input_f32 = tf.image.convert_image_dtype(input_u8, tf.float32, saturate=True)
  h, w, _ = input_f32.shape

  GAMMA=2.2
  GAMMA_LOW=2.2

  if min(h, w) >= 2048:
    print("NOOOOOOOOOO!!!")
    input_image = center_crop(input_f32, 2048, 2048)[None, Ellipsis]
    input_low = tf.image.resize(
        input_image, [512, 512], method=tf.image.ResizeMethod.AREA)
    pred_scene_low = tf.clip_by_value(model(input_low), 0.0, 1.0)
    pred_flare_low = utils.remove_flare(input_low, pred_scene_low, GAMMA_LOW)
    pred_flare = tf.image.resize(pred_flare_low, [2048, 2048], antialias=True)
    pred_scene = utils.remove_flare(input_image, pred_flare, GAMMA)
  else:
    print("GOHEREEEEEE!!!")
    input_image = center_crop(input_f32, 512, 512)[None, Ellipsis]
    input_image = tf.concat([input_image] * FLAGS.batch_size, axis=0)
    pred_scene = tf.clip_by_value(model(input_image), 0.0, 1.0)
    pred_flare = utils.remove_flare(input_image, pred_scene, GAMMA)
    pred_flare = tf.clip_by_value(pred_flare, np.array(pred_flare).mean(), 1.0)
    pred_scene -= 3*pred_flare
  pred_blend = utils.blend_light_source(input_image[0, Ellipsis], pred_scene[0, Ellipsis])

  out_filename_stem = os.path.splitext(os.path.basename(image_path))[0]
  if separate_out_dirs:
    write_outputs_separate_dir(
        out_dir,
        out_filename_stem + '.png',
        input_image=input_image[0, Ellipsis],
        pred_scene=pred_scene[0, Ellipsis],
        pred_flare=pred_flare[0, Ellipsis],
        pred_blend=pred_blend)
  else:
    write_outputs_same_dir(
        out_dir,
        out_filename_stem,
        input_image=input_image[0, Ellipsis],
        pred_scene=pred_scene[0, Ellipsis],
        pred_flare=pred_flare[0, Ellipsis],
        pred_blend=pred_blend)

def process_one_image_same_size_kernel(model, input_f32):
  h, w, _ = input_f32.shape

  GAMMA=2.2

  input_image = tf.image.resize(input_f32, [512, 512])[None, Ellipsis]

  batch_size = 2
  # input_image = center_crop(input_f32, 512, 512)[None, Ellipsis]
  input_image = tf.concat([input_image]*batch_size, axis=0)
  pred_scene = tf.clip_by_value(model(input_image), 0.0, 1.0)
  pred_flare = utils.remove_flare(input_image, pred_scene, GAMMA)
  pred_flare = tf.clip_by_value(pred_flare, np.array(pred_flare).mean(), 1.0)
  pred_blend = utils.blend_light_source(input_image[0, Ellipsis], pred_scene[0, Ellipsis])

  input_image = tf.image.resize(input_image[0, Ellipsis], [h, w])
  pred_flare = tf.image.resize(pred_flare[0, Ellipsis], [h, w])
  pred_scene = tf.image.resize(pred_scene[0, Ellipsis], [h, w])
  pred_blend = tf.image.resize(pred_blend, [h, w])

  return (input_image, pred_flare, pred_scene, pred_blend)


def process_one_image_same_size(model, image_path, out_dir, separate_out_dirs):
  """Reads one image and writes inference results to disk."""
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    blob = f.read()
  input_u8 = tf.image.decode_image(blob)[Ellipsis, :3]
  input_f32 = tf.image.convert_image_dtype(input_u8, tf.float32, saturate=True)
  # h, w, _ = input_f32.shape

  # print("PARAMS: ", FLAGS.batch_size, " ", Ellipsis)

  # GAMMA=2.2

  # input_image = tf.image.resize(input_f32, [512, 512])[None, Ellipsis]

  # # input_image = center_crop(input_f32, 512, 512)[None, Ellipsis]
  # input_image = tf.concat([input_image] * FLAGS.batch_size, axis=0)
  # pred_scene = tf.clip_by_value(model(input_image), 0.0, 1.0)
  # pred_flare = utils.remove_flare(input_image, pred_scene, GAMMA)
  # pred_flare = tf.clip_by_value(pred_flare, np.array(pred_flare).mean(), 1.0)
  # pred_blend = utils.blend_light_source(input_image[0, Ellipsis], pred_scene[0, Ellipsis])

  # input_image = tf.image.resize(input_image[0, Ellipsis], [h, w])
  # pred_flare = tf.image.resize(pred_flare[0, Ellipsis], [h, w])
  # pred_scene = tf.image.resize(pred_scene[0, Ellipsis], [h, w])
  # pred_blend = tf.image.resize(pred_blend, [h, w])

  input_image, pred_flare, pred_scene, pred_blend = process_one_image_same_size_kernel(model, input_f32)

  out_filename_stem = os.path.splitext(os.path.basename(image_path))[0]
  if separate_out_dirs:
    write_outputs_separate_dir(
        out_dir,
        out_filename_stem + '.png',
        input_image=input_image,
        pred_scene=pred_scene,
        pred_flare=pred_flare,
        pred_blend=pred_blend)
  else:
    write_outputs_same_dir(
        out_dir,
        out_filename_stem,
        input_image=input_image[0, Ellipsis],
        pred_scene=pred_scene[0, Ellipsis],
        pred_flare=pred_flare[0, Ellipsis],
        pred_blend=pred_blend)


def load_model(path,
               model_type = None,
               batch_size = None):
  """Loads a model from SavedModel or standard TF checkpoint."""
  try:
    return tf.keras.models.load_model(path)
  except (ImportError, IOError):
    print(f'Didn\'t find SavedModel at "{path}". '
          'Trying latest checkpoint next.')
  model = models.build_model(model_type, batch_size)
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_path = tf.train.latest_checkpoint(path) or path
  ckpt.restore(ckpt_path).assert_existing_objects_matched()
  return model


def main(_):
  out_dir = FLAGS.out_dir or os.path.join(FLAGS.input_dir, 'model_output')
  tf.io.gfile.makedirs(out_dir)

  model = load_model(FLAGS.ckpt, FLAGS.model, FLAGS.batch_size)

  # The following grep works for both png and jpg.
  input_files = sorted(tf.io.gfile.glob(os.path.join(FLAGS.input_dir, '*.*g')))
  for input_file in tqdm.tqdm(input_files):
    # process_one_image(model, input_file, out_dir, FLAGS.separate_out_dirs)
    process_one_image_same_size(model, input_file, out_dir, FLAGS.separate_out_dirs)

  print('done')

class BlareRemoval:
  def __init__(self, model_path="flare_correction/trained_model"):
    self.model = load_model(model_path, None, 1)

  def Process(self, image):
    input_f32 = tf.image.convert_image_dtype(image, tf.float32, saturate=True)

    input_image, pred_flare, pred_scene, pred_blend = process_one_image_same_size_kernel(self.model, input_f32)

    return (input_image, pred_flare, pred_scene, pred_blend)

def apply_flare_model(image):
  ten_in = tf.convert_to_tensor(np.asarray(image))

  model = BlareRemoval()
  out_tup = model.Process(ten_in)
  out = out_tup[2]
  out = np.array(out)*255
  print(out[0, :10])
  print(out.shape)

  out = np.clip(out,0,255).astype(np.uint8)
  return out

if __name__ == '__main__':
  app.run(main)
