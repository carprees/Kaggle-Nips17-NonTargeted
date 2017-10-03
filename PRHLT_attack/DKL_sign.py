"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


class InceptionModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        logits = logits[:,1:]
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return [logits, probs]   


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    label = np.ones((FLAGS.batch_size,1000))/np.float32(1000.0)
    steps = 1   

    tf.logging.set_verbosity(tf.logging.INFO)

    eps = np.float32(2.0 * FLAGS.max_epsilon / 255.0)

    # x_input is used as input for the original images
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    # x_hat is used as input for the images that we want to modify
    x_hat = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)
    logits, probs = model(x_hat)

    restore_vars = [
    var for var in tf.global_variables()
        if var.name.startswith('InceptionV3/')
    ]

    y_hat = tf.placeholder(tf.float32, [None, 1000])

    # DKL
    logits_soft = tf.nn.softmax(logits, dim=-1)
    loss = y_hat * tf.log(y_hat/logits_soft)

    grad, = tf.gradients(loss, x_hat)

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x_hat - tf.sign(grad))

    below = x_input-eps
    above = x_input+eps
    adv_x = tf.clip_by_value(tf.clip_by_value(adv_x, below, above), -1, 1)    

    saver = tf.train.Saver(restore_vars) 
    sess = tf.Session()
    saver.restore(sess, FLAGS.checkpoint_path)

    for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        images = images.astype(np.float32)
        images_hat = images
  
        for i in range(0,steps):
            adv_images=sess.run(adv_x, feed_dict={x_input: images, x_hat: images_hat, y_hat: label})  
            images_hat = adv_images

        save_images(adv_images, filenames, FLAGS.output_dir)      

if __name__ == '__main__':
    tf.app.run()
