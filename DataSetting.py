
import tensorflow as tf
import convert_tf_record
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    'RAF',
    'A name of the dataset')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    './raw_data/mnist',
    'The save path of the tensor record file')
tf.app.flags.DEFINE_integer(
    'num_shards',
    5,
    'A number of sharding for TFRecord files(integer).')
tf.app.flags.DEFINE_integer(
    'image_width',
    224,
    'A number of width to resize(integer).')
tf.app.flags.DEFINE_integer(
    'image_height',
    224,
    'A number of height to resize(integer).')
tf.app.flags.DEFINE_float(
    'ratio_val',
    0.1,
    'A ratio of validation datasets for TFRecord files(flaot, 0 ~ 1).')
tf.app.flags.DEFINE_string(
    'label_file',
    './label.txt',
    'The path of the label file.')
def main(_):
  if not FLAGS.label_file:
    raise ValueError('You must supply the label_file path with --label_file')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  convert_tf_record.run(FLAGS.dataset_name, FLAGS.dataset_dir, FLAGS.num_shards, FLAGS.ratio_val, FLAGS.label_file, FLAGS.image_width, FLAGS.image_height)
if __name__ == '__main__':
  tf.app.run()