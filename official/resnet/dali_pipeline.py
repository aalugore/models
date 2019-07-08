# DALI Imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec

# Generic Imports
import os
from subprocess import call
import tensorflow as tf


# Global Variables
NUM_GPUS = 1
NUM_THREADS = 1

USED_GPU = 1

_NUM_TRAIN_FILES = 1024



#TODO: I am hard coding just one tfrecord here to pass to TFRecordReader Constructor.
# it CAN however take the entire list of record/idx files
tfrecord_path = "/mnt/data/train-00001-of-01024"
tfrecord_idx_path = "/home/builder/imagenet_idx_files/train-00001-of-01024.idx"

# Utility functions
def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128' % i)
            for i in range(128)]

def get_idx_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024.idx' % i)
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128.idx' % i)
            for i in range(128)]

# Utility function used to create index files for DALI.
# Note: Index files are used by DALI mainly to properly shard the dataset between multiple workers.
#       The index for a given TFRecord file can be obtained from that file using tfrecord2idx utility
#       included with DALI. Creating the index file is required only once per TFRecord file.
def create_idx_files(filenames):

    # Name the tfrecord2idx script
    tfrecord2idx_script = "tfrecord2idx"
    # Create location to store index files
    if not os.path.exists("/home/builder/imagenet_idx_files"):
        os.mkdir("/home/builder/imagenet_idx_files")

    for tfrecord in filenames:
        tfrecord_idx = "/home/builder/imagenet_idx_files/"+(os.path.basename(tfrecord))+".idx"

        if not os.path.isfile(tfrecord_idx):
            call([tfrecord2idx_script, tfrecord, tfrecord_idx])


# Utility method for creating Index Files for the DALI pipeline
def generate_index():
    print("STARTING IDX FILE GENERATION")
    create_idx_files(get_filenames(True, "/mnt/data/"))
    create_idx_files(get_filenames(False,"/mnt/data/"))
    print("FINISHED")



# Define the pipeline
class ResnetPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, tfrecords, idx_paths):
        super(ResnetPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)


        # Transformation operations below.
        # From https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/supported_ops.html
        self.input = ops.TFRecordReader(path = tfrecords,
                                        index_path = idx_paths,
                                        features = {"image/encoded":  tfrec.FixedLenFeature([], tfrec.string, ""),
                                            "image/class/label":      tfrec.FixedLenFeature([1], tfrec.float32, 0.0),
                                            "image/class/text":       tfrec.FixedLenFeature([], tfrec.string, ""),
                                            "image/object/bbox/xmin": tfrec.VarLenFeature(tfrec.float32, 0.0),
                                            "image/object/bbox/ymin": tfrec.VarLenFeature(tfrec.float32, 0.0),
                                            "image/object/bbox/xmax": tfrec.VarLenFeature(tfrec.float32, 0.0),
                                            "image/object/bbox/ymax": tfrec.VarLenFeature(tfrec.float32, 0.0)})

        self.decode = ops.nvJPEGDecoder(device = "mixed",
                                        output_type = types.RGB)

        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR,
                                 resize_shorter = 256.)

        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                           output_dtype = types.FLOAT,
                                           crop = (224, 224),
                                           image_type = types.RGB,
                                           mean = [0., 0., 0.],
                                           std = [1., 1., 1])

        self.uniform = ops.Uniform(range = (0.0, 1.0))

        self.transpose = ops.Transpose( device = "gpu", perm=[0,3,1,2] )


        self.cast = ops.Cast(device = "gpu",
                             dtype = types.INT32)

        self.iter = 0

    def define_graph(self):

       # Get the TFRecordReader Object
        inputs = self.input()

        print("DEFINE_GRAPH")

        # Decode the images from the dictionary object
        images = self.decode(inputs["image/encoded"])

        # Resize images
        resized_images = self.resize(images)

        # Crop/Mirror/Normalize images
        output = self.cmn(resized_images,
                          crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())



        # Get label tensor
        labels = inputs["image/class/label"]

        # Image tensor already comes out on gpu where labels does not.
        # Cast labels tensor to GPU
        return (output, labels.gpu())

    # Required to be overridden by subclasses of pipeline. For our purposes it's a no-op
    def iter_setup(self):
        pass



# Data Preprocessing input function. This function will be passed directly to the estimator
# as the 'input_fn' parameter.
# Note: Required to return either a tf.Dataset Object or a tuple of Tensors.
def dali_input_fn(batch_size,
                  num_channels,
                  image_height,
                  image_width,
                  is_training):

    # Get all the file names
    # old vars: tfrecord_path, tfrecord_idx_path
    tfrecords = get_filenames(is_training, "/mnt/data/")
    idx_paths = get_idx_filenames(is_training, "/home/builder/imagenet_idx_files")


    dali_pipe = ResnetPipeline(batch_size = batch_size,
                               num_threads = NUM_THREADS,
                               device_id = USED_GPU,
                               tfrecords = tfrecords,
                               idx_paths = idx_paths)

    # Serialize Pipeline into Protobuf string
    serialized = dali_pipe.serialize()

    # Assign the iterator that will produce the (image, label) tensors
    daliop = dali_tf.DALIIterator()


    with tf.device("/gpu:1"):
        # daliop returns what we specify from define_graph() above
        # Shapes are the expected output Tensor shapes.
        # i.e.: img shape =[32,3,224,224], lbl shape=[<unknown>]
        # dtypes are the type of the 2 output tensors respectively
        img, label = daliop(serialized_pipeline = serialized,
                             shapes = [(batch_size, num_channels, image_height, image_width),()],
                             dtypes = [tf.float32, tf.int32])


    img=tf.reshape(img, shape=[batch_size,image_height,image_width,num_channels])
    label=tf.reshape(label, shape=[batch_size])
    return (img, label)


#MAIN AREA
if __name__ == '__main__':
    print("MAIN DALI_PIPELINE EXECUTED")

    # Unit testing for debugging
    batch_size = 32
    channels = 3
    img_size = 224






    image, label = dali_input_fn(batch_size, channels, img_size, img_size)
    print(image)
    print(label)
