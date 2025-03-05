import os

import tensorflow as tf
print("CUDA_HOME:", os.environ["CUDA_HOME"])
print(tf.config.list_physical_devices('GPU'))
