
import random
import os
import numpy as np
import tensorflow as tf

def seed_everything(seed=21,tf_version=2):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if(tf_version==1):
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)


        