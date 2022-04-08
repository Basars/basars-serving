import os
import logging

import tensorflow as tf

from basars_serving.models import create_stairs_vision_transformer


logging.basicConfig(level=logging.INFO)

saved_model_dir = os.getenv('BASARS_SAVED_MODEL_DIR', '/models/basars_stairs/1')

model = create_stairs_vision_transformer()
logging.info('Saving the Basars model...')
tf.saved_model.save(model, saved_model_dir)
logging.info('Basars model have been saved.')
