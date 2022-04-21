import os
import cv2
import logging

import tensorflow as tf

from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.prediction_log_pb2 import PredictionLog, PredictLog


logging.basicConfig(level=logging.INFO)

WARMUP_REQUEST_FILENAME = 'tf_serving_warmup_requests'

saved_model_dir = os.getenv('BASARS_SAVED_MODEL_DIR', '/models/basars_stairs/1')
sample_images_dir = os.getenv('BASARS_SAMPLE_IMAGES_DIR', '/basars_serving/sample_images')


warmup_request_dir = os.path.join(saved_model_dir, 'assets.extra')
if not os.path.exists(warmup_request_dir):
    os.makedirs(warmup_request_dir, exist_ok=True)
warmup_request_path = os.path.join(warmup_request_dir, WARMUP_REQUEST_FILENAME)


def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img


def create_prediction_requests(img_dir):
    requests = []
    for filename in os.listdir(img_dir):
        filepath = os.path.join(img_dir, filename)

        image = load_image(filepath)
        image = image[tf.newaxis, ...]
        image = tf.cast(image / 255., dtype=tf.float32)
        image = tf.make_tensor_proto(image, dtype=tf.float32)

        pred_req = PredictRequest()
        pred_req.model_spec.name = 'basars_stairs'
        pred_req.model_spec.signature_name = 'serving_default'
        pred_req.inputs['VisionTransformer_input'].CopyFrom(image)
        requests.append(pred_req)
    return requests


logging.info('Generating warmup requests at {}'.format(warmup_request_path))
with tf.io.TFRecordWriter(warmup_request_path) as writer:
    pred_requests = create_prediction_requests(sample_images_dir)
    for request in pred_requests:
        log = PredictionLog(predict_log=PredictLog(request=request))
        writer.write(log.SerializeToString())
logging.info('Job has finished')
