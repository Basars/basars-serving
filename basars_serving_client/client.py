import os
import cv2
import grpc
import logging
import numpy as np

import tensorflow as tf

from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
from basars_serving_client.postprocessing import save_as_readable_image


def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img


def run():
    logging.basicConfig(level=logging.INFO)

    grpc_host = os.getenv('BASARS_HOST', 'localhost')
    grpc_port = os.getenv('BASARS_PORT', 9000)

    source_images_dir = os.getenv('BASARS_IMAGE_SOURCE_DIR', 'sample_images')
    target_images_dir = os.getenv('BASARS_IMAGE_TARGET_DIR', 'target_images')

    logging.info('Joining in to the Basars serving gRPC server...')
    address = '{}:{}'.format(grpc_host, grpc_port)
    with grpc.insecure_channel(address) as channel:
        logging.info('Successfully connected to the gRPC server: {}'.format(address))
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        for filename in os.listdir(source_images_dir):
            filepath = os.path.join(source_images_dir, filename)
            original_image = load_image(filepath)

            image = original_image[tf.newaxis, ...]
            image = tf.cast(image / 255., dtype=tf.float32)
            image = tf.make_tensor_proto(image, dtype=tf.float32)

            pred_req = predict_pb2.PredictRequest()
            pred_req.model_spec.name = 'basars_stairs'
            pred_req.model_spec.signature_name = 'serving_default'
            pred_req.inputs['VisionTransformer_input'].CopyFrom(image)

            pred_response = stub.Predict(pred_req)
            logging.info('The gRPC server have responded.')

            outputs = np.array(pred_response.outputs['conv2d'].float_val)
            outputs = np.reshape(outputs, (224, 224, 5))

            phase_images = [outputs[:, :, axis:axis + 1] for axis in range(outputs.shape[-1])]

            name_only = filename.split('.')[0]
            if not os.path.exists(target_images_dir):
                os.makedirs(target_images_dir, exist_ok=True)

            dst_filepath = '{}/analysis_{}.jpg'.format(target_images_dir, name_only)
            save_as_readable_image(original_image, phase_images, dst_filepath)
            logging.info('The readable image have been saved at: {}'.format(dst_filepath))


if __name__ == '__main__':
    run()
