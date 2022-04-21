FROM tensorflow/serving

ENV ROOT_DIR /basars_serving
ENV MODEL_NAME basars_stairs
ENV BASARS_SAVED_MODEL_DIR /models/$MODEL_NAME/1
ENV BASARS_SAMPLE_IMAGES_DIR $ROOT_DIR/sample_images

RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive \
    apt install python3 python3-pip python3-opencv git -y
RUN mkdir $ROOT_DIR

COPY basars_serving/ $ROOT_DIR/basars_serving
COPY sample_images/ $BASARS_SAMPLE_IMAGES_DIR

WORKDIR $ROOT_DIR

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install git+https://github.com/Basars/trans-unet.git
RUN python3 -m pip install progressbar tensorflow-serving-api opencv-python
RUN python3 -m basars_serving.create_saved_model
RUN python3 -m basars_serving.create_warmup_requests

EXPOSE 8500