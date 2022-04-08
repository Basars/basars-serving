FROM tensorflow/serving

ENV ROOT_DIR /basars_serving
ENV BASARS_SAVED_MODEL_DIR /models/basars_stairs/1
ENV MODEL_NAME basars_stairs

RUN apt update -y
RUN apt install python3 python3-pip git -y
RUN mkdir $ROOT_DIR

COPY basars_serving/ $ROOT_DIR/basars_serving

WORKDIR $ROOT_DIR

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install progressbar
RUN python3 -m pip install git+https://github.com/Basars/trans-unet.git
RUN python3 -m basars_serving.create_saved_model

EXPOSE 8500