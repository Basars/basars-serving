# Basars — TensorFlow Serving

TensorFlow Serving implementation for Basars models

### Installing dependencies
```bash
pip install tensorflow-serving-api matplotlib opencv-python
```

### Building from Dockerfile
```bash
git clone https://github.com/Basars/basars-serving.git
cd basars-serving
docker build -t basars/serving:1.0 .
```

### Running from built Docker image
```bash
docker run -d -p 8500:8500 --name basars-serving basars/serving:1.0
```

### Running client
```bash
python -m basars_serving_client.client
```

### Environmental Variables
| Key                       | Default Value |
|---------------------------|---------------|
| `BASARS_HOST`             | localhost     |
| `BASARS_PORT`             | 8500          |
| `BASARS_IMAGE_SOURCE_DIR` | sample_images |
| `BASARS_IMAGE_TARGET_DIR` | target_images |
