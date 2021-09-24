MODELS_TO_ADDR = {
    "mobilenet": 'weights/mobilenet_2021-09-24_19.12.41.037223.h5',
    "resnet50": 'weights/resnet50_2021-09-25_00.29.53.915829.h5',

}

MODELS_TO_ARGS = {
    "mobilenet": {
        'input_shape': (200, 200, 3)
    },
    "resnet50": {
        'input_shape': (200, 200, 3)
    },
}
