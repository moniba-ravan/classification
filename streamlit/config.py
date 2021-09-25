MODELS_TO_ADDR = {
    "mobilenet": 'weights/mobilenet_trained_on_augmented_dataset.h5',
    "resnet50": 'weights/resnet50_2021-09-25_00.29.53.915829.h5',

}

MODELS_TO_ARGS = {
    "mobilenet": {
        'input_shape': (200, 200, 3),
        'compile': True
    },
    "resnet50": {
        'input_shape': (200, 200, 3)
    },
}
