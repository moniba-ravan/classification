from .resnet50 import resnet50_args

ARGUMENTS = dict(resnet50=resnet50_args,
                 # other args
                 )


def load_model(model_name, image_size, n_classes, fine_tune):
    """Get models"""
    return MODELS[model_name](image_size=image_size,
                              n_classes=n_classes,
                              fine_tune=fine_tune
                              ).get_model()