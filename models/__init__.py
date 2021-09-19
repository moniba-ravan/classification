from .resnet50 import Resnet50

MODELS = dict(resnet50=Resnet50,
              # other models
              )


def load_model(model_name, image_size):
    """Get models"""
    return MODELS[model_name](image_size).get_model()
