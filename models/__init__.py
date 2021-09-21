from .mobile_net import MobileNet
from .resnet50 import Resnet50

MODELS = dict(resnet50=Resnet50,
              mobileNet=MobileNet
              # other models

              )


def load_model(model_name, **kwargs):
    """Get models"""
    return MODELS[model_name](**kwargs).get_model()
