from model_resnet50 import Resnet50

MODELS = dict(resnet50=Resnet50,
              # other models
              )


def load_model(model_name, image_size, n_classes):
    """Get models"""
    return MODELS[model_name](image_size=image_size,
                              n_classes=n_classes
                              ).get_model()
