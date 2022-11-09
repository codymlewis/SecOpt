from . import densenet, mobilenet_v2, inception_v3

import flax.linen as nn


def load_model(name: str, classes: int = 10) -> nn.Module:
    """Load one of the models by name"""
    match name:
        case "densenet": return densenet.DenseNet121(classes=classes)
        case "mobilenet": return mobilenet_v2.MobileNetV2(classes=classes)
        case "inception": return inception_v3.InceptionV3(classes=classes)
        case _: raise NotImplementedError(f"Model {name} has not been implemented.")
