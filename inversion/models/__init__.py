from . import densenet, mobilenet_v2, resnetrs

import flax.linen as nn


def load_model(name: str, classes: int = 10) -> nn.Module:
    """Load one of the models by name"""
    match name:
        case "densenet": return densenet.DenseNet121(classes=classes)
        case "mobilenet": return mobilenet_v2.MobileNetV2(classes=classes)
        case "resnet": return resnetrs.ResNetRS50(classes=classes)
        case _: raise NotImplementedError(f"Model {name} has not been implemented.")
