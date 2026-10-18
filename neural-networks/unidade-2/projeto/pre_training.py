"""
Dicionário direto com os modelos pré-treinados do tf.keras.applications
pronto para iterar sobre diferentes backbones com imagens redimensionadas.
"""

from tensorflow.keras import applications

pre_trained_models = {
    # ConvNeXt
    "ConvNeXtTiny": applications.ConvNeXtTiny,
    "ConvNeXtSmall": applications.ConvNeXtSmall,
    "ConvNeXtBase": applications.ConvNeXtBase,

    # DenseNet
    "DenseNet121": applications.DenseNet121,
    "DenseNet169": applications.DenseNet169,
    "DenseNet201": applications.DenseNet201,

    # EfficientNet
    "EfficientNetB0": applications.EfficientNetB0,
    "EfficientNetB4": applications.EfficientNetB4,
    "EfficientNetB7": applications.EfficientNetB7,

    # EfficientNetV2
    "EfficientNetV2B0": applications.EfficientNetV2B0,
    "EfficientNetV2S": applications.EfficientNetV2S,
    "EfficientNetV2M": applications.EfficientNetV2M,

    # Inception
    "InceptionV3": applications.InceptionV3,
    "InceptionResNetV2": applications.InceptionResNetV2,

    # MobileNet
    "MobileNet": applications.MobileNet,
    "MobileNetV2": applications.MobileNetV2,
    "MobileNetV3Small": applications.MobileNetV3Small,
    "MobileNetV3Large": applications.MobileNetV3Large,

    # NASNet
    "NASNetMobile": applications.NASNetMobile,
    "NASNetLarge": applications.NASNetLarge,

    # ResNet
    "ResNet50": applications.ResNet50,
    "ResNet101": applications.ResNet101,
    "ResNet152": applications.ResNet152,

    # ResNetV2
    "ResNet50V2": applications.ResNet50V2,
    "ResNet101V2": applications.ResNet101V2,
    "ResNet152V2": applications.ResNet152V2,

    # VGG
    "VGG16": applications.VGG16,
    "VGG19": applications.VGG19,

    # Xception
    "Xception": applications.Xception
}

if __name__ == "__main__":
    print(f"Total de modelos pré-treinados prontos para iterar: {len(pre_trained_models)}")
    for name in pre_trained_models:
        print(f" - {name}")