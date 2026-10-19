"""
Mapeamento de arquiteturas pré-treinadas do tf.keras.applications
organizado por família (1 versão representativa por família) para a Etapa 3.
"""

from tensorflow.keras import applications

# 1 versão representativa por família (Triagem Arquitetural)
representative_models_per_family = {
    "ConvNeXt": applications.ConvNeXtTiny,
    "DenseNet": applications.DenseNet121,          # Já executado na PoC
    "EfficientNetV2": applications.EfficientNetV2B0, # V1 descartada
    "Inception": applications.InceptionV3,         # Descartada (superada por Xception)
    "MobileNet": applications.MobileNetV2,         # Descartada (foco em edge, menor capacidade)
    "NASNet": applications.NASNetMobile,           # Descartada (lenta em GPU / RL irregular)
    "ResNet": applications.ResNet50V2,             # Selecionada (pré-ativação V2)
    "VGG": applications.VGG16,                     # Descartada (obsoleta, sem skip, 138M params)
    "Xception": applications.Xception              # Reserva técnica
}

# Modelos selecionados para treinamento no Benchmarking (Etapa 3)
# Nota: DenseNet121 já foi treinada e seus pesos estão salvos em models/
benchmark_selected_models = {
    "EfficientNetV2B0": applications.EfficientNetV2B0,
    "ResNet50V2": applications.ResNet50V2,
    "ConvNeXtTiny": applications.ConvNeXtTiny,
    "Xception": applications.Xception,
}

if __name__ == "__main__":
    print(f"Total de famílias candidatas: {len(representative_models_per_family)}")
    print(f"Modelos selecionados para novo treino na Etapa 3: {list(benchmark_selected_models.keys())}")
    print("Nota: DenseNet121 já possui checkpoint e métricas na PoC para compor o Ensemble.")