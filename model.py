import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _residual_block(inputs, filters):
    shortcut = inputs
    if inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)

    features = layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    features = layers.BatchNormalization()(features)
    features = layers.Activation("relu")(features)
    features = layers.Conv2D(filters, 3, padding="same", use_bias=False)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Add()([shortcut, features])
    return layers.Activation("relu")(features)


def ResUNetSVM(input_shape, num_classes):
    """
    Build a ResUNet feature extractor with a linear SVM-style classification head.
    Train this model with categorical hinge loss to optimize the class margins.
    """
    inputs = keras.Input(shape=input_shape)

    encoder1 = _residual_block(inputs, 32)
    encoder2 = _residual_block(layers.MaxPooling2D()(encoder1), 64)
    encoder3 = _residual_block(layers.MaxPooling2D()(encoder2), 128)
    bottleneck = _residual_block(layers.MaxPooling2D()(encoder3), 256)

    decoder3 = layers.UpSampling2D()(bottleneck)
    decoder3 = layers.Concatenate()([decoder3, encoder3])
    decoder3 = _residual_block(decoder3, 128)
    decoder2 = layers.UpSampling2D()(decoder3)
    decoder2 = layers.Concatenate()([decoder2, encoder2])
    decoder2 = _residual_block(decoder2, 64)
    decoder1 = layers.UpSampling2D()(decoder2)
    decoder1 = layers.Concatenate()([decoder1, encoder1])
    features = _residual_block(decoder1, 32)

    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.5)(features)
    decision_scores = layers.Dense(num_classes, activation="linear", name="svm_decision_scores")(features)
    return keras.Model(inputs=inputs, outputs=decision_scores, name="resunet_svm")


