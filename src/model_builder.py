"""
Paso 3: Construcción de la red neuronal (EfficientNet-B0).
"""

from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam


def build_model(learning_rate=1e-4):
    model = Sequential()
    model.add(EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
