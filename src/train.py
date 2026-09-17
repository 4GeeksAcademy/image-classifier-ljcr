"""
Paso 4: Optimización, callbacks y entrenamiento.
"""

from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


def get_callbacks(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
    )

    return [early_stopping, model_checkpoint]


def train_model(model, trdata, tsdata, checkpoint_path, epochs=20):
    callbacks = get_callbacks(checkpoint_path)

    history = model.fit(
        trdata,
        validation_data=tsdata,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history


def evaluate_model(model, tsdata):
    tsdata.reset()
    y_true = tsdata.classes
    class_labels = list(tsdata.class_indices.keys())

    predictions = model.predict(tsdata, steps=len(tsdata))
    y_pred = np.argmax(predictions, axis=1)

    print(classification_report(y_true, y_pred, target_names=class_labels))
