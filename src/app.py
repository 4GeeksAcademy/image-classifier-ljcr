"""
Script de extremo a extremo: clasificación de imágenes Perros vs. Gatos
con EfficientNet-B0.
"""

from pathlib import Path

from keras.models import load_model

from dataset_preparation import prepare_dataset
from visualization_preprocessing import create_data_generators, visualize_dataset
from model_builder import build_model
from train import evaluate_model, train_model

RAW_TRAIN_DIR = Path("../data/raw/train")
DATASET_DIR = Path("../data/processed/dataset")
CHECKPOINT_PATH = Path("../models/checkpoint_best.keras")
FINAL_MODEL_PATH = Path("../models/dogs_vs_cats_efficientnet.keras")
EPOCHS = 20


def main():
    # Paso 1: preparar el dataset (dog.N.jpg / cat.N.jpg -> dataset/train|test/dogs|cats)
    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        prepare_dataset(RAW_TRAIN_DIR, DATASET_DIR, train_split=0.8)

    # Paso 2: visualización de muestras y generadores de datos
    visualize_dataset(DATASET_DIR)
    trdata, tsdata = create_data_generators(DATASET_DIR)

    # Paso 3: construcción y compilación del modelo
    model = build_model()
    model.summary()

    # Paso 4: entrenamiento con callbacks y evaluación del mejor modelo
    train_model(model, trdata, tsdata, CHECKPOINT_PATH, epochs=EPOCHS)
    best_model = load_model(CHECKPOINT_PATH)
    evaluate_model(best_model, tsdata)

    # Paso 5: guardado del modelo final
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_model.save(FINAL_MODEL_PATH)
    print(f"Modelo guardado en {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
