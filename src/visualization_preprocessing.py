"""
Paso 2: Visualización y preprocesamiento.

Incluye una función para visualizar cuadrículas 3x3 de imágenes por clase y
la creación de generadores de datos (`ImageDataGenerator` + `flow_from_directory`)
pensados para entornos con RAM limitada (<12 GB), ya que cargan las imágenes
en lotes directamente desde disco en lugar de en memoria.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def plot_sample_grid(folder, title, n_images=9):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"El directorio '{folder}' no existe.")

    image_paths = list(folder.glob("*.jpg"))[:n_images]
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en '{folder}'.")

    cols = int(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(9, 9))
    fig.suptitle(title)
    for ax, path in zip(axes.flat, image_paths):
        img = load_img(path, target_size=IMG_SIZE)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    plot_sample_grid(dataset_dir / "train" / "dogs", "Perros (muestra)")
    plot_sample_grid(dataset_dir / "train" / "cats", "Gatos (muestra)")


def create_data_generators(dataset_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE):
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    for d in (train_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(f"El directorio '{d}' no existe.")

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    trdata = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    tsdata = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return trdata, tsdata
