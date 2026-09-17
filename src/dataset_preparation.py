"""
Paso 1: Preparación del conjunto de datos.

Reorganiza un directorio plano de imágenes mezcladas (dog.0.jpg, cat.0.jpg, ...)
en la estructura de carpetas que espera `flow_from_directory()` de Keras:

    dataset/train/dogs
    dataset/train/cats
    dataset/test/dogs
    dataset/test/cats
"""

import random
import shutil
import zipfile
from pathlib import Path


def extract_dataset(zip_path, extract_to):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo zip: {zip_path}")

    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to


def prepare_dataset(source_dir, dataset_dir, train_split=0.8, seed=42):
    source_dir = Path(source_dir)
    dataset_dir = Path(dataset_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"El directorio origen '{source_dir}' no existe. "
            "Verifica que las imágenes estén descomprimidas."
        )

    splits = ("train", "test")
    classes = ("dogs", "cats")
    for split in splits:
        for cls in classes:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    dog_files = sorted(source_dir.glob("dog.*.jpg"))
    cat_files = sorted(source_dir.glob("cat.*.jpg"))

    if not dog_files or not cat_files:
        raise FileNotFoundError(
            f"No se encontraron imágenes 'dog.*.jpg' / 'cat.*.jpg' en {source_dir}"
        )

    random.seed(seed)
    random.shuffle(dog_files)
    random.shuffle(cat_files)

    def split_and_copy(files, class_name):
        cut = int(len(files) * train_split)
        train_files, test_files = files[:cut], files[cut:]
        for f in train_files:
            shutil.copy2(f, dataset_dir / "train" / class_name / f.name)
        for f in test_files:
            shutil.copy2(f, dataset_dir / "test" / class_name / f.name)

    split_and_copy(dog_files, "dogs")
    split_and_copy(cat_files, "cats")

    return dataset_dir
