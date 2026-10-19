import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
KAGGLE_DATASET = "obulisainaren/multi-cancer"
DEFAULT_DATA_DIR = os.environ.get("KAGGLE_DATA_DIR", os.path.join("data", "multi-cancer"))

CLASS_NAMES = [
    "Acute Lymphoblastic Leukemia",
    "Brain Cancer",
    "Breast Cancer",
    "Cervical Cancer",
    "Kidney Cancer",
    "Lung and Colon Cancer",
    "Lymphoma",
    "Oral Cancer",
]

CLASS_FOLDER_NAMES = {
    "all": CLASS_NAMES[0],
    "acute lymphoblastic leukemia": CLASS_NAMES[0],
    "brain cancer": CLASS_NAMES[1],
    "breast cancer": CLASS_NAMES[2],
    "cervical cancer": CLASS_NAMES[3],
    "kidney cancer": CLASS_NAMES[4],
    "lung and colon cancer": CLASS_NAMES[5],
    "lung cancer": CLASS_NAMES[5],
    "colon cancer": CLASS_NAMES[5],
    "lymphoma": CLASS_NAMES[6],
    "oral cancer": CLASS_NAMES[7],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}



def get_dataframe_from_directory(data_dir):
    records = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return pd.DataFrame(columns=["filepaths", "labels"])

    for image_path in data_path.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = next(
            (
                CLASS_FOLDER_NAMES[part.lower()]
                for part in image_path.relative_to(data_path).parts[:-1]
                if part.lower() in CLASS_FOLDER_NAMES
            ),
            None,
        )
        if label is not None:
            records.append({"filepaths": str(image_path), "labels": label})

    return pd.DataFrame(records, columns=["filepaths", "labels"])


def _download_kaggle_dataset(data_dir):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The Kaggle package is required to download the dataset. "
            "Install it with `python -m pip install kaggle`."
        ) from exc

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(data_dir), unzip=True)


def preprocess_data(data_dir=DEFAULT_DATA_DIR, random_state=42):
    dataframe = get_dataframe_from_directory(data_dir)
    if dataframe.empty:
        print(f"Downloading Kaggle dataset {KAGGLE_DATASET} into {data_dir}...")
        _download_kaggle_dataset(data_dir)
        dataframe = get_dataframe_from_directory(data_dir)

    if dataframe.empty:
        raise ValueError(
            f"No supported images were found under {data_dir}. "
            "Expected image files inside cancer-type folders from the Kaggle dataset."
        )

    found_classes = set(dataframe["labels"])
    missing_classes = set(CLASS_NAMES) - found_classes
    if missing_classes:
        raise ValueError(
            "The downloaded dataset is missing expected cancer-type folders: "
            + ", ".join(sorted(missing_classes))
        )

    train_df, remaining_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=random_state,
        stratify=dataframe["labels"],
    )
    validation_df, test_df = train_test_split(
        remaining_df,
        test_size=0.5,
        random_state=random_state,
        stratify=remaining_df["labels"],
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    evaluation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator_options = {
        "x_col": "filepaths",
        "y_col": "labels",
        "target_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "class_mode": "categorical",
        "color_mode": "grayscale",
        "classes": CLASS_NAMES,
    }

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        **generator_options,
    )
    validation_generator = evaluation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        shuffle=False,
        **generator_options,
    )
    test_generator = evaluation_datagen.flow_from_dataframe(
        dataframe=test_df,
        shuffle=False,
        **generator_options,
    )

    return train_generator, validation_generator, test_generator, len(CLASS_NAMES)


if __name__ == "__main__":
    train_data, val_data, test_data, num_classes = preprocess_data()
    print("Data preprocessing complete.")
    print(f"Number of training samples: {train_data.samples}")
    print(f"Number of validation samples: {val_data.samples}")
    print(f"Number of test samples: {test_data.samples}")
    print(f"Class names: {train_data.class_indices}")
    print(f"Number of classes: {num_classes}")
