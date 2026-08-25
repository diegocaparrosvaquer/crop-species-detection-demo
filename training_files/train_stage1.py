import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CropSpeciesDataset
from model import DINOv2Classifier


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = (
    "/workspace/data/train.csv"
)

VALIDATION_CSV = (
    "/workspace/data/validation.csv"
)

OUTPUT_DIR = (
    "/workspace/outputs/dinov2/"
    "hierarchical/stage1"
)

NUM_EPOCHS = 30

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 8

PATIENCE = 5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}",
    flush=True
)

if device.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}",
        flush=True
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        flush=True
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\nLoading datasets...",
    flush=True
)

train_df = pd.read_csv(
    TRAIN_CSV
)

validation_df = pd.read_csv(
    VALIDATION_CSV
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "image_path",
    "label",
]

for column in REQUIRED_COLUMNS:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing required column '{column}' "
            f"in training CSV.\n"
            f"Available columns: {train_df.columns.tolist()}"
        )

    if column not in validation_df.columns:

        raise ValueError(
            f"Missing required column '{column}' "
            f"in validation CSV.\n"
            f"Available columns: {validation_df.columns.tolist()}"
        )


# ============================================================
# DATASET OVERVIEW
# ============================================================

print(
    "\nDataset columns:",
    flush=True
)

print(
    train_df.columns.tolist(),
    flush=True
)

print(
    f"\nTraining images: "
    f"{len(train_df):,}",
    flush=True
)

print(
    f"Validation images: "
    f"{len(validation_df):,}",
    flush=True
)


if "source" in train_df.columns:

    print(
        "\nTraining source distribution:",
        flush=True
    )

    print(
        train_df["source"].value_counts(),
        flush=True
    )


if "source" in validation_df.columns:

    print(
        "\nValidation source distribution:",
        flush=True
    )

    print(
        validation_df["source"].value_counts(),
        flush=True
    )


# ============================================================
# STAGE 1 LABELS
# ============================================================
#
# Stage 1:
#
#     no cropland
#            vs
#        cropland
#
# All crop classes become "cropland".
#
# Final crop classes in the combined dataset:
#
#     banana
#     maize
#     millets
#     no cropland
#     rapeseed
#     soya
#     sorghum
#     sunflower
#     vineyard
#     wheat type crop
#
# ============================================================

CROP_CLASSES = [
    "banana",
    "maize",
    "millets",
    "rapeseed",
    "soya",
    "sorghum",
    "sunflower",
    "vineyard",
    "wheat type crop",
]

VALID_CLASSES = [
    "no cropland"
] + CROP_CLASSES


# ============================================================
# FILTER VALID CLASSES
# ============================================================

train_df = train_df[
    train_df["label"].isin(VALID_CLASSES)
].copy()

validation_df = validation_df[
    validation_df["label"].isin(VALID_CLASSES)
].copy()


print(
    "\nAfter class filtering:",
    flush=True
)

print(
    f"Training images: "
    f"{len(train_df):,}",
    flush=True
)

print(
    f"Validation images: "
    f"{len(validation_df):,}",
    flush=True
)


# ============================================================
# CONVERT TO STAGE 1 LABEL
# ============================================================

def convert_to_stage1_label(class_name):

    if class_name == "no cropland":

        return "no cropland"

    return "cropland"


train_df["Stage1Name"] = train_df[
    "label"
].apply(
    convert_to_stage1_label
)

validation_df["Stage1Name"] = validation_df[
    "label"
].apply(
    convert_to_stage1_label
)


# ============================================================
# CLASS MAPPING
# ============================================================

classes = [
    "no cropland",
    "cropland",
]

class_to_idx = {
    class_name: index
    for index, class_name in enumerate(classes)
}

idx_to_class = {
    index: class_name
    for class_name, index in class_to_idx.items()
}


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

with open(
    f"{OUTPUT_DIR}/class_mapping.json",
    "w",
) as file:

    json.dump(
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "classes": classes,
        },
        file,
        indent=4,
    )


print(
    "\nStage 1 classes:",
    flush=True
)

for class_name in classes:

    print(
        f"  {class_name}",
        flush=True
    )


# ============================================================
# STAGE 1 DISTRIBUTION
# ============================================================

print(
    "\nTraining Stage 1 distribution:",
    flush=True
)

print(
    train_df[
        "Stage1Name"
    ].value_counts(),
    flush=True
)


print(
    "\nValidation Stage 1 distribution:",
    flush=True
)

print(
    validation_df[
        "Stage1Name"
    ].value_counts(),
    flush=True
)


# ============================================================
# SAVE TEMPORARY DATASETS
# ============================================================

STAGE1_TRAIN_CSV = (
    f"{OUTPUT_DIR}/stage1_train.csv"
)

STAGE1_VALIDATION_CSV = (
    f"{OUTPUT_DIR}/stage1_validation.csv"
)


train_df.to_csv(
    STAGE1_TRAIN_CSV,
    index=False,
)

validation_df.to_csv(
    STAGE1_VALIDATION_CSV,
    index=False,
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(

    [

        transforms.Resize(
            (224, 224)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(
            15
        ),

        transforms.ColorJitter(

            brightness=0.2,

            contrast=0.2,

            saturation=0.2,

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],

        ),

    ]

)


validation_transform = transforms.Compose(

    [

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],

        ),

    ]

)


# ============================================================
# DATASETS
# ============================================================

train_dataset = CropSpeciesDataset(

    STAGE1_TRAIN_CSV,

    class_to_idx,

    train_transform,

    label_column="Stage1Name",

)


validation_dataset = CropSpeciesDataset(

    STAGE1_VALIDATION_CSV,

    class_to_idx,

    validation_transform,

    label_column="Stage1Name",

)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=NUM_WORKERS > 0,

)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=NUM_WORKERS > 0,

)


print(
    f"\nTraining batches: "
    f"{len(train_loader):,}",
    flush=True
)

print(
    f"Validation batches: "
    f"{len(validation_loader):,}",
    flush=True
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading DINOv2 model...",
    flush=True
)

model = DINOv2Classifier(

    num_classes=len(classes)

)


# Freeze DINOv2 backbone for Stage 1

for parameter in model.backbone.parameters():

    parameter.requires_grad = False


model = model.to(
    device
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (

    train_df[
        "Stage1Name"
    ]

    .value_counts()

)


class_weights = []

for class_name in classes:

    count = class_counts.get(
        class_name,
        0
    )

    if count == 0:

        raise ValueError(
            f"Class '{class_name}' "
            f"has zero training samples."
        )

    weight = (

        len(train_df)

        /

        (
            len(classes)

            *

            count
        )

    )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(

    class_weights,

    dtype=torch.float32,

).to(device)


print(
    "\nStage 1 class weights:",
    flush=True
)

for class_name, weight in zip(
    classes,
    class_weights.cpu().numpy(),
):

    print(
        f"  {class_name}: "
        f"{weight:.4f}",
        flush=True
    )


criterion = nn.CrossEntropyLoss(

    weight=class_weights

)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    filter(

        lambda parameter:
        parameter.requires_grad,

        model.parameters(),

    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

)


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(

    "cuda",

    enabled=device.type == "cuda",

)


# ============================================================
# TRAINING
# ============================================================

best_balanced_accuracy = 0.0

epochs_without_improvement = 0

history = []


print(
    "\nStarting Stage 1 training...",
    flush=True
)


for epoch in range(
    NUM_EPOCHS
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    train_true = []

    train_pred = []


    for batch_index, (
        images,
        labels,
    ) in enumerate(train_loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=device.type == "cuda",

        ):

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )

        scaler.update()


        train_loss += loss.item()


        predictions = torch.argmax(
            logits,
            dim=1,
        )


        train_true.extend(
            labels.cpu().numpy()
        )

        train_pred.extend(
            predictions.cpu().numpy()
        )


        if (
            batch_index + 1
        ) % 100 == 0:

            print(

                f"Epoch "
                f"{epoch + 1}/"
                f"{NUM_EPOCHS} | "
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}",

                flush=True,

            )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0

    validation_true = []

    validation_pred = []


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )


            with torch.autocast(

                device_type=device.type,

                dtype=torch.float16,

                enabled=device.type == "cuda",

            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels,
                )


            validation_loss += loss.item()


            predictions = torch.argmax(
                logits,
                dim=1,
            )


            validation_true.extend(
                labels.cpu().numpy()
            )

            validation_pred.extend(
                predictions.cpu().numpy()
            )


    # ========================================================
    # METRICS
    # ========================================================

    train_accuracy = accuracy_score(

        train_true,

        train_pred,

    )


    validation_accuracy = accuracy_score(

        validation_true,

        validation_pred,

    )


    validation_balanced_accuracy = balanced_accuracy_score(

        validation_true,

        validation_pred,

    )


    validation_macro_f1 = f1_score(

        validation_true,

        validation_pred,

        average="macro",

        zero_division=0,

    )


    # ========================================================
    # EPOCH RESULT
    # ========================================================

    epoch_result = {

        "epoch": epoch + 1,

        "train_loss":
        train_loss / len(train_loader),

        "validation_loss":
        validation_loss / len(validation_loader),

        "train_accuracy":
        train_accuracy,

        "validation_accuracy":
        validation_accuracy,

        "validation_balanced_accuracy":
        validation_balanced_accuracy,

        "validation_macro_f1":
        validation_macro_f1,

    }


    history.append(
        epoch_result
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        f"STAGE 1 EPOCH "
        f"{epoch + 1}/"
        f"{NUM_EPOCHS}",
        flush=True,
    )

    print(
        f"Train Loss: "
        f"{epoch_result['train_loss']:.4f}",
        flush=True,
    )

    print(
        f"Validation Loss: "
        f"{epoch_result['validation_loss']:.4f}",
        flush=True,
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}",
        flush=True,
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}",
        flush=True,
    )

    print(
        f"Validation Balanced Accuracy: "
        f"{validation_balanced_accuracy:.4f}",
        flush=True,
    )

    print(
        f"Validation Macro F1: "
        f"{validation_macro_f1:.4f}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        validation_balanced_accuracy
        >
        best_balanced_accuracy
    ):

        best_balanced_accuracy = (
            validation_balanced_accuracy
        )

        epochs_without_improvement = 0


        torch.save(

            {

                "model_state_dict":
                model.state_dict(),

                "class_to_idx":
                class_to_idx,

                "idx_to_class":
                idx_to_class,

                "classes":
                classes,

                "best_balanced_accuracy":
                best_balanced_accuracy,

                "epoch":
                epoch + 1,

            },

            f"{OUTPUT_DIR}/stage1_best.pt",

        )


        print(
            "Saved new best Stage 1 model.",
            flush=True,
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping.",
            flush=True,
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    f"{OUTPUT_DIR}/training_history.csv"
)

pd.DataFrame(
    history
).to_csv(
    history_path,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nStage 1 training complete.",
    flush=True,
)

print(
    f"Best validation balanced accuracy: "
    f"{best_balanced_accuracy:.4f}",
    flush=True,
)

print(
    f"Best checkpoint: "
    f"{OUTPUT_DIR}/stage1_best.pt",
    flush=True,
)

print(
    f"Class mapping: "
    f"{OUTPUT_DIR}/class_mapping.json",
    flush=True,
)

print(
    f"Training history: "
    f"{history_path}",
    flush=True,
)