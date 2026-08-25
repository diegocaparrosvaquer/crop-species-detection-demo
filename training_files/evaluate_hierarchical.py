import json
import os

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import CropSpeciesDataset
from model import DINOv2Classifier


# ============================================================
# CONFIGURATION
# ============================================================

TEST_CSV = (
    "/workspace/data/combined/test.csv"
)

STAGE1_CHECKPOINT = (
    "/workspace/outputs/dinov2/hierarchical/stage1/"
    "stage1_best.pt"
)

STAGE2_CHECKPOINT = (
    "/workspace/outputs/dinov2/hierarchical/stage2/"
    "stage2_best.pt"
)

OUTPUT_DIR = (
    "/workspace/outputs/dinov2/hierarchical/evaluation"
)

BATCH_SIZE = 32

NUM_WORKERS = 8


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
# HIERARCHICAL CLASS DEFINITIONS
# ============================================================

NO_CROPLAND_CLASS = "no cropland"

CROP_CLASSES = [
    "banana",
    "maize",
    "millets",
    "rapeseed",
    "soya",
    "sorghum",
    "sunflower",
    "vineyard",
    "wheat type crop"
]

VALID_CLASSES = [
    NO_CROPLAND_CLASS
] + CROP_CLASSES


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(checkpoint_path):
    print(
        f"\nLoading checkpoint:\n{checkpoint_path}",
        flush=True
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    required_keys = [
        "model_state_dict",
        "class_to_idx",
        "classes"
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise ValueError(
                f"Checkpoint {checkpoint_path} "
                f"is missing required key: {key}"
            )

    return checkpoint


# ============================================================
# LOAD TEST DATA
# ============================================================

print(
    "\nLoading test data...",
    flush=True
)

test_df = pd.read_csv(
    TEST_CSV
)


# ============================================================
# CHECK TEST CSV
# ============================================================

required_columns = [
    "image_path",
    "label"
]

for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"Test CSV is missing required column "
            f"'{column}'.\n"
            f"Available columns: "
            f"{test_df.columns.tolist()}"
        )


# ============================================================
# FILTER VALID CLASSES
# ============================================================

test_df = test_df[
    test_df["label"].isin(
        VALID_CLASSES
    )
].reset_index(
    drop=True
)


print(
    f"Test images: "
    f"{len(test_df):,}",
    flush=True
)


print(
    "\nTest class distribution:",
    flush=True
)

print(
    test_df["label"].value_counts(),
    flush=True
)


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

if "source" in test_df.columns:

    print(
        "\nTest source distribution:",
        flush=True
    )

    print(
        test_df["source"].value_counts(),
        flush=True
    )


# ============================================================
# TRANSFORMS
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[

            0.485,
            0.456,
            0.406

        ],

        std=[

            0.229,
            0.224,
            0.225

        ]

    )

])


# ============================================================
# TEST DATASET
# ============================================================
#
# We use the same dataset implementation as training so that
# both local paths and s3:// paths are supported.
#
# ============================================================

class HierarchicalTestDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

        # Dummy mapping because we only need image loading.
        # The actual ground-truth label is returned as a string.
        self.loader = CropSpeciesDataset.__new__(
            CropSpeciesDataset
        )

        self.loader.s3_endpoint_url = os.getenv(
            "S3_ENDPOINT_URL",
            "https://hw81s3.iiasa.ac.at"
        )

        self.loader.s3_region = os.getenv(
            "AWS_DEFAULT_REGION",
            "us-east-1"
        )

        self.loader.s3_client = None


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        image_path = row[
            "image_path"
        ]

        image = self.loader._load_image(
            image_path
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        label = row[
            "label"
        ]

        return image, label


# ============================================================
# DATASET AND DATALOADER
# ============================================================

test_dataset = HierarchicalTestDataset(

    dataframe=test_df,

    transform=test_transform

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=(
        NUM_WORKERS > 0
    )

)


# ============================================================
# LOAD STAGE 1 CHECKPOINT
# ============================================================

stage1_checkpoint = load_checkpoint(
    STAGE1_CHECKPOINT
)

stage1_class_to_idx = (
    stage1_checkpoint["class_to_idx"]
)

stage1_classes = (
    stage1_checkpoint["classes"]
)

stage1_idx_to_class = {
    int(index): class_name
    for class_name, index
    in stage1_class_to_idx.items()
}


print(
    "\nStage 1 classes:",
    flush=True
)

for class_name in stage1_classes:

    print(
        f"  {class_name}",
        flush=True
    )


# ============================================================
# LOAD STAGE 2 CHECKPOINT
# ============================================================

stage2_checkpoint = load_checkpoint(
    STAGE2_CHECKPOINT
)

stage2_class_to_idx = (
    stage2_checkpoint["class_to_idx"]
)

stage2_classes = (
    stage2_checkpoint["classes"]
)

stage2_idx_to_class = {
    int(index): class_name
    for class_name, index
    in stage2_class_to_idx.items()
}


print(
    "\nStage 2 classes:",
    flush=True
)

for class_name in stage2_classes:

    print(
        f"  {class_name}",
        flush=True
    )


# ============================================================
# VERIFY EXPECTED CLASSES
# ============================================================

if set(stage1_classes) != {
    "no cropland",
    "cropland"
}:

    raise ValueError(
        "Stage 1 checkpoint does not contain the "
        "expected classes {'no cropland', 'cropland'}."
    )


if set(stage2_classes) != set(CROP_CLASSES):

    raise ValueError(
        "Stage 2 checkpoint classes do not match "
        "the current crop classes.\n\n"
        f"Expected:\n{CROP_CLASSES}\n\n"
        f"Checkpoint:\n{stage2_classes}"
    )


# ============================================================
# LOAD STAGE 1 MODEL
# ============================================================

print(
    "\nLoading Stage 1 model...",
    flush=True
)

stage1_model = DINOv2Classifier(

    num_classes=len(
        stage1_classes
    )

)

stage1_model.load_state_dict(

    stage1_checkpoint[
        "model_state_dict"
    ]

)

stage1_model = stage1_model.to(
    device
)

stage1_model.eval()


# ============================================================
# LOAD STAGE 2 MODEL
# ============================================================

print(
    "\nLoading Stage 2 model...",
    flush=True
)

stage2_model = DINOv2Classifier(

    num_classes=len(
        stage2_classes
    )

)

stage2_model.load_state_dict(

    stage2_checkpoint[
        "model_state_dict"
    ]

)

stage2_model = stage2_model.to(
    device
)

stage2_model.eval()


# ============================================================
# EVALUATION STORAGE
# ============================================================

all_true_labels = []

all_predicted_labels = []

all_stage1_predictions = []

all_stage1_confidences = []

all_stage2_predictions = []

all_stage2_confidences = []

all_stage2_was_used = []


# ============================================================
# HIERARCHICAL EVALUATION
# ============================================================

print(
    "\nEvaluating hierarchical pipeline...",
    flush=True
)


with torch.no_grad():

    for batch_index, (
        images,
        labels
    ) in enumerate(test_loader):

        images = images.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # STAGE 1
        # ----------------------------------------------------

        with torch.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=device.type == "cuda"

        ):

            stage1_logits = stage1_model(
                images
            )


        stage1_probabilities = torch.softmax(
            stage1_logits,
            dim=1
        )


        stage1_predictions = torch.argmax(
            stage1_probabilities,
            dim=1
        )


        stage1_confidences = (
            stage1_probabilities.max(
                dim=1
            ).values
        )


        # ----------------------------------------------------
        # STAGE 2
        # ----------------------------------------------------
        #
        # We run Stage 2 only on examples predicted as
        # cropland by Stage 1.
        #
        # ----------------------------------------------------

        batch_stage2_predictions = [
            None
            for _ in range(len(labels))
        ]

        batch_stage2_confidences = [
            np.nan
            for _ in range(len(labels))
        ]

        crop_indices = [

            i

            for i in range(len(labels))

            if stage1_idx_to_class[
                stage1_predictions[i].item()
            ] == "cropland"

        ]


        if crop_indices:

            crop_indices_tensor = torch.tensor(
                crop_indices,
                dtype=torch.long,
                device=device
            )


            crop_images = images[
                crop_indices_tensor
            ]


            with torch.autocast(

                device_type=device.type,

                dtype=torch.float16,

                enabled=device.type == "cuda"

            ):

                stage2_logits = stage2_model(
                    crop_images
                )


            stage2_probabilities = torch.softmax(
                stage2_logits,
                dim=1
            )


            stage2_predictions = torch.argmax(
                stage2_probabilities,
                dim=1
            )


            stage2_confidences = (
                stage2_probabilities.max(
                    dim=1
                ).values
            )


            for local_index, original_index in enumerate(
                crop_indices
            ):

                batch_stage2_predictions[
                    original_index
                ] = stage2_idx_to_class[
                    stage2_predictions[
                        local_index
                    ].item()
                ]


                batch_stage2_confidences[
                    original_index
                ] = stage2_confidences[
                    local_index
                ].item()


        # ----------------------------------------------------
        # BUILD FINAL HIERARCHICAL PREDICTION
        # ----------------------------------------------------

        for index in range(len(labels)):

            true_label = labels[index]


            stage1_prediction = (
                stage1_idx_to_class[
                    stage1_predictions[
                        index
                    ].item()
                ]
            )


            stage1_confidence = (
                stage1_confidences[
                    index
                ].item()
            )


            stage2_prediction = (
                batch_stage2_predictions[
                    index
                ]
            )


            stage2_confidence = (
                batch_stage2_confidences[
                    index
                ]
            )


            if (
                stage1_prediction
                ==
                NO_CROPLAND_CLASS
            ):

                final_prediction = (
                    NO_CROPLAND_CLASS
                )

                stage2_used = False

            else:

                final_prediction = (
                    stage2_prediction
                )

                stage2_used = True


            all_true_labels.append(
                true_label
            )

            all_predicted_labels.append(
                final_prediction
            )

            all_stage1_predictions.append(
                stage1_prediction
            )

            all_stage1_confidences.append(
                stage1_confidence
            )

            all_stage2_predictions.append(
                stage2_prediction
            )

            all_stage2_confidences.append(
                stage2_confidence
            )

            all_stage2_was_used.append(
                stage2_used
            )


        if (
            batch_index + 1
        ) % 10 == 0:

            print(

                f"Processed "
                f"{batch_index + 1}/"
                f"{len(test_loader)} batches",

                flush=True

            )


# ============================================================
# CONVERT RESULTS TO ARRAYS
# ============================================================

all_true_labels = np.array(
    all_true_labels,
    dtype=object
)

all_predicted_labels = np.array(
    all_predicted_labels,
    dtype=object
)

all_stage1_predictions = np.array(
    all_stage1_predictions,
    dtype=object
)

all_stage1_confidences = np.array(
    all_stage1_confidences,
    dtype=float
)

all_stage2_predictions = np.array(
    all_stage2_predictions,
    dtype=object
)

all_stage2_confidences = np.array(
    all_stage2_confidences,
    dtype=float
)

all_stage2_was_used = np.array(
    all_stage2_was_used,
    dtype=bool
)


# ============================================================
# HIERARCHICAL FINAL METRICS
# ============================================================

accuracy = accuracy_score(

    all_true_labels,

    all_predicted_labels

)


balanced_accuracy = balanced_accuracy_score(

    all_true_labels,

    all_predicted_labels

)


macro_f1 = f1_score(

    all_true_labels,

    all_predicted_labels,

    labels=VALID_CLASSES,

    average="macro",

    zero_division=0

)


weighted_f1 = f1_score(

    all_true_labels,

    all_predicted_labels,

    labels=VALID_CLASSES,

    average="weighted",

    zero_division=0

)


print(
    "\n" + "=" * 80,
    flush=True
)

print(
    "HIERARCHICAL TEST RESULTS",
    flush=True
)

print(
    "=" * 80,
    flush=True
)

print(
    f"Accuracy:           {accuracy:.4f}",
    flush=True
)

print(
    f"Balanced Accuracy:  {balanced_accuracy:.4f}",
    flush=True
)

print(
    f"Macro F1:           {macro_f1:.4f}",
    flush=True
)

print(
    f"Weighted F1:        {weighted_f1:.4f}",
    flush=True
)


# ============================================================
# STAGE 1 GROUND TRUTH
# ============================================================

stage1_true_labels = np.array([

    NO_CROPLAND_CLASS
    if label == NO_CROPLAND_CLASS
    else "cropland"

    for label in all_true_labels

])


# ============================================================
# STAGE 1 METRICS
# ============================================================

stage1_accuracy = accuracy_score(

    stage1_true_labels,

    all_stage1_predictions

)


stage1_balanced_accuracy = (
    balanced_accuracy_score(
        stage1_true_labels,
        all_stage1_predictions
    )
)


stage1_macro_f1 = f1_score(

    stage1_true_labels,

    all_stage1_predictions,

    average="macro",

    zero_division=0

)


print(
    "\n" + "=" * 80,
    flush=True
)

print(
    "STAGE 1 RESULTS",
    flush=True
)

print(
    "=" * 80,
    flush=True
)

print(
    f"Accuracy:           {stage1_accuracy:.4f}",
    flush=True
)

print(
    f"Balanced Accuracy:  {stage1_balanced_accuracy:.4f}",
    flush=True
)

print(
    f"Macro F1:           {stage1_macro_f1:.4f}",
    flush=True
)


# ============================================================
# STAGE 2 EVALUATION ON TRUE CROP IMAGES
# ============================================================
#
# This measures the intrinsic performance of Stage 2,
# independent of Stage 1 errors.
#
# ============================================================

true_crop_mask = (
    all_true_labels != NO_CROPLAND_CLASS
)


stage2_true_labels = (
    all_true_labels[
        true_crop_mask
    ]
)


stage2_predicted_labels = (
    all_stage2_predictions[
        true_crop_mask
    ]
)


stage2_accuracy = accuracy_score(

    stage2_true_labels,

    stage2_predicted_labels

)


stage2_balanced_accuracy = (
    balanced_accuracy_score(
        stage2_true_labels,
        stage2_predicted_labels
    )
)


stage2_macro_f1 = f1_score(

    stage2_true_labels,

    stage2_predicted_labels,

    labels=CROP_CLASSES,

    average="macro",

    zero_division=0

)


stage2_weighted_f1 = f1_score(

    stage2_true_labels,

    stage2_predicted_labels,

    labels=CROP_CLASSES,

    average="weighted",

    zero_division=0

)


print(
    "\n" + "=" * 80,
    flush=True
)

print(
    "STAGE 2 RESULTS ON TRUE CROPLAND",
    flush=True
)

print(
    "=" * 80,
    flush=True
)

print(
    f"Crop test samples:  {len(stage2_true_labels):,}",
    flush=True
)

print(
    f"Accuracy:           {stage2_accuracy:.4f}",
    flush=True
)

print(
    f"Balanced Accuracy:  {stage2_balanced_accuracy:.4f}",
    flush=True
)

print(
    f"Macro F1:           {stage2_macro_f1:.4f}",
    flush=True
)

print(
    f"Weighted F1:        {stage2_weighted_f1:.4f}",
    flush=True
)


# ============================================================
# STAGE 2 REPORT
# ============================================================

stage2_report = classification_report(

    stage2_true_labels,

    stage2_predicted_labels,

    labels=CROP_CLASSES,

    target_names=CROP_CLASSES,

    zero_division=0,

    output_dict=True

)


pd.DataFrame(
    stage2_report
).transpose().to_csv(

    f"{OUTPUT_DIR}/"
    "stage2_classification_report.csv"

)


# ============================================================
# FINAL CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    all_true_labels,

    all_predicted_labels,

    labels=VALID_CLASSES,

    target_names=VALID_CLASSES,

    zero_division=0,

    output_dict=True

)


report_df = pd.DataFrame(
    report
).transpose()


report_df.to_csv(

    f"{OUTPUT_DIR}/"
    "classification_report.csv"

)


print(
    "\nHierarchical Classification Report:",
    flush=True
)

print(

    classification_report(

        all_true_labels,

        all_predicted_labels,

        labels=VALID_CLASSES,

        target_names=VALID_CLASSES,

        zero_division=0

    ),

    flush=True

)


# ============================================================
# FINAL CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_true_labels,

    all_predicted_labels,

    labels=VALID_CLASSES

)


cm_df = pd.DataFrame(

    cm,

    index=VALID_CLASSES,

    columns=VALID_CLASSES

)


cm_df.to_csv(

    f"{OUTPUT_DIR}/"
    "confusion_matrix.csv"

)


# ============================================================
# STAGE 1 CONFUSION MATRIX
# ============================================================

stage1_cm = confusion_matrix(

    stage1_true_labels,

    all_stage1_predictions,

    labels=[
        NO_CROPLAND_CLASS,
        "cropland"
    ]

)


pd.DataFrame(

    stage1_cm,

    index=[
        NO_CROPLAND_CLASS,
        "cropland"
    ],

    columns=[
        NO_CROPLAND_CLASS,
        "cropland"
    ]

).to_csv(

    f"{OUTPUT_DIR}/"
    "stage1_confusion_matrix.csv"

)


# ============================================================
# STAGE 2 CONFUSION MATRIX
# ============================================================

stage2_cm = confusion_matrix(

    stage2_true_labels,

    stage2_predicted_labels,

    labels=CROP_CLASSES

)


pd.DataFrame(

    stage2_cm,

    index=CROP_CLASSES,

    columns=CROP_CLASSES

).to_csv(

    f"{OUTPUT_DIR}/"
    "stage2_confusion_matrix.csv"

)


# ============================================================
# PREDICTION RESULTS
# ============================================================

results_df = test_df.copy()


results_df["stage1_prediction"] = (
    all_stage1_predictions
)

results_df["stage1_confidence"] = (
    all_stage1_confidences
)

results_df["stage2_prediction"] = (
    all_stage2_predictions
)

results_df["stage2_confidence"] = (
    all_stage2_confidences
)

results_df["stage2_used"] = (
    all_stage2_was_used
)

results_df["final_prediction"] = (
    all_predicted_labels
)

results_df["correct"] = (
    results_df["label"].values
    ==
    results_df["final_prediction"].values
)


results_df.to_csv(

    f"{OUTPUT_DIR}/"
    "test_predictions.csv",

    index=False

)


# ============================================================
# SOURCE-SPECIFIC EVALUATION
# ============================================================

def evaluate_subset(
    subset_df,
    true_labels,
    predicted_labels,
    name
):

    if len(subset_df) == 0:

        return None


    subset_accuracy = accuracy_score(

        true_labels,

        predicted_labels

    )


    subset_balanced_accuracy = (
        balanced_accuracy_score(
            true_labels,
            predicted_labels
        )
    )


    subset_macro_f1 = f1_score(

        true_labels,

        predicted_labels,

        labels=VALID_CLASSES,

        average="macro",

        zero_division=0

    )


    print(
        "\n" + "=" * 80,
        flush=True
    )

    print(
        f"SOURCE: {name}",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )

    print(
        f"Samples:             {len(subset_df):,}",
        flush=True
    )

    print(
        f"Accuracy:            {subset_accuracy:.4f}",
        flush=True
    )

    print(
        f"Balanced Accuracy:   {subset_balanced_accuracy:.4f}",
        flush=True
    )

    print(
        f"Macro F1:            {subset_macro_f1:.4f}",
        flush=True
    )


    return {

        "samples": int(
            len(subset_df)
        ),

        "accuracy": float(
            subset_accuracy
        ),

        "balanced_accuracy": float(
            subset_balanced_accuracy
        ),

        "macro_f1": float(
            subset_macro_f1
        )

    }


source_metrics = {}


if "source" in test_df.columns:

    for source_name in sorted(
        test_df["source"].dropna().unique()
    ):

        source_mask = (
            test_df["source"].values
            ==
            source_name
        )


        source_metrics[source_name] = evaluate_subset(

            test_df.loc[source_mask],

            all_true_labels[
                source_mask
            ],

            all_predicted_labels[
                source_mask
            ],

            source_name

        )


# ============================================================
# FINAL METRICS SUMMARY
# ============================================================

summary = {

    "test_samples": int(
        len(all_true_labels)
    ),

    "classes": VALID_CLASSES,

    "accuracy": float(
        accuracy
    ),

    "balanced_accuracy": float(
        balanced_accuracy
    ),

    "macro_f1": float(
        macro_f1
    ),

    "weighted_f1": float(
        weighted_f1
    ),

    "stage1": {

        "accuracy": float(
            stage1_accuracy
        ),

        "balanced_accuracy": float(
            stage1_balanced_accuracy
        ),

        "macro_f1": float(
            stage1_macro_f1
        )

    },

    "stage2_true_crop": {

        "samples": int(
            len(stage2_true_labels)
        ),

        "accuracy": float(
            stage2_accuracy
        ),

        "balanced_accuracy": float(
            stage2_balanced_accuracy
        ),

        "macro_f1": float(
            stage2_macro_f1
        ),

        "weighted_f1": float(
            stage2_weighted_f1
        )

    },

    "source_metrics": source_metrics

}


with open(

    f"{OUTPUT_DIR}/"
    "metrics.json",

    "w"

) as file:

    json.dump(

        summary,

        file,

        indent=4

    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80,
    flush=True
)

print(
    "EVALUATION COMPLETE",
    flush=True
)

print(
    "=" * 80,
    flush=True
)

print(
    f"Results saved to:\n"
    f"{OUTPUT_DIR}",
    flush=True
)

print(
    "\nFiles:",
    flush=True
)

print(
    "  metrics.json",
    flush=True
)

print(
    "  classification_report.csv",
    flush=True
)

print(
    "  stage2_classification_report.csv",
    flush=True
)

print(
    "  confusion_matrix.csv",
    flush=True
)

print(
    "  stage1_confusion_matrix.csv",
    flush=True
)

print(
    "  stage2_confusion_matrix.csv",
    flush=True
)

print(
    "  test_predictions.csv",
    flush=True
)