import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# Load configuration from YAML
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# Constants from config
DATA_DIR = config["data_dir"]
MODEL_ID = config["model_id"]
OUTPUT_DIR = config["output_dir"]
SHUFFLE = config["shuffle"]
TRAIN_TEST_SPLIT = config["train_test_split"]
TRAINING_ARGS = config["training_args"]
EARLY_STOPPING_PATIENCE = config["early_stopping_patience"]

# Load the dataset
dataset = load_dataset("imagefolder", data_dir=DATA_DIR)

if SHUFFLE:
    combined_ds = concatenate_datasets([
        # dataset["train"], 
                                        dataset["test"]
                                        ])
    ds = DatasetDict({"train": combined_ds})
    train_val_split = combined_ds.train_test_split(test_size=TRAIN_TEST_SPLIT, shuffle=True)
else:
    train_val_split = dataset

ds = DatasetDict({
    "train": train_val_split["train"],
    "validation": train_val_split["test"]
})

# Feature extractor
feature_extractor = AutoImageProcessor.from_pretrained(MODEL_ID)
labels = ds["train"].features["label"].names

# Data preprocessing class
class XRayTransform:
    def __init__(self):
        resize_size = feature_extractor.size if isinstance(feature_extractor.size, int) else 224
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ])

    def __call__(self, example_batch):
        example_batch["pixel_values"] = [self.transforms(pil_img) for pil_img in example_batch["image"]]
        return example_batch

ds.set_transform(XRayTransform())

# Collate function
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

# Metric computation
metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# Model initialization
model = ResNetForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: str(i) for i, label in enumerate(labels)},
    ignore_mismatched_sizes=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    eval_strategy=TRAINING_ARGS["eval_strategy"],
    save_strategy=TRAINING_ARGS["save_strategy"],
    logging_strategy=TRAINING_ARGS["logging_strategy"],
    learning_rate=TRAINING_ARGS["learning_rate"],
    per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],
    per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
    num_train_epochs=TRAINING_ARGS["num_train_epochs"],
    save_total_limit=TRAINING_ARGS["save_total_limit"],
    warmup_ratio=TRAINING_ARGS["warmup_ratio"],
    load_best_model_at_end=TRAINING_ARGS["load_best_model_at_end"],
    metric_for_best_model=TRAINING_ARGS["metric_for_best_model"],
    greater_is_better=TRAINING_ARGS["greater_is_better"],
    fp16=TRAINING_ARGS["fp16"],
    report_to=TRAINING_ARGS["report_to"]
)

# Early stopping
early_stopping = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# Training
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
