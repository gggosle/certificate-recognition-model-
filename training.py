from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from dataset import CertificateDataset
from transformers import LayoutLMv3Processor
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import random_split


# 1. Define Labels (MUST match your JSON labels exactly + "O")
# Standard BIO tagging usually looks like: ["O", "B-NAME", "I-NAME", "B-COURSE", "I-COURSE"...]
# For simple extraction, just matching the entity name works too.
label_list = ["O", "NAME", "COURSE_NAME", "ISSUER", "SIGNATURE", "DATE"]

# 2. Setup Processor
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False # IMPORTANT: We provide our own OCR
)
IMAGE_DIR = "./certificate-dataset-v1/data/images"
LABEL_DIR = "./certificate-dataset-v1/data/labels"

# 3. Create Dataset
train_dataset = CertificateDataset(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    processor=processor,
    label_list=label_list,
)

dataset_size = len(train_dataset)
train_size = int(0.9 * dataset_size)
eval_size = dataset_size - train_size

train_ds, eval_ds = random_split(train_dataset, [train_size, eval_size])

# 4. Load Model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list)
)

# 5. Training Args
# Clear any existing CUDA cache
torch.cuda.empty_cache()

args = TrainingArguments(
    output_dir="layoutlmv3-finetuned-certificates",

    num_train_epochs=10,
    per_device_train_batch_size=1,   # 🔥 MUST
    per_device_eval_batch_size=1,    # 🔥 MUST
    gradient_accumulation_steps=4,   # keep effective batch = 

    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    eval_strategy="steps",
    eval_steps=30,

    save_strategy="steps",
    save_steps=30,
    save_total_limit=3,

    logging_steps=10,

    remove_unused_columns=False,

    fp16=True,                       # 🔥 MUST
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

print("\n" + "="*50)
print("TRAINING ON GPU")
print("="*50 + "\n")

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=None # Default collator usually works if tensors are padded right
)

torch.cuda.empty_cache()
# 7. Start Training
trainer.train(resume_from_checkpoint=False)

# 8. Save
trainer.save_model("my_custom_model")
print("🎉 Model trained and saved!")