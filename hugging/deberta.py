
import logging
import matplotlib
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import warnings

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, accuracy_score, log_loss, \
    precision_score, recall_score

matplotlib.use('Agg')

warnings.simplefilter('ignore')
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Random seed
RANDOM_SEED = 44

BASE_MODEL_NAME = "deberta"

# Model variant
MODEL_VARIANT = 'v3-large'

# Version number for naming of saved deberta_models
VER = f'{MODEL_VARIANT}-smooth0.1-{RANDOM_SEED}'

# Path for loading model
# https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0
MODEL_PATH = f"microsoft/{BASE_MODEL_NAME}-{MODEL_VARIANT}"

TRAIN_DATA_PATH = '/home/dennis/projects/nzta/interventions/training/training.csv'
TEXT_ROOT = "/home/dennis/projects/nzta/interventions/training/text"
CLEANED_TEXT_NAME = 'cleaned-text.txt'

SAVE_PATH = "/fastdata/nzta/llm-models"

N_SPLITS = 5
INITIAL_LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
GRADIENT_ACCUMULATION_STEPS = 8
# ATTN_IMPLEMENTATION = "flash_attention_2"

MAX_LENGTH = 1024

TRAIN_EPOCHS = 20
# EVAL_SAVE_STEPS = 0.0832
EVAL_SAVE_STEPS = 0.0245


# if MODEL_VARIANT.endswith('small'):
#     TRAIN_BATCH_SIZE = 6
# elif MODEL_VARIANT.endswith('large'):
#     TRAIN_BATCH_SIZE = 1
# else:
TRAIN_BATCH_SIZE = 2


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed=RANDOM_SEED)

# Get the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({
        'pad_token': '<pad>'
    })


def compute_metrics_for_classification(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    ac_value = accuracy_score(labels, preds)
    f1_value = f1_score(labels, preds)
    precision_value = precision_score(labels, preds)
    recall_value = recall_score(labels, preds)
    results = {
        'acc': ac_value,
        'f1': f1_value,
        'precision': precision_value,
        'recall': recall_value
    }
    return results


training_df = pd.read_csv(TRAIN_DATA_PATH)
is_intervention = training_df['is intervention'].to_numpy(dtype=np.int32)
document_uuids = training_df['uuid'].to_list()

document_input_ids = []
document_attention_masks = []
for document_uuid in document_uuids:
    with open(os.path.join(TEXT_ROOT, document_uuid, CLEANED_TEXT_NAME)) as f:
        full_text = f.read()[:MAX_LENGTH * 8]
    full_tokens = tokenizer.tokenize(full_text)[:MAX_LENGTH]
    use_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(full_tokens))
    tokenized_document = tokenizer(use_text, return_tensors='pt')
    document_input_ids.append(tokenized_document['input_ids'][0])
    document_attention_masks.append(tokenized_document['attention_mask'][0])

train_df = pd.DataFrame({
    'input_ids': document_input_ids,
    'attention_mask': document_attention_masks,
    'labels': is_intervention,
    'uuids': document_uuids
})
input_lengths = [len(d) for d in document_input_ids]
print(f"Training on {len(input_lengths)} documents with input lengths {min(input_lengths)} to {max(input_lengths)}")

# Create the splits
train_df = train_df.sample(frac=1).reset_index(drop=True)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
for i, (_, val_index) in enumerate(skf.split(train_df, train_df["labels"])):
    train_df.loc[val_index, "fold"] = i
output_dir = f'{SAVE_PATH}/output_{BASE_MODEL_NAME}-{VER}'
os.makedirs(output_dir, exist_ok=True)
train_df[['uuids', 'fold']].to_csv(os.path.join(output_dir, 'folds.csv'), index=False)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16=False,
    fp16=False,
    learning_rate=INITIAL_LR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_grad_norm=3,  # 0.3,
    optim='adamw_torch',
    # group_by_length=True,
    # torch_compile=True,
    num_train_epochs=TRAIN_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    # evaluation_strategy='epoch',
    evaluation_strategy='steps',
    eval_steps=EVAL_SAVE_STEPS,
    metric_for_best_model='f1',
    save_strategy='steps',
    save_steps=EVAL_SAVE_STEPS,
    save_total_limit=1,
    disable_tqdm=True,
    load_best_model_at_end=True,
    report_to='tensorboard',
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type='cosine',  # "cosine" or "linear" or "constant"
    logging_first_step=True,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    logging_steps=10,
    label_smoothing_factor=0.1,
    # neftune_noise_alpha=2.5,
)

for fold in range(N_SPLITS):

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        # torch_dtype="auto",
    )
    model.config.use_cache = False

    train_fold_df = train_df[train_df['fold'] != fold]
    valid_fold_df = train_df[train_df['fold'] == fold]
    train_ds = Dataset.from_dict({
        'input_ids': train_fold_df['input_ids'].to_list(),
        # 'token_type_ids': train_fold_df['token_type_ids'].to_list(),
        'attention_mask': train_fold_df['attention_mask'].to_list(),
        'labels': train_fold_df['labels'].to_list(),
    })
    valid_fold_df = valid_fold_df.sort_values(by='input_ids', key=lambda i: i.str.len())
    valid_ds = Dataset.from_dict({
        'input_ids': valid_fold_df['input_ids'].to_list(),
        # 'token_type_ids': valid_fold_df['token_type_ids'].to_list(),
        'attention_mask': valid_fold_df['attention_mask'].to_list(),
        'labels': valid_fold_df['labels'].to_list(),
    })
    print(f"Training labels for fold: {train_fold_df['labels'].to_list()}")
    print(f"Validation labels for fold: {valid_fold_df['labels'].to_list()}")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.gradient_checkpointing = True

    # TRAIN WITH TRAINER
    train_start = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_for_classification,
    )
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    print(f'Fold {fold} took {int(time.time() - train_start)} seconds')

    y_true = valid_fold_df['labels'].values
    fold_preds = trainer.predict(valid_ds).predictions
    val_metrics = compute_metrics_for_classification((fold_preds, y_true))

    # SAVE FOLD MODEL AND TOKENIZER
    save_name = f'{BASE_MODEL_NAME}_{VER}/fold_{fold}_acc{val_metrics["acc"]:.4f}_f1{val_metrics["f1"]:.4f}_prec{val_metrics["precision"]:.4f}_recall{val_metrics["recall"]:.4f}'
    trainer.save_model(os.path.join(SAVE_PATH, save_name))

