
import logging
import matplotlib
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import warnings

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, \
    PromptTuningConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

matplotlib.use('Agg')

warnings.simplefilter('ignore')
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Random seed
RANDOM_SEED = 3452

BASE_MODEL_NAME = "RedPajama-INCITE"

# Model variant
MODEL_VARIANT = 'Instruct-3B-v1'

# Version number for naming of saved models
VER = f'{MODEL_VARIANT}-8bit-pref-{RANDOM_SEED}'

# Path for loading model
MODEL_PATH = f"togethercomputer/{BASE_MODEL_NAME}-{MODEL_VARIANT}"

TRAIN_DATA_PATH = '/home/dennis/projects/nzta/interventions/training/training.csv'
TEXT_ROOT = "/home/dennis/projects/nzta/interventions/training/text"
CLEANED_TEXT_NAME = 'cleaned-text.txt'

SAVE_PATH = "/home/dennis/projects/nzta/interventions/llm-models"

N_SPLITS = 5
INITIAL_LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
GRADIENT_ACCUMULATION_STEPS = 2
ATTN_IMPLEMENTATION = "flash_attention_2"

MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 516

INITIAL_PROMPT = """
Does the document relate to interventions involving transportation intended to have some desirable impact, such
as improving health or safety, reducing transit times, encouraging use of public transportation, providing economic
benefits, etc?
"""

TRAIN_EPOCHS = 10
# EVAL_SAVE_STEPS = 0.0832
EVAL_SAVE_STEPS = 0.019


# if MODEL_VARIANT.endswith('small'):
#     TRAIN_BATCH_SIZE = 6
# elif MODEL_VARIANT.endswith('large'):
#     TRAIN_BATCH_SIZE = 1
# else:
TRAIN_BATCH_SIZE = 4


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

document_texts = []
document_input_ids = []
document_attention_masks = []
for document_uuid in document_uuids:
    with open(os.path.join(TEXT_ROOT, document_uuid, CLEANED_TEXT_NAME)) as f:
        full_text = f.read()[:MAX_LENGTH * 8]
    full_tokens = tokenizer.tokenize(full_text)[:MAX_LENGTH]
    use_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(full_tokens))
    document_texts.append(use_text[:use_text.rfind('\n')])
    tokenized_document = tokenizer(use_text, return_tensors='pt')
    document_input_ids.append(tokenized_document['input_ids'][0])
    document_attention_masks.append(tokenized_document['attention_mask'][0])

train_df = pd.DataFrame({
    'input_ids': document_input_ids,
    'attention_mask': document_attention_masks,
    'labels': is_intervention
})


def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # model name parsing
            parsed_name = '.'.join(name.split('.')[4:]).split('.')[0]
            if parsed_name:
                layer_names.append(parsed_name)

    return layer_names


# Create the splits
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
for i, (_, val_index) in enumerate(skf.split(train_df, train_df["labels"])):
    train_df.loc[val_index, "fold"] = i

training_args = TrainingArguments(
    output_dir=f'{SAVE_PATH}/output_{BASE_MODEL_NAME}-{VER}',
    overwrite_output_dir=True,
    bf16=True,
    learning_rate=INITIAL_LR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_grad_norm=3,  # 0.3,
    optim='paged_adamw_32bit',
    group_by_length=True,
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
    neftune_noise_alpha=2.5,
)

for fold in range(N_SPLITS):

    bnb_config = BitsAndBytesConfig(
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=False,
        # bnb_4bit_compute_dtype=torch.bfloat16
        load_in_8bit=True,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        torch_dtype="auto",
        attn_implementation=ATTN_IMPLEMENTATION,
        quantization_config=bnb_config,
    )
    base_model.config.use_cache = False
    base_model.resize_token_embeddings(len(tokenizer))

    base_model = prepare_model_for_kbit_training(base_model)

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

    target_modules = list(set(get_specific_layer_names(base_model)))
    print(f"Configured target_modules: {target_modules}")

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=20,
        token_dim=768,
        num_transformer_submodules=1,
        num_attention_heads=12,
        num_layers=12,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=INITIAL_PROMPT,
        tokenizer_name_or_path=MODEL_PATH
    )

    model = get_peft_model(base_model, peft_config)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    print(model.print_trainable_parameters())
    model.config.gradient_checkpointing = True

    # TRAIN WITH TRAINER
    train_start = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_for_classification
    )
    trainer.train()
    print(f'Fold {fold} took {int(time.time() - train_start)} seconds')

    y_true = valid_fold_df['labels'].values
    fold_preds = trainer.predict(valid_ds).predictions
    val_metrics = compute_metrics_for_classification((fold_preds, y_true))

    # SAVE FOLD MODEL AND TOKENIZER
    save_name = f'{BASE_MODEL_NAME}_{VER}/fold_{fold}_acc{val_metrics["acc"]}_f1{val_metrics["f1"]}'
    trainer.save_model(os.path.join(SAVE_PATH, save_name))

    # # PLOT CONFUSION MATRIX
    # cm = confusion_matrix(y_true, fold_preds.argmax(-1), labels=[x for x in range(3)])
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_title(f'{VER} fold {fold} acc {val_metrics["acc"]:.4f} lloss {val_metrics["log_loss"]:.4f}')
    # plt.imshow(cm)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    #
    # # Save the plot to a file
    # plt.savefig(os.path.join(SAVE_PATH, f'{BASE_MODEL_NAME}_{VER}/fold_{fold}.png'), dpi=300, bbox_inches='tight')

