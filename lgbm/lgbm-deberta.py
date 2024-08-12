# Originally based on https://www.kaggle.com/code/ryenhails/deberta-lgbm-with-detailed-code-comments/notebook

import json
import language_tool_python
import lightgbm as lgb
import logging
import numpy as np
import os
import pandas as pd
import pickle
import re
import warnings

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

TRAIN_DATA_PATH = '../../training/training.csv'
TEXT_ROOT = "../../training/text"
CLEANED_TEXT_NAME = 'cleaned-text.txt'

SAVE_PATH = "../../lgbm-models"

WORDS_PATH = "../words.txt"

EVAL_BATCH_SIZE = 8

# LGBM parameters
MAX_DEPTH = 5
NUM_LEAVES = 10
N_ESTIMATORS = 70


# Read the English vocabulary from a file and store it as a set of lowercase words
with open(WORDS_PATH, 'r') as file:
    english_vocab = set(word.strip().lower() for word in file)

# Dictionary for contractions and their expanded forms
contraction_expansions = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
    "could've": "could have",
    "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not",
    "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
    "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
    "I've": "I have", "isn't": "is not",
    "it'd": "it had", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
    "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not",
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so is",
    "that'd": "that would", "that'd've": "that would have",
    "that's": "that is", "there'd": "there had", "there'd've": "there would have", "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
    "they've": "they have",
    "to've": "to have", "wasn't": "was not", "we'd": "we had", "we'd've": "we would have", "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
    "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
    "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'alls": "you alls", "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had", "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

# Compile a regex pattern for the contractions in contraction_expansions
c_re = re.compile('(%s)' % '|'.join(contraction_expansions.keys()))


def create_logging(log_dir, basename, filemode='w'):
    log_path = os.path.join(log_dir, basename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode,
        force=True
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger('')


def expandContractions(text, c_re=c_re):
    """
    Expand contractions in the given text based on the contraction_expansions dictionary.

    Parameters:
    text (str): The input text containing contractions.
    c_re (re.Pattern): The compiled regex pattern for matching contractions.

    Returns:
    str: The text with contractions expanded.
    """

    def replace(match):
        return contraction_expansions[match.group(0)]

    return c_re.sub(replace, text)


def data_preprocessing(x):
    """
    Preprocess the input text by performing a series of cleaning steps:
    - Convert to lowercase
    - Remove contractions
    - Remove digits
    - Remove URLs
    - Remove extra whitespaces
    - Expand contractions
    - Remove repeated punctuation

    Parameters:
    x (str): The input text to preprocess.

    Returns:
    str: The cleaned and preprocessed text.
    """
    x = x.lower()  # Convert to lowercase
    x = re.sub("'\d+", '', x)  # Remove contractions
    x = re.sub("\d+", '', x)  # Remove digits
    x = re.sub("http\w+", '', x)  # Remove URLs
    x = re.sub(r"\s+", " ", x)  # Remove extra whitespaces
    x = expandContractions(x)  # Expand contractions
    x = re.sub(r"\.+", ".", x)  # Remove repeated periods
    x = re.sub(r"\,+", ",", x)  # Remove repeated commas
    x = x.strip()  # Remove leading and trailing whitespaces
    return x


def build_features(train_df, model_path_names, out_dir):

    processed_texts = [data_preprocessing(t) for t in train_df['texts']]

    # Initialize a TfidfVectorizer for word-level TF-IDF features
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',  # Normalize unicode characters
        analyzer='word',  # Perform word-level analysis
        ngram_range=(1, 5),  # Consider n-grams from 1 to 5 words
        min_df=0.05,  # Ignore terms with a document frequency lower than 5%
        max_df=0.95,  # Ignore terms with a document frequency higher than 95%
        sublinear_tf=True,  # Apply sublinear TF scaling
    )

    # Fit the vectorizer on the training data_df and transform the text data_df into TF-IDF features
    train_tfid = vectorizer.fit_transform(processed_texts)
    with open(os.path.join(out_dir, f"tfidf_vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Convert the sparse TF-IDF matrix to a dense array
    dense_matrix = train_tfid.toarray()

    # Create a DataFrame from the dense TF-IDF matrix
    features_df = pd.DataFrame(dense_matrix)

    # Create column names for the TF-IDF features
    tfid_columns = [f'tfidw_{i}' for i in range(len(features_df.columns))]
    features_df.columns = tfid_columns
    logging.info(f"Tfidf feature count: {len(tfid_columns)}")

    # Add 'uuid' and ' to the TF-IDF DataFrame for merging
    features_df['uuids'] = train_df['uuids']
    features_df['labels'] = train_df['labels']

    # Load out-of-fold (OOF) predictions from a models
    for model_path, model_name in model_path_names:
        llm_oof_df = pd.read_csv(os.path.join(model_path, 'oof.csv'))
        llm_oof_df = llm_oof_df.drop(columns=['actual', 'errors'])
        llm_oof_df = llm_oof_df.rename(columns={'predict': f'{model_name}-pred'})
        features_df = pd.merge(features_df, llm_oof_df, on='uuids', how='left')

    # Print the shape of the updated training features DataFrame
    logging.info(f"Training features shape: {features_df.shape}")
    feature_names = list(features_df)
    feature_names.remove('uuids')
    feature_names.remove('labels')
    return features_df, feature_names


def feature_select_wrapper(X, y, feature_names, keep_count=500, random_state=0):
    """
    Perform feature selection using a LightGBM regressor.

    The function trains the model on multiple folds of the data_df, evaluates performance,
    and aggregates feature importances to select the top features.

    Returns:
    list: A list of the top 500 selected features based on their importances.
    """
    features = feature_names

    # Initialize StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Initialize a Series to store feature importances
    fse = pd.Series(0, index=features)

    # Lists to store model performance metrics
    models = []
    predictions = []
    accuracy_scores = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        # Split the data_df into training and testing sets for the current fold
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Initialize the LightGBM classifier with specified hyperparameters
        model = lgb.LGBMClassifier(
            objective='binary',
            metrics=['accuracy', 'f1'],
            learning_rate=0.05,
            max_depth=MAX_DEPTH,
            num_leaves=NUM_LEAVES,
            colsample_bytree=0.3,
            reg_alpha=0.7,
            reg_lambda=0.1,
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
            extra_trees=True,
            class_weight='balanced',
            verbosity=-1,
            # early_stopping_round=75
        )

        # Train the model on the training fold and validate on the test fold
        predictor = model.fit(
            X_train_fold,
            y_train_fold,
            eval_names=['accuracy', 'f1'],
            eval_set=[(X_test_fold, y_test_fold)],
            eval_metric='f1'
        )
        models.append(predictor)

        # Make predictions on the test fold
        predictions_fold = predictor.predict(X_test_fold)
        predictions.append(predictions_fold)

        # Calculate F1 score and Cohen kappa score for the fold
        accuracy_fold = accuracy_score(y_test_fold, predictions_fold)
        accuracy_scores.append(accuracy_fold)
        f1_fold = f1_score(y_test_fold, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        logging.info(f'Accuracy score across fold: {accuracy_fold}')
        logging.info(f'F1 score across fold: {f1_fold}')

        # Aggregate feature importances
        fse += pd.Series(predictor.feature_importances_, index=features)

    # Select the top keep_count features based on their importances
    feature_select = fse.sort_values(ascending=False).index.tolist()[:keep_count]
    return feature_select


def run_training(train, out_dir, model_path_names, var_name, keep_count, n_splits=15, random_state=0):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = create_logging(out_dir, "log.txt")
    logger.info(f"Beginning run for {var_name}")

    train_feats, feature_names = build_features(train, model_path_names, out_dir)

    # Extract feature values from the training features DataFrame, convert to float32, and store in array X
    X = train_feats[feature_names].astype(np.float32).values

    # Extract the 'labels' column, convert to integers, and store in array y
    y = train_feats['labels'].astype(int).values

    # Perform feature selection and get the top features
    feature_select = feature_select_wrapper(X, y, feature_names, keep_count, random_state)
    X = train_feats[feature_select].astype(np.float32).values
    with open(os.path.join(out_dir, "top_features.json"), 'w') as f:
        json.dump(feature_select, f, indent=2)

    # Initialize StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize lists to store model performance metrics and predictions
    accuracy_scores = []
    f1_scores = []
    lgbm_models = []
    predictions = []

    # Iterate through each fold in the cross-validation
    for i, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        logger.info(f'fold{i}')

        # Split the data_df into training and testing sets for the current fold
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Initialize the LightGBM regressor with specified hyperparameters
        model = lgb.LGBMClassifier(
            objective='binary',
            metrics=['accuracy', 'f1'],
            learning_rate=0.05,
            max_depth=MAX_DEPTH,
            num_leaves=NUM_LEAVES,
            colsample_bytree=0.3,
            reg_alpha=0.7,
            reg_lambda=0.1,
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
            extra_trees=True,
            class_weight='balanced',
            verbosity=-1,
            # early_stopping_round=75
        )

        # Train the model on the training fold and validate on the test fold
        predictor = model.fit(
            X_train_fold,
            y_train_fold,
            eval_names=['accuracy', 'f1'],
            eval_set=[(X_test_fold, y_test_fold)],
            eval_metric='f1'
        )
        lgbm_models.append(predictor)

        # Make predictions on the test fold
        predictions_fold = predictor.predict(X_test_fold)
        predictions.append(predictions_fold)

        # Calculate F1 score and Cohen kappa score for the fold
        accuracy_fold = accuracy_score(y_test_fold, predictions_fold)
        accuracy_scores.append(accuracy_fold)
        f1_fold = f1_score(y_test_fold, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        # Save individual model with score
        with open(os.path.join(out_dir, f"lgbm-fold{i}-{f1_fold:0.4f}.pkl"), 'wb') as f:
            pickle.dump(predictor, f)

        logging.info(f'Accuracy score across fold: {accuracy_fold}')
        logging.info(f'F1 score across fold: {f1_fold}')

    # Calculate mean F1 score and accuracy score across all folds
    mean_acc_score = np.mean(accuracy_scores)
    mean_f1_score = np.mean(f1_scores)

    logging.info(f'Mean accuracy score across {n_splits} folds: {mean_acc_score}')
    logging.info(f'Mean F1 score across {n_splits} folds: {mean_f1_score}')

    with open(os.path.join(out_dir, 'lgbm_models.pkl'), 'wb') as f:
        pickle.dump(lgbm_models, f)


if __name__ == '__main__':
    training_df = pd.read_csv(TRAIN_DATA_PATH)
    is_intervention = training_df['is intervention'].to_numpy(dtype=np.int32)
    document_uuids = training_df['uuid'].to_list()

    document_texts = []
    for document_uuid in document_uuids:
        with open(os.path.join(TEXT_ROOT, document_uuid, CLEANED_TEXT_NAME)) as f:
            document_texts.append(f.read())

    train_df = pd.DataFrame({
        'labels': is_intervention,
        'uuids': document_uuids,
        'texts': document_texts,
    })

    model_path_names = [
        ("/fastdata/nzta/llm-models/deberta_v3-base-smooth0.1-43", 'db1'),
        ("/fastdata/nzta/llm-models/deberta_v3-large-smooth0.1-44", 'db2'),
        ("/fastdata/nzta/llm-models/deberta_v3-large-smooth0.1-45", 'db3'),
    ]

    variation_name = "debertas-43b-44l-45l"
    keep_count, n_splits = 60, 8
    run_training(
        train_df,
        os.path.join(SAVE_PATH, variation_name),
        model_path_names,
        variation_name,
        keep_count,
        n_splits,
        42
    )

