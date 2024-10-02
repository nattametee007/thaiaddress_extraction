import joblib
import jsonlines
import json
import scipy
from sklearn_crfsuite import metrics, CRF
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from pythainlp.tokenize import word_tokenize

# from .parser import tokens_to_features
from .utils import range_intersect, preprocess

import joblib

LABELS_MAP = {
    "ชื่อ": "NAME",
    "เลขที่/ถนน": "ADDR",
    "ที่อยู่": "LOC",
    "รหัส": "POST",
    "เบอร์โทร": "PHONE",
    "อีเมล์": "EMAIL",
}
LABELS = list(LABELS_MAP.values())


def address_to_token(address: dict):
    """
    Transform address dictionary to a list of tokens

    Input
    -----
    >>> address = {
        "text": ...,
        "labels": [[start1, stop1, label1], [start2, stop2, label2]]
    }

    Output
    ------
    >>> [(token1, label1), (token2, label2), ...]
    """
    if address["labels"] != []:
        tokens = []
        s = 0
        for token in word_tokenize(address["text"],engine='newmm-safe'):
            start = s
            stop = s + len(token)

            label = "O"
            for s, st, c in address["labels"]:
                if range_intersect(range(start, stop), range(s, st)):
                    label = c
            tokens.append((token, label))
            s = stop
        return tokens
    else:
        return None


def address_to_feature(address: dict):
    """
    Transform address dictionary to features and labels
    """
    tokens = address_to_token(address)
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    labels = [LABELS_MAP.get(label, "O") for _, label in tokens]
    return features, labels


def addresses_to_features(addresses: list):
    """
    Transform list of addresses to features and labels
    """
    X, y = [], []
    for address in addresses:
        # check if already labeled
        if len(address["labels"]) > 0:
            features, labels = address_to_feature(address)
            X.append(features)
            y.append(labels)
    return X, y


def read_file(file_path: str) -> list:
    """
    Read traning path in JSON and return it into a list
    """
    addresses = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            addresses.append(obj)
    return addresses

def json_read(fpath:str) -> list:
    '''
    Read json file
    '''
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_to_file(addresses: list, file_path: str, clean_text=True):
    """
    Save list of addresses into a JSON line file
    """
    if isinstance(addresses[0], str):
        if clean_text:
            addresses = [{"text": preprocess(address)} for address in addresses]
        else:
            addresses = [{"text": address} for address in addresses]
    else:
        print("Address has to be a list of addresses string")
        return
    with jsonlines.open(file_path, mode="w") as writer:
        for address in addresses:
            writer.write(address)
    print("Done saving to {}".format(file_path))


def train(file_path: str, model__output_path: str = None,checkpoint_path:str =None):
    """Training CRF model from a given `file_path`"""
    addresses = json_read(file_path)
    addresses_train, addresses_val = train_test_split(
        addresses, test_size=0.05, random_state=1
    )

    X_train, y_train = addresses_to_features(addresses_train)
    X_val, y_val = addresses_to_features(addresses_val)

    if checkpoint_path:
        crf = joblib.load(checkpoint_path)
        print("Loaded model from {}".format(checkpoint_path))
    else:
        crf = CRF(algorithm="lbfgs", max_iterations=100, all_possible_transitions=True)

    params_space = {
        "c1": scipy.stats.expon(scale=0.5),
        "c2": scipy.stats.expon(scale=0.05),
    }
    f1_scorer = make_scorer(
        metrics.flat_f1_score,
        average="weighted",
        labels=[l for l in LABELS if l != "O"],
    )

    # search
    rs = RandomizedSearchCV(
        crf, params_space, cv=3, verbose=1, n_jobs=-1, n_iter=50, scoring=f1_scorer
    )
    
    rs.fit(X_train, y_train)
    crf = rs.best_estimator_  # get best estimator
    y_pred_val = crf.predict(X_val)
    f1_score_val = metrics.flat_f1_score(
        y_val, y_pred_val, average="weighted", labels=[l for l in LABELS if l != "O"]
    )
    print("Flat F1-Score on validation set = {}".format(f1_score_val))

    if model__output_path:
        joblib.dump(crf, model__output_path)
        print("Save model to {}".format(model__output_path))

    return crf

    