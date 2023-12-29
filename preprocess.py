from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import logging

# Constants
DATASET_NAME = "liar"
TEST_SIZE = 0.2
NUM_WORDS = 100000
OOV_TOKEN = "<OOV>"
DATA_FILE = "data.json"

logging.basicConfig(level=logging.INFO)

def split_dataset() -> dict:
    """
    Splits the dataset into training and testing sets, tokenizes and pads the statements,
    and saves the processed data to a JSON file.

    Returns:
        dict: A dictionary containing the dataset, training data, test data, raw training, 
              and raw testing statements and labels.
    """
    try:
        logging.info("Loading dataset...")
        dataset = load_dataset(DATASET_NAME).select_columns(["label", "statement"])
        data_split = dataset["train"].train_test_split(test_size=TEST_SIZE, shuffle=True)

        train_x, train_y = preprocess_data(data_split["train"])
        test_x, test_y = preprocess_data(data_split["test"])

        raw_x = data_split["train"]["statement"]
        raw_y = [0 if label <= 2 else 1 for label in data_split["train"]["label"]]

        raw_test_x = data_split["test"]["statement"]
        raw_test_y = [0 if label <= 2 else 1 for label in data_split["test"]["label"]]

        save_data_to_file(train_x, train_y, test_x, test_y)

        return {
            "dataset": dataset, 
            "train_x": train_x, 
            "train_y": train_y, 
            "test_x": test_x, 
            "test_y": test_y,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "raw_test_x": raw_test_x,
            "raw_test_y": raw_test_y
        }

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

def preprocess_data(data_subset):
    """
    Tokenizes and pads the statements in the given data subset.

    Args:
        data_subset (Dataset): A subset of the dataset (train or test).

    Returns:
        tuple: Tokenized and padded statements (x) and corresponding labels (y).
    """
    x = data_subset["statement"]
    y = [0 if label <= 2 else 1 for label in data_subset["label"]]

    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, padding="post")

    return x, y

def save_data_to_file(train_x, train_y, test_x, test_y):
    """
    Saves the processed training and testing data to a JSON file.

    Args:
        train_x (list): Processed training statements.
        train_y (list): Training labels.
        test_x (list): Processed testing statements.
        test_y (list): Testing labels.
    """
    data = {
        "train_x": train_x.tolist(),
        "train_y": train_y,
        "test_x": test_x.tolist(),
        "test_y": test_y
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)
        logging.info(f"Data saved to {DATA_FILE}")

if __name__ == "__main__":
    split_dataset()
