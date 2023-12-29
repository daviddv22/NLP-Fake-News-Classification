import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from classification_MLP import MLP
import hyperparameters as hp
from preprocess import split_dataset

EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
FORMALITY_MODEL = "s-nlp/roberta-base-formality-ranker"
EMBEDDING_FILE = 'embedding.pt'
FORMALITY_SCORE_FILE = 'formality_score.pt'
MODEL_CHECKPOINT = 'model.ckpt3'

def get_binary_labels(raw_labels):
    labels = torch.tensor([1.0 if x in [1.0, 2.0] else 0.0 for x in map(float, raw_labels)]).reshape(len(raw_labels), 1)
    return labels

def sentiment_scores(dataset, tokenizer, model):
    """
    This function takes in a list of sentences and uses the twitter-roberta-base-sentiment pre-trained model
    from Hugging Face to predict the sentiment scores (positive, negative, or neutral) for each sentence
    
    Parameters:
    dataset (list): A list of sentences to analyze the sentiment of.
    
    Returns:
    sentiment_scores: A tensor containing the predicted sentiment scores for each sentence in the dataset.
    """
    sentiment_scores = []
    for _, sentence in enumerate(tqdm(dataset)):
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits)
        sentiment_scores.append(predicted_class)

    return torch.tensor(sentiment_scores)

def train_embedding_generator(dataset, embedding_model):
    """
    generates sentence embeddings using a pretrained model called paraphrase-MiniLM-L6-v2.

    Args:

    dataset (list): A list of strings containing the sentences to be embedded.
    Returns:

    embeddings (list): A list of numpy arrays, where each array is a sentence embedding of the corresponding input sentence.
    """
    # gets the tokenizer and pretrained model
    embeddings = []
    for _, sentence in enumerate(tqdm(dataset)):
        embeddings.append(embedding_model.encode(sentence))
        
    return embeddings

def formality_loss(dataset, tokenizer, model):
    """_summary_
    This function takes in a list of sentences and uses a pre-trained RoBERTa-based formality ranker model 
    from Hugging Face to calculate the formality score of each sentence

    Parameters:

    Args:
        dataset (List[str]): A list of strings where each string is a sentence. 

    Returns:
       a tensor containing formality scores for each of the input sentences.
    """    
    # gets the formality score for the sentence
    toreturn = []
    for _, sentence in enumerate(tqdm(dataset)):
        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        score = torch.nn.functional.softmax(logits, dim=1)[0, predicted_class_id].item()
        toreturn.append(score)
    return torch.tensor(toreturn)


def initialize_models_and_tokenizers():
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    formality_tokenizer = AutoTokenizer.from_pretrained(FORMALITY_MODEL)
    formality_model = AutoModelForSequenceClassification.from_pretrained(FORMALITY_MODEL)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return sentiment_tokenizer, sentiment_model, formality_tokenizer, formality_model, embedding_model

def train_model(model_ip, labels, model, loss_function):
    for epoch in tqdm(range(hp.num_epochs), desc="Training Epochs"):
        learning_rate = (1/(1 + hp.decay_rate * epoch)) * hp.inital_learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for data, label in zip(model_ip, labels):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()

def main():
    data = split_dataset()
    raw_x = data["raw_x"]
    raw_y = data["raw_y"]
    
    labels = get_binary_labels(raw_y)

    sentiment_tokenizer, sentiment_model, formality_tokenizer, formality_model, embedding_model = initialize_models_and_tokenizers()

    embeddings = train_embedding_generator(raw_x, embedding_model)
    formality_score = formality_loss(raw_x, formality_tokenizer, formality_model)
    
    formality_score = formality_score.reshape(formality_score.shape[0], 1)
    model_ip = torch.cat((torch.tensor(embeddings), formality_score), 1)

    model = MLP(hp.input_dim, [hp.hidden_dim, hp.hidden_dim1, hp.hidden_dim2, hp.hidden_dim3], hp.output_dim)
    loss_function = nn.BCELoss()

    train_model(model_ip, labels, model, loss_function)

    torch.save(model.state_dict(), MODEL_CHECKPOINT)

if __name__ == '__main__':
    main()