import torch
from classification_MLP import MLP
from preprocess import split_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import hyperparameters as hp

def initialize_tokenizer_and_model():
    """ Initialize the tokenizer and model for formality ranking. """
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
    return tokenizer, model

def print_results(raw_test_x, results, tokenizer, model):
    """ Print the first 10 results with formality ranking. """
    for i, (label, predicted) in enumerate(results[:10]):
        line = raw_test_x[i]
        correct = 'Correct' if label.item() == predicted.item() else 'Incorrect'
        to_print = f"{line}\t{correct}\t"

        inputs = tokenizer(line, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        formality = 'Informal' if predicted_class_id == 0 else 'Formal'
        print(f"{to_print}{formality}")

def evaluate_model(test_model_ip, test_labels, model):
    """ Evaluate the model and return the results. """
    results = []
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_model_ip):
            outputs = model(data)
            predicted = torch.round(outputs.data)
            results.append((test_labels[i], predicted))
            correct += (predicted == test_labels[i]).sum().item()

    accuracy = 100 * correct / test_labels.size(0)
    print(f'Accuracy of the model on the test images: {accuracy:.2f} %')
    return results

def main():
    # Load data using the updated split_dataset function
    data = split_dataset()

    # Extract the relevant data from the returned dictionary
    raw_test_x = data["raw_test_x"]
    raw_test_y = data["raw_test_y"]

    # Convert raw_test_y into binary labels and format them for the model
    test_labels = [1.0 if y in [1.0, 2.0] else 0.0 for y in map(float, raw_test_y)]
    test_labels = torch.tensor(test_labels).reshape(len(test_labels), 1)

    # Load the tokenizer and model for formality ranking
    tokenizer, model = initialize_tokenizer_and_model()

    # Assuming test_embedding, test_formality_score are already processed and saved
    test_embedding = torch.load('test_embedding.pt')
    test_formality_score = torch.load('test_formality_score.pt').reshape(-1, 1)
    test_model_ip = torch.cat((torch.tensor(test_embedding), test_formality_score), 1)

    # Load the MLP model
    mlp_model = MLP(hp.input_dim, [hp.hidden_dim, hp.hidden_dim1, hp.hidden_dim2, hp.hidden_dim3], hp.output_dim)
    mlp_model.load_state_dict(torch.load('model.ckpt3'))

    # Evaluate the model and print results
    results = evaluate_model(test_model_ip, test_labels, mlp_model)
    print_results(raw_test_x, results, tokenizer, model)

if __name__ == '__main__':
    main()
