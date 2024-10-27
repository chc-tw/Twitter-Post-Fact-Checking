import torch
import transformers
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
import pandas as pd

from src.dataset import TextDataset
from src.model import train_model, get_predictions
from src.utils import report, create_prediction
from src.data_processing import fetch_claim_label, evidence_extract

def main():
    # Constants
    PRETRAINED_MODEL_NAME = "bert-base-uncased"
    NUM_LABELS = 3
    BATCH_SIZE = 16
    EPOCHS = 6
    LEARNING_RATE = 1e-5

    
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    
    train_dataset = TextDataset('train', tokenizer)
    valid_dataset = TextDataset('valid', tokenizer)
    test_dataset = TextDataset('test', tokenizer)

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    testloader = DataLoader(test_dataset, batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    transformers.logging.set_verbosity_error()

    train_model(model, trainloader, validloader, optimizer, EPOCHS, device)

    predictions = get_predictions(model, testloader)
    
    y_truth = valid_dataset.df['label'].to_list()
    y_tr = [int(t) for t in y_truth]
    y_pre = [int(t) for t in predictions]
    report(y_pre, y_tr)

    create_prediction(predictions, "test.json", "submission.csv")

if __name__ == "__main__":
    main()
