import torch
from transformers import BertForSequenceClassification
import tqdm

def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        print("verifying...")
        for data in tqdm.tqdm(dataloader):
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]

            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors,
                          token_type_ids=segments_tensors,
                          attention_mask=masks_tensors)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

def train_model(model, trainloader, validloader, optimizer, num_epochs, device):
    best_f1 = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        
        for data in tqdm.tqdm(trainloader):
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

            optimizer.zero_grad()

            outputs = model(input_ids=tokens_tensors,
                          token_type_ids=segments_tensors,
                          attention_mask=masks_tensors,
                          labels=labels)

            loss = outputs[0]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        _, f1, acc = get_predictions(model, validloader, compute_acc=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, f"best{epoch + 1}.pt")
        
        print(f'[epoch {epoch + 1}] loss: {running_loss:.3f}, acc: {acc:.3f}, f1: {f1:.3f}')
