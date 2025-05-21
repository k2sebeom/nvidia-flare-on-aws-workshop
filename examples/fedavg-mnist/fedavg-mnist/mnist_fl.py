import os
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import nvflare.client as flare
from nvflare.client import ParamsType, FLModel
from nvflare.client.tracking import MLflowWriter


from data.digit_dataset import DigitDataset
from mnist_net import Net


DATASET_PATH = '/tmp/nvflare/data/mnist'
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_accuracy(model: Net, args: argparse.Namespace):
    dataset = DigitDataset.load(
        f'{DATASET_PATH}/test.pt',
        bucket_name=f'{args.bucket_name}-{flare.get_site_name()}',
        download=f'{flare.get_site_name()}_test.pt'
    )

    test_loader = DataLoader(dataset, batch_size=100, shuffle=False)
    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def _train(input_model: FLModel, args: argparse.Namespace):
    batch_size = args.batch_size
    epochs = args.epochs
    bucket_name = args.bucket_name

    dataset = DigitDataset.load(
        f'{DATASET_PATH}/train.pt',
        bucket_name=f'{bucket_name}-{flare.get_site_name()}',
        download=f'{flare.get_site_name()}.pt'
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Net().to(DEVICE)
    print(f"current_round={input_model.current_round}")

    try:
        model.load_state_dict(input_model.params)
    except Exception as e:
        print(f'Failed to load model {e}')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    steps = epochs * len(train_loader)

    log_writer = MLflowWriter()
    log_writer.set_tag("client", flare.get_site_name())
    log_period = args.log_period

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print statistics every {log_period} batches
            if (batch_idx + 1) % log_period == 0:
                avg_loss = running_loss / log_period
                print(f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{batch_idx+1}/{len(train_loader)}], '
                    f'Loss: {avg_loss:.4f}')

                running_loss = 0

            global_step = input_model.current_round * steps + epoch * len(train_loader) + batch_idx
            log_writer.log_metrics({
                "train_loss": loss.item(),
            }, global_step)

        # Calculate average loss for the epoch
        print(f'Epoch [{epoch+1}/{epochs}], '
            f'Loss: {loss.item():.4f}')

    print(f"finished round: {input_model.current_round}")
    torch.save(model.state_dict(), f'./round_{input_model.current_round}.pt')
    torch.save(model.state_dict(), './latest.pt')

    acc = get_accuracy(model, args)
    log_writer.log_metrics({
        "accuracy": acc,
    }, input_model.current_round)

    output_model = flare.FLModel(
        params=model.cpu().state_dict(),
        metrics={ "accuracy": acc },
        current_round=input_model.current_round,
        params_type=ParamsType.FULL,
    )
    flare.send(output_model)


def _submit_model():
    model = Net()
    model.load_state_dict(torch.load('./latest.pt'))
    output_model = flare.FLModel(
        params=model.cpu().state_dict(),
        params_type=ParamsType.FULL,
    )
    flare.send(output_model)


def _evaluate(input_model: FLModel, args: argparse.Namespace):
    model = Net().to(DEVICE)

    try:
        model.load_state_dict(input_model.params)
    except Exception as e:
        print(f'Failed to load model {e}')
    
    acc = get_accuracy(model, args)
    output_model = flare.FLModel(
        metrics={"accuracy": acc},
    )
    flare.send(output_model)


def main():
    parser = argparse.ArgumentParser(description='MNIST Federated Learning Client')
    parser.add_argument('--bucket-name', type=str, required=True,
                      help='Name of the bucket to store/retrieve data')
    parser.add_argument('--batch-size', type=int, default=512,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for optimizer')
    parser.add_argument('--log-period', type=int, default=10,
                      help='How often to log training statistics')
    args = parser.parse_args()

    flare.init()

    while flare.is_running():
        input_model = flare.receive()
        print('!!! Received model')
        print(input_model)
        if flare.is_submit_model():
            print("!!!! Submit Model !!!")
            _submit_model()
        elif flare.is_evaluate():
            print("!!!! Evaluate !!!")
            _evaluate(input_model, args)
        elif flare.is_train():
            print("!!!! Train !!!")
            _train(input_model, args)
        else:
            flare.send(None, True)
        

if __name__ == "__main__":
    main()
