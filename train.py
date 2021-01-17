import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Net

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def evaluate(testloader: torch.utils.data.DataLoader, net: Net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(
                outputs.data, 1
            )  # 1 - "dimension" with activation energy for each class
            total += labels.size(
                0
            )  # 0 - "dimension" depends on batch_size in DataLoader
            correct += (predicted == labels).sum().item()

    return correct * 100 / total


if __name__ == "__main__":
    # load data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # configure model, loss, optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()  # optimization criterion (loss)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # optimizer

    # train the network
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward prop
            outputs = net(inputs)
            # calculating loss
            loss = criterion(outputs, labels)
            # back prop
            loss.backward()
            # optimizer step
            optimizer.step()
            # log statistics every 2000 mini-batches
            running_loss += loss.item()
            if i % 2000 == 1999:
                logger.info(f"epoch: {epoch}, mini-batch: {i}, loss: {loss:.4f}")
                running_loss = 0.0
    logger.info("finished training")

    # save model
    output_path = "model.pth"
    torch.save(net.state_dict(), output_path)

    # evaluate
    accuracy = evaluate(testloader, net)
    logger.info(f"accuracy: {accuracy}")
