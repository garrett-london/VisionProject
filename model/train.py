import math
import random
import model.misc
import torch
import torch.utils.data
import torch.nn as nn
from PIL import Image
from torchvision import datasets
import os
import numpy as np

path = r"Images"

def train_loop(training_set, model) -> model:
    size = len(training_set.dataset)
    current = 0
    for batch, data in enumerate(training_set):
        current += 1
        loss = m.run_epoch(data)
        print(f"loss: {loss:>7f}  [{(5 * current):>5d}/{size:>5d}]")
    return m


def test_loop(pop, m) -> float:
    dataloader, classes = pop
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0.0
    keys = dict((v, k) for k, v in classes.items())
    with torch.no_grad():
        Imgdata = datasets.ImageFolder(path)
        path2, dirs, files = next(os.walk(path + "\\" + keys[0]))
        count = len(files)
        sizecopy = size
        q = 0
        while q < size:
            rand = random.randint(0, sizecopy - 1)
            temp = 0
            while temp < rand:
                temp += count
            temp = math.floor(temp / count)
            newdata = Imgdata.imgs.pop(rand)
            Imgdata.imgs.insert(rand, newdata)
            image = Image.open(newdata[0])
            pred = m.predict(image)
            if len(pred) > 0 and temp > 0:
                if not pred.get(keys[temp - 1], -1) == -1:
                    correct += 1
            q += 1
            sizecopy -= 1
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")
    return correct


if __name__ == '__main__':
    batch = 5
    device = torch.device('cuda:0')
    pop = model.misc.TrainingModel.get_data(root=path, batch_size=batch, shuffle=True, pin_memory=True)
    training_set, classes = pop
    classes = sorted(list(classes.keys()), key=lambda cls: classes[cls])
    m = model.misc.TrainingModel(labels=classes)
    num_epochs = 8
    best_acc = 0.0
    loss = []
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    try:
        for t in range(num_epochs):
            print(f"Epoch {t + 1}\n")
            m = train_loop(training_set, m)
            acc = test_loop(pop, m, loss_fn)
            if acc >= best_acc:
                m.export(r"model/saved_model.pth")
                best_accuracy = acc
    except KeyboardInterrupt:
        print("Stopped")
