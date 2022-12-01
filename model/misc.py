import torch
import torch.nn.functional
import torch.optim as opt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from typing import Union

from torchvision.models import ResNet50_Weights


class Model:
    def __init__(self, labels: "list[str]", model: "torch.nn.Module" = None, threshold=0.6):
        self._labels = labels
        if model is None:
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self._labels))
        elif model.fc.out_features != len(labels):
            raise ValueError(f"Module provided ({model.fc.out_features} features)"
                             f" does not match the labels provided ({len(labels)} labels)")
        self._model = model.to(torch.device("cuda:0"))
        self.threshold = float(threshold)

    @classmethod
    def get_transform(cls) -> "torchvision.transforms.Compose":
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,337)),
            torchvision.transforms.ToTensor(),
        ])

    @classmethod
    def transform(cls, data: "Image.Image") -> "torch.tensor":
        return cls.get_transform()(data)

    @torch.no_grad()
    def predict(self, img_data: "Union[torch.Tensor, Image.Image]") -> "dict[str, float]":
        if isinstance(img_data, Image.Image):
            img_data = Model.transform(img_data).unsqueeze(0)
        m = self._model.eval()
        predictions = m(img_data.to(torch.device("cuda:0")))
        probabilities = torch.nn.functional.softmax(predictions, dim=1).squeeze(0)
        results = {label: 0.0 for label in self._labels}
        for label_id, score in enumerate(probabilities):
            results[self._labels[label_id]] += float(score)
        return results

    @classmethod
    def from_file(cls, path: "str", labels: "list[str]", threshold=0.6) -> "Model":
        pth_model = torchvision.models.resnet18(pretrained=False, num_classes=len(labels))
        pth_model.load_state_dict(torch.load(path))
        return Model(model=pth_model, threshold=threshold, labels=labels)

    def export(self, path: "str"):
        torch.save(self._model.state_dict(), path)


class TrainingModel(Model):
    def __init__(self, model: "torch.nn.Module" = None, threshold=0.6, labels: "list[str]" = None, optimizer=None,
                 loss=None):
        super(TrainingModel, self).__init__(labels=labels, model=model, threshold=threshold)
        self._model.train()
        if optimizer is None:
            optimizer = opt.SGD(self._model.parameters(), lr=0.001)
        self.__optimizer = optimizer

        if loss is None:
            loss = torch.nn.CrossEntropyLoss()
        self.__loss = loss

    def detach(self) -> "Model":
        return Model(labels=self._labels, model=self._model.eval(), threshold=self.threshold)

    def run_epoch(self, data: "Dataset") -> "float":
        m = self._model.train()
        values, labels = data
        device = torch.device("cuda:0")
        values = values.to(device)
        labels = labels.to(device)
        self.__optimizer.zero_grad()
        result = m(values)
        loss = self.__loss(result, labels)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    @classmethod
    def get_data(cls, root: "str", transform: "torchvision.transforms.Compose" = None,
                 **loader_options) -> "tuple[DataLoader, dict[str,int]]":
        if transform is None:
            transform = Model.get_transform()
        img_set = ImageFolder(root=root, transform=transform)
        return DataLoader(dataset=img_set, **loader_options), img_set.class_to_idx
