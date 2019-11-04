from torch.utils.data import Dataset
import os
import grequests
import requests
import numpy as np
import torch


class QuickDrawDataset(Dataset):
    def __init__(self, classes, download=False):

        if download:
            self.download(classes)

        data = np.load("data/" + classes[0] + ".npy")
        labels = np.full(data.shape[0], 0)
        for i in range(1, len(classes)):
            temp = np.load("data/" + classes[i] + ".npy")
            data = np.append(data, temp, axis=0)
            labels = np.append(labels, np.full(temp.shape[0], i))

        data = data.reshape((-1, 1, 28, 28))
        data = torch.from_numpy(data).float()
        data[:] = (data[:] - 128) / 128
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def download(self, classes):
        urls = []
        url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
        try:
            os.mkdir("data")
        except FileExistsError:
            pass

        for c in classes:
            if not os.path.exists(os.path.join("data", c + ".npy")):
                urls.append(url + requests.utils.quote(c) + ".npy")

        rs = (grequests.get(u, allow_redirects=True) for u in urls)
        responses = grequests.map(rs)
        for r in responses:
            with open("data/" + requests.utils.unquote(r.url.split("/")[-1]), "wb") as f:
                f.write(r.content)

