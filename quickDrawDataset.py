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
        for c in classes[1:]:
            data = np.append(data, np.load("data/" + c + ".npy"), axis=0)

        data = data.reshape((data.shape[0], 28, 28))

        self.data = torch.from_numpy(data).byte()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

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

