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

        data = np.concatenate([np.load("data/" + c + ".npy", mmap_mode='r') for c in classes])
        data = data.reshape((-1, 1, 28, 28))
        data = torch.from_numpy(data).float()
        self.data = data
        del data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx] - 128) / 128, 0

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

