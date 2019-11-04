from torch.utils.data import Dataset
import os
import grequests
import requests

class QuickDrawDataset(Dataset):
    def __init__(self, classes, transfrom=None, download=False):

        if download:
            self.download(classes)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

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

