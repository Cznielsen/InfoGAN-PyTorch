from torch.utils.data import Dataset
import os
import grequests
import requests
import numpy as np
import torch
import convert

from config import params

class QuickDrawDataset(Dataset):
    def __init__(self, classes, download=False, from_src=False):
        if download:
            self.download(classes, from_src)

        data = np.concatenate([np.load("data/" + c + ".npy", mmap_mode='r')[: params["num_img"]] for c in classes])
        data = data.reshape((-1, 1, 28, 28))
        data = torch.from_numpy(data).float()
        self.data = data
        del data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx] - 128) / 128, 0

    def download(self, classes, from_src):
        urls = []
        if from_src:
            url = 'https://storage.googleapis.com/quickdraw_dataset/full/binary/'
        else:
            url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
        try:
            os.mkdir("data")
        except FileExistsError:
            pass
        try:
            os.mkdir("temp")
        except FileExistsError:
            pass

        ext = '.npy'
        folder = 'data/'
        if from_src:
            ext = '.bin'
            folder = 'temp/'

        for c in classes:
            if not os.path.exists(os.path.join(folder, c + ext)):
                urls.append(url + requests.utils.quote(c) + ext)

        rs = (grequests.get(u, allow_redirects=True) for u in urls)
        responses = grequests.map(rs)
        for r in responses:
            with open(folder + requests.utils.unquote(r.url.split("/")[-1]), "wb") as f:
                f.write(r.content)

        if from_src:
            for r, d, files in os.walk('temp/'):
                for f in files:
                    images = []
                    if not os.path.exists(os.path.join('data/', str(f).split('.')[0] + '.npy')):
                        for res in convert.unpack_drawings('temp/'+f):
                            if res['recognized']:
                                images.append(res['image'])
                        np.save('data/' + str(f).split('.')[0], convert.vector_to_raster(images))
