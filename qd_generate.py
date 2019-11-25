import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.qd_model import Generator
from config import params

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

# Load the checkpoint file
state_dict = torch.load(args.load_path, map_location=device)


# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']


# Create the generator network.
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

c = np.linspace(-2, 2, 10).reshape(1, -1)
#c = np.repeat(c, 10, 0).reshape(-1, 1)
c = np.repeat(c, params['dis_c_dim'], 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(10*params['dis_c_dim'], 1, 1, 1, device=device)

# Continuous latent code.
c2 = torch.cat((c, zeros), dim=1)
c3 = torch.cat((zeros, c), dim=1)

#idx = np.arange(10).repeat(10)
idx = np.arange(params['dis_c_dim']).repeat(10)
#dis_c = torch.zeros(100, 10, 1, 1, device=device)
dis_c = torch.zeros(10*params['dis_c_dim'], params['dis_c_dim'], 1, 1, device=device)
dis_c[torch.arange(0, 10*params['dis_c_dim']), idx] = 1.0
# Discrete latent code.
#c1 = dis_c.view(100, -1, 1, 1)
c1 = dis_c.view(10*params['dis_c_dim'], -1, 1, 1)

#z = torch.randn(100, 62, 1, 1, device=device)
z = torch.randn(10*params['dis_c_dim'], params['num_z'], 1, 1, device=device)
print(z.shape)

# To see variation along c2 (Horizontally) and c1 (Vertically)
noise1 = torch.cat((z, c1, c2), dim=1)
# To see variation along c3 (Horizontally) and c1 (Vertically)
noise2 = torch.cat((z, c1, c3), dim=1)
print(noise1.shape, c1.shape, c2.shape, c3.shape)

# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()
# Display the generated image.
#fig = plt.figure(figsize=(10, 10))
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
#plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=params['dis_c_dim'], padding=2, normalize=True), (1,2,0)))
plt.show()

# Generate image.
with torch.no_grad():
    generated_img2 = netG(noise2).detach().cpu()
# Display the generated image.
#fig = plt.figure(figsize=(10, 10))
fig = plt.figure(figsize=(10, params['dis_c_dim']))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
#plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=params['dis_c_dim'], padding=2, normalize=True), (1,2,0)))
plt.show()