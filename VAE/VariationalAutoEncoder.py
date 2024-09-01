#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
# %%
dataset_path = "~/datasets"
cuda = True
device = torch.device("cuda" if cuda else "cpu")
# %%
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 200
lr = 1e-3
epochs = 30
# %%
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# %%
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])
kwargs = {"num_workers" : 0, "pin_memory" : True}

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# %% VAE
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder,self).__init__()
        self.fc_input = nn.Linear(input_dim,hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim,latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyRelu = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self,x):
        h_ = self.LeakyRelu(self.fc_input(x))
        h_ = self.LeakyRelu(self.fc_input2(h_))
        mean = self.fc_mean(h_)
        var = self.fc_var(h_)
        return mean, var
# %%
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder,self).__init__()
        self.fc_hidden = nn.Linear(input_dim,hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc_output = nn.Linear(hidden_dim,output_dim)

        self.LeakyRelu = nn.LeakyReLU(0.2)
    def forward(self, x):
        h = self.LeakyRelu(self.fc_hidden(x))
        h = self.LeakyRelu(self.fc_hidden2(h))
        x_hat = torch.sigmoid(self.fc_output(h))
        return x_hat
# %%
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var
# %%
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim,latent_dim=latent_dim)
decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
Model = Model(Encoder=encoder,Decoder=decoder).to(device)
# %%
from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(Model.parameters(), lr=lr)
# %%
print("Start training VAE...")
Model.train()
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = Model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
print("Finish!!")
# %%
Model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(device)
        
        x_hat, _, _ = Model(x)

        break
#%%
def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())
#%%
show_image(x, idx=0)
#%%
show_image(x_hat, idx=0)
#%%
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(device)
    generated_images = decoder(noise)
#%%
save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')
#%%
show_image(generated_images, idx=12)
#%%
show_image(generated_images, idx=0)
#%%
show_image(generated_images, idx=1)
#%%
show_image(generated_images, idx=10)
#%%
show_image(generated_images, idx=20)
#%%
show_image(generated_images, idx=50)