import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import numpy as np

# Custom Dataset for Geophysical Data Fields
class GeophysicalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.filenames[idx])
        image = np.load(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Transform for the dataset
class ToTensor:
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))  # Assuming image is HxWxC
        return {'image': torch.from_numpy(image).float()}

# Setting up Distributed Training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Example Distributed Training Loop
def train(rank, world_size, epochs, data_dir):
    setup(rank, world_size)

    # Model, Loss, and Optimizer
    generator = UNetGenerator(in_channels=3).to(rank)
    discriminator = ResidualAttentionDiscriminator(in_channels=3).to(rank)
    gan_loss = GANLoss()

    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Data Loader
    dataset = GeophysicalDataset(data_dir, transform=ToTensor())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            real_data = batch['image'].to(rank)
            noise = torch.randn(real_data.size(0), 100, 1, 1, device=rank)  # Noise dimension (100) is arbitrary
            fake_data = generator(noise)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_logits = discriminator(real_data)
            fake_logits = discriminator(fake_data.detach())
            d_loss = gan_loss.discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_logits = discriminator(fake_data)
            g_loss = gan_loss.generator_loss(fake_logits, real_data, fake_data)
            g_loss.backward()
            optimizer_g.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    cleanup()

# Main function to spawn training processes
def main():
    world_size = 4  # Number of GPUs
    torch.multiprocessing.spawn(train, args=(world_size, 50, './data'), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
