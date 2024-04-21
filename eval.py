import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np

# Custom Dataset for Geophysical Data Fields
class GeophysicalTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
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

# Setup for Distributed Testing
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Testing function
def test_model(rank, world_size, data_dir):
    setup(rank, world_size)

    # Load the model (assume it is already trained and loaded)
    generator = UNetGenerator(in_channels=3).to(rank)
    generator = DDP(generator, device_ids=[rank])

    # Load the dataset
    dataset = GeophysicalTestDataset(data_dir, transform=ToTensor())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)

    generator.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No gradients needed
        for i, batch in enumerate(dataloader):
            real_data = batch['image'].to(rank)
            # Forward pass to generate outputs
            output = generator(real_data)
            # Here you can save or evaluate the output as needed
            print(f"Rank {rank}, Batch {i}, Sample Output Shape: {output.shape}")

    cleanup()

# Main function to spawn testing processes
def main():
    world_size = 4  # Adjust based on the number of GPUs available
    torch.multiprocessing.spawn(test_model, args=(world_size, './test_data'), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
