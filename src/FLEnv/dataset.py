import os
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
import numpy as np
from PIL import Image


data_dir = os.path.join(os.getcwd(), 'data', 'chest_xray')
print(data_dir)

class OverSampler(Sampler):
    def __init__(self, indices):
        self.indices_normal = indices[0]
        self.indices_pneu = indices[1]
        self.indices = self._compute_indices()

    def _compute_indices(self):
        num_normal = len(self.indices_normal)
        num_pneu = len(self.indices_pneu)
        difference = np.abs(num_normal - num_pneu)
        extra_samples = np.random.choice(self.indices_normal, difference, replace=True).tolist()
        self.indices_normal += extra_samples
        return [elem for pair in zip(self.indices_normal, self.indices_pneu) for elem in pair]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
    
    def print_length_of_indices(self):
        print(f'Number of indices normal: {len(self.indices_normal)}, Number of indices pneu: {len(self.indices_pneu)}')

class X_Ray_Dataset(Dataset):
    def __init__(self, data_normal, data_pneumonia, transform):
        self.data_normal = data_normal + data_pneumonia
        self.indices_normal = np.arange(0, len(data_normal)).tolist()
        self.indices_pneumonia = np.arange(len(data_normal), len(data_pneumonia) + len(data_normal)).tolist()
        self.transform = transform


    def __len__(self):
        return len(self.data)                         
                                                      
    def __getitem__(self, index):
        img = self.transform(Image.open(self.data[index]))
        label = 0 if 'NORMAL' in self.data[index] else 1
        return self.transform(img), torch.tensor([label])
    
    def get_indices(self):
        return self.indices_normal, self.indices_pneumonia

def get_data():
    """Get Train and Test Dataset and apply minimal transformation"""

    train_images = os.listdir(os.path.join(data_dir, 'test'))
    val_images = os.listdir(os.path.join(data_dir, 'val'))
    test_images = os.listdir(os.path.join(data_dir, 'test'))

    return train_images, val_images, test_images


def prepare_dataset(num_partitions: int, batch_size: int, train_ratio: float = 0.9):
    '''Get the full Dataset consisting of train (for training clients), val (For validating server model), test (For testing server model)'''

    #basic transform
    tr = Compose([ToTensor(), Normalize((0, ), (1, ))])

    #get dataset
    train_images , val_images, test_images = get_data()
    length_normal = len(os.listdir(os.path.join(data_dir, 'train', train_images[1])))
    len_normal_per_client = length_normal // num_partitions
    normal_files = os.listdir(os.path.join(data_dir, 'train', train_images[1]))


    length_pneumonia = len(os.listdir(os.path.join(data_dir, 'train', train_images[0])))
    len_pneu_per_client = length_pneumonia // num_partitions
    pneumonia_files = os.listdir(os.path.join(data_dir, 'train', train_images[0]))

    client_dataloaders = []
    for i in range(num_partitions):
        normal_per_client = normal_files[len_normal_per_client * i: len_normal_per_client * i + len_normal_per_client]
        pneumonia_per_client = pneumonia_files[len_pneu_per_client * i: len_pneu_per_client * i + len_pneu_per_client]
        
        normal_per_client_train = normal_per_client[0:int(len(normal_per_client) * train_ratio)]
        pneumonia_per_client_train = pneumonia_per_client[0:int(len(pneumonia_per_client) * train_ratio)]
        
        normal_per_client_valid = normal_per_client[int(len(normal_per_client) * train_ratio):]
        pneumonia_per_client_valid = pneumonia_per_client[int(len(pneumonia_per_client) * train_ratio): ]

        dataset_train = X_Ray_Dataset(normal_per_client_train, pneumonia_per_client_train, transform=tr)
        train_sampler = OverSampler(dataset_train.get_indices())
        #train_sampler.print_length_of_indices()
        dataset_valid = X_Ray_Dataset(normal_per_client_valid, pneumonia_per_client_valid, transform=tr)
        client_dataloaders.append(DataLoader(dataset_train, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=2), DataLoader(dataset_valid, batch_size=batch_size, num_workers=2))


    normal_files = os.listdir(os.path.join(data_dir, 'valid', val_images[1]))
    pneumonia_files = os.listdir(os.path.join(data_dir, 'valid', val_images[0]))
    valid_dataset = X_Ray_Dataset(normal_files, pneumonia_files)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, num_workers=2)

    normal_files = os.listdir(os.path.join(data_dir, 'test', test_images[1]))
    pneumonia_files = os.listdir(os.path.join(data_dir, 'test', test_images[0]))
    test_dataset = X_Ray_Dataset(normal_files, pneumonia_files)
    test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=2)

    return client_dataloaders, valid_dataloader, test_dataloader












