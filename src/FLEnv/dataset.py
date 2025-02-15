import os
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
import numpy as np
import re
import cv2 as cv



data_dir = os.path.join(os.getcwd(), 'normData')
#print(data_dir)
p = re.compile(r'\d+')

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
    def __init__(self, data_normal, data_pneumonia, transform, linear=False):
        self.data = data_normal + data_pneumonia
        self.indices_normal = np.arange(0, len(data_normal)).tolist()
        self.indices_pneumonia = np.arange(len(data_normal), len(data_pneumonia) + len(data_normal)).tolist()
        self.transform = transform
        self.linear = linear


    def __len__(self):
        return len(self.data)                         
                                                      
    def __getitem__(self, index):
        img = cv.imread(self.data[index], cv.IMREAD_GRAYSCALE)
        img = img / 255.
        img = self.transform(img)
        label = 0 if 'NORMAL' in self.data[index] else 1
        if self.linear:
            img = torch.flatten(img, start_dim=1)


        return img.float(), torch.tensor(label)
    
    def get_indices(self):
        return self.indices_normal, self.indices_pneumonia
    


def get_data():
    """Get Train and Test Dataset and apply minimal transformation"""

    train_images = os.listdir(os.path.join(data_dir, 'test'))
    val_images = os.listdir(os.path.join(data_dir, 'val'))
    test_images = os.listdir(os.path.join(data_dir, 'test'))

    return train_images, val_images, test_images


def prepare_dataset(num_partitions: int, batch_size: int, train_ratio: float = 0.9, linear: bool = False, equal_distribution: bool = True):
    '''Get the full Dataset consisting of train (for training clients), val (For validating server model), test (For testing server model)'''

    #basic transform
    tr = Compose([ToTensor(), Normalize((0, ), (1, )), Resize((224, 224))])

    #get dataset
    train_images , val_images, test_images = get_data()
    
    normal_files = sorted(os.listdir(os.path.join(data_dir, 'train', train_images[1])), key=lambda x: int(p.findall(x)[0]))
    normal_files = [os.path.join(data_dir, 'train', 'NORMAL', file) for file in normal_files]

    pneumonia_files = sorted(os.listdir(os.path.join(data_dir, 'train', train_images[0])), key=lambda x: int(p.findall(x)[0]))
    pneumonia_files = [os.path.join(data_dir, 'train', 'PNEUMONIA', file) for file in pneumonia_files]
    length_normal = len(os.listdir(os.path.join(data_dir, 'train', train_images[1])))
    length_pneumonia = len(os.listdir(os.path.join(data_dir, 'train', train_images[0])))
    client_train_loaders = []
    client_valid_loaders = []
    if equal_distribution:
        for i in range(num_partitions):
            len_normal_per_client = length_normal // num_partitions

            len_pneu_per_client = length_pneumonia // num_partitions

            normal_per_client = normal_files[len_normal_per_client * i: len_normal_per_client * i + len_normal_per_client]
            pneumonia_per_client = pneumonia_files[len_pneu_per_client * i: len_pneu_per_client * i + len_pneu_per_client]
            
            normal_per_client_train = normal_per_client[0:int(len(normal_per_client) * train_ratio)]
            pneumonia_per_client_train = pneumonia_per_client[0:int(len(pneumonia_per_client) * train_ratio)]
            
            normal_per_client_valid = normal_per_client[int(len(normal_per_client) * train_ratio):]
            pneumonia_per_client_valid = pneumonia_per_client[int(len(pneumonia_per_client) * train_ratio): ]

            dataset_train = X_Ray_Dataset(normal_per_client_train, pneumonia_per_client_train, transform=tr, linear=linear)
            dataset_valid = X_Ray_Dataset(normal_per_client_valid, pneumonia_per_client_valid, transform=tr, linear=linear)
            client_train_loaders.append(DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2))
            client_valid_loaders.append(DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=2))
    else:
        'We assume that there are equal amount of normal and pneumonia images in the dataset'
        samples_per_client = distribute_samples(total = length_normal, num_clients=num_partitions, min_per_client=800)
        prev = 0
        for i in samples_per_client:
            normal_per_client = normal_files[prev: i + prev]
            pneumonia_per_client = pneumonia_files[prev: i + prev]

            normal_per_client_train = normal_per_client[0:int(len(normal_per_client) * train_ratio)]
            pneumonia_per_client_train = pneumonia_per_client[0:int(len(pneumonia_per_client) * train_ratio)]
            
            normal_per_client_valid = normal_per_client[int(len(normal_per_client) * train_ratio):]
            pneumonia_per_client_valid = pneumonia_per_client[int(len(pneumonia_per_client) * train_ratio): ]
            dataset_train = X_Ray_Dataset(normal_per_client_train, pneumonia_per_client_train, transform=tr, linear=linear)
            dataset_valid = X_Ray_Dataset(normal_per_client_valid, pneumonia_per_client_valid, transform=tr, linear=linear)
            client_train_loaders.append(DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2))
            client_valid_loaders.append(DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=2))
            prev += i

    for i, (train_loader, valid_loader) in enumerate(zip(client_train_loaders, client_valid_loaders)):
        print(f'Client: {i + 1} train images: {len(train_loader.dataset)}, validation images: {len(valid_loader.dataset)}') 

    normal_files = os.listdir(os.path.join(data_dir, 'val', val_images[1]))
    normal_files = [os.path.join(data_dir, 'val', 'NORMAL', file) for file in normal_files]
    pneumonia_files = os.listdir(os.path.join(data_dir, 'val', val_images[0]))
    pneumonia_files = [os.path.join(data_dir, 'val', 'PNEUMONIA', file) for file in pneumonia_files]
    valid_dataset = X_Ray_Dataset(normal_files, pneumonia_files, transform=tr, linear=linear)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=2)

    print(f'Global validation set images: {len(valid_dataloader.dataset)}')

    normal_files = os.listdir(os.path.join(data_dir, 'test', test_images[1]))
    normal_files = [os.path.join(data_dir, 'test', 'NORMAL', file) for file in normal_files]
    pneumonia_files = os.listdir(os.path.join(data_dir, 'test', test_images[0]))
    pneumonia_files = [os.path.join(data_dir, 'test', 'PNEUMONIA', file) for file in pneumonia_files]
    test_dataset = X_Ray_Dataset(normal_files, pneumonia_files, transform=tr, linear=linear)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    print(f'Test set images: {len(test_dataloader.dataset)}')

    return client_train_loaders, client_valid_loaders, valid_dataloader, test_dataloader



def distribute_samples(total=7500, num_clients=3, min_per_client=1500) -> np.array:
    remaining = total - (num_clients * min_per_client)
    random_splits = np.random.dirichlet(np.ones(num_clients))
    random_splits *= remaining
    random_splits = np.round(random_splits).astype(int)  # Ensure integers

    # Step 3: Add back the minimum samples, one random split might be zero
    client_samples = random_splits + min_per_client

    # Adjust in case of rounding errors, add the diff, either positive or negative, to a random client
    while client_samples.sum() != total:
        diff = total - client_samples.sum()
        client_samples[np.random.choice(num_clients)] += diff

    return client_samples

