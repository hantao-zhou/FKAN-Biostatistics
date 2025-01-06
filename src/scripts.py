import warnings
warnings.filterwarnings("ignore")
import os
#from PIL import Image
import numpy as np
import albumentations as A
import cv2 as cv
from tqdm import tqdm
from sklearn.model_selection import train_test_split

'''Script with some helper function to make life easier'''

cwd = os.getcwd()
data_path = os.path.join(cwd, 'data', 'chest_xray')
def change_image_names():
    data_dir = os.path.join(os.getcwd(), 'data', 'chest_xray')
    test_dir = os.path.join(data_dir, 'test')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    dirs = [test_dir, val_dir, train_dir]

    for dir in dirs:
        normal_dir_name = os.path.join(dir, 'NORMAL')
        normal_dir = os.listdir(normal_dir_name)
        
        for i, file in tqdm(enumerate(normal_dir)):
                img = cv.imread(os.path.join(normal_dir_name, file), cv.IMREAD_GRAYSCALE)
                if img is not None:
                    cv.imwrite(os.path.join(normal_dir_name, f'NORMAL_{i}.jpeg'), img)
                    os.remove(os.path.join(normal_dir_name, file))
                else: print(os.path.join(os.path.join(normal_dir_name, file)))

        
        pneumonia_dir_name = os.path.join(dir, 'PNEUMONIA')
        pneumonia_dir = os.listdir(pneumonia_dir_name)

        for i, file in tqdm(enumerate(pneumonia_dir)):
                img = cv.imread(os.path.join(pneumonia_dir_name, file), cv.IMREAD_GRAYSCALE)
                if img is not None:
                    cv.imwrite(os.path.join(pneumonia_dir_name, f'PNEUMONIA_{i}.jpeg'), img)
                    os.remove(os.path.join(pneumonia_dir_name, file))
                else: print(os.path.join(pneumonia_dir_name, file))
                    


def create_synthetic_images():
    normal_path = os.path.join(data_path, 'train', 'NORMAL')
    pneumonia_path = os.path.join(data_path, 'train', 'PNEUMONIA')

    len_pneumonia = len(os.listdir(pneumonia_path))
    normal_images = os.listdir(normal_path)
    normal_images = [os.path.join(data_path, 'train', 'NORMAL', file) for file in normal_images]
    len_normal = len(normal_images)
    difference = np.abs(len_normal - len_pneumonia)

    
    additional_images = np.random.choice(normal_images, difference, replace=True).tolist()

    

    print(f'Length normal images {len(normal_images)}, additional images {len(additional_images)}, Length Pneumonia images {len_pneumonia}')

    for i, img_path in tqdm(enumerate(additional_images)):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            height, width = img.shape
            transform = A.Compose([
                A.Rotate(limit=20),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.5),
                A.Resize(height=height, width=width) 
            ])     

            img = transform(image=img)['image']
            cv.imwrite(os.path.join(normal_path, f'NORMAL_{len_normal + i + 1}.jpeg'), img)

        else: print('ERROR')


def normalize():
    folders = ['train', 'val', 'test']

    for folder in folders:
        normal_images = os.listdir(os.path.join(data_path, 'chest_xray', folder, 'NORMAL'))
        normal_images = [os.path.join(data_path, 'chest_xray', folder, 'NORMAL', file) for file in normal_images]

        pneumonia_images = os.listdir(os.path.join(data_path, 'chest_xray', folder, 'PNEUMONIA'))
        pneumonia_images = [os.path.join(data_path, 'chest_xray', folder, 'PNEUMONIA', file) for file in pneumonia_images]

        normal_images.extend(pneumonia_images) 
        #print(normal_images)

        for i, img_path in tqdm(enumerate(normal_images)):
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = img / 255.
                cv.imwrite(os.path.join(img_path), img)
            else: print(img_path)
            


def calculate_mean_std():
    normal_images = os.listdir(os.path.join(data_path, 'train', 'NORMAL'))
    normal_images = [os.path.join(data_path, 'train' ,'NORMAL', file) for file in normal_images]

    pneumonia_images = os.listdir(os.path.join(data_path, 'train', 'PNEUMONIA'))
    pneumonia_images = [os.path.join(data_path, 'train', 'PNEUMONIA', file) for file in pneumonia_images]

    normal_images.extend(pneumonia_images)

    print('computing mean')
    mean = 0
    for img_path in tqdm(normal_images):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            img = img.astype(float) / 255.
            mean += img.mean()

    mean = mean / len(normal_images)

    print('computing std')
    std = 0
    for img_path in tqdm(normal_images):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            img = img.astype(float) / 255.
            std += ((img - mean)**2).sum() / (img.shape[0] * img.shape[1])

    std = np.sqrt(std / len(normal_images))
    print(f'Mean: {mean}')
    print(f'std: {std}')

    #out
    #Mean: 0.4880792792941513
    #Std: 0.23403704091782646

def download_dataset():
    import kagglehub
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

    print("Path to dataset files:", path)

def create_train_test_split():
    normal_path = os.path.join(data_path, 'train', 'NORMAL')
    pneu_path = os.path.join(data_path, 'train', 'PNEUMONIA')

    normal_images = os.listdir(normal_path)
    normal_images = [os.path.join(normal_path, file) for file in normal_images]
    pneu_images = os.listdir(pneu_path)
    pneu_images = [os.path.join(pneu_path, file) for file in pneu_images]


    X_train, X_test = train_test_split(normal_images, test_size=0.2, random_state=42)

    for x_test in X_test:
        x_test_filename = x_test.split('/')[-1]
        os.rename(x_test, os.path.join(data_path, 'val', 'NORMAL', x_test_filename))

    X_train, X_test = train_test_split(pneu_images, test_size=0.2, random_state=42)


    for x_test in X_test:
        x_test_filename = x_test.split('/')[-1]
        os.rename(x_test, os.path.join(data_path, 'val', 'PNEUMONIA', x_test_filename))
    



#download_dataset()
#create_train_test_split()
#change_image_names()
#create_synthetic_images()
calculate_mean_std()

