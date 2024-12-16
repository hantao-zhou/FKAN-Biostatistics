import os
'''Script with some helper function to make life easier'''


def change_image_names():
    data_dir = os.path.join(os.getcwd(), 'data', 'chest_xray')
    test_dir = os.path.join(data_dir, 'test')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    dirs = [test_dir, train_dir, val_dir]

    for dir in dirs:
        normal_dir_name = os.path.join(dir, 'NORMAL')
        normal_dir = os.listdir(normal_dir_name)

        for i, file in enumerate(normal_dir):
            os.rename(os.path.join(normal_dir_name, file), os.path.join(normal_dir_name, f'NORMAL_{i}.jpeg'))
            
        pneumonia_dir_name = os.path.join(dir, 'PNEUMONIA')
        pneumonia_dir = os.listdir(pneumonia_dir_name)

        for i, file in enumerate(pneumonia_dir):
            os.rename(os.path.join(pneumonia_dir_name, file), os.path.join(pneumonia_dir_name, f'PNEUMONIA_{i}.jpeg'))
        


