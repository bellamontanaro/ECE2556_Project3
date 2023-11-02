import os

# split list of input and label paths into train, val, test
def data_split(root_path, exp_name, split_ratio=0.8):
    train_dict, val_dict = {}, {}

    img_path = os.listdir(os.path.join(root_path, 'imgs'))
    label_path = os.listdir(os.path.join(root_path, 'masks'))
    if exp_name == 'covid100':
        img_path.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        label_path.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    elif exp_name == 'covid20':
        img_path.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
        label_path = img_path
        # label_path.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
    elif exp_name == 'covid_mos_med':
        img_path.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
        label_path.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
        
    img_paths = [os.path.join(root_path, 'imgs', path) for path in img_path]
    label_paths = [os.path.join(root_path, 'masks', path) for path in label_path]
    num_train = int(len(img_paths) * split_ratio)
    train_dict['imgs'] = img_paths[:num_train]
    train_dict['masks'] = label_paths[:num_train]
    val_dict['imgs'] = img_paths[num_train:]
    val_dict['masks'] = label_paths[num_train:]
    
    return train_dict, val_dict