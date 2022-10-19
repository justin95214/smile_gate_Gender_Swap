import os
from shutil import copyfile



gender_map = {}

with open('./CelebA-HQ-attribute.txt') as f:
    lines = f.readlines()

    a =lines[1].split()
    print(a[0], a[20])

    for line in lines[2:]:
        
        splited = line.strip().split()
        print(f'test : {splited[0]} {splited[21]} ')
        file_name, male = splited[0], splited[21]
        file_name = file_name
        print(file_name)
        gender_map[file_name] = male

print(f'There are {len(set(gender_map.values()))} classes.')
print(f'There are {len(gender_map.keys())} images.')



source_root = './256_img/CelebAMask-HQ/resize_img_256/'
target_root = './gender_dataset_256__/'
file_list = os.listdir(source_root)




for file in file_list:
    print(file)
    gender = gender_map[file]
    if gender == '1':
        gender = 'male'
    else:
        gender = 'female'
    source = os.path.join(source_root, file)
    print(source[:10])
    target = os.path.join(target_root, gender, file)
    if not os.path.exists(os.path.join(target_root, gender)):
        os.makedirs(os.path.join(target_root, gender))
    copyfile(source, target)



folder_root = './gender_dataset_256__/'
folder_list = os.listdir(folder_root)

male_cnt = 0
female_cnt = 0

train_images = 0
test_images = 0
train_ratio = 0.65

for folder in folder_list:
    file_list = os.path.join(folder_root, folder)
    file_list = os.listdir(file_list)
    if folder == 'male':
        male_cnt += len(file_list)
    else:
        female_cnt += len(file_list)
    num_train = int(train_ratio * len(file_list))
    for file in file_list[:num_train]:
        train_images += 1
        source = os.path.join(folder_root, folder, file)
        target = os.path.join(folder_root, 'train', folder, file)
        if not os.path.exists(os.path.join(folder_root, 'train', folder)):
            os.makedirs(os.path.join(folder_root, 'train', folder))
        os.rename(source, target)
    for file in file_list[num_train:]:
        test_images += 1
        source = os.path.join(folder_root, folder, file)
        target = os.path.join(folder_root, 'test', folder, file)
        if not os.path.exists(os.path.join(folder_root, 'test', folder)):
            os.makedirs(os.path.join(folder_root, 'test', folder))
        os.rename(source, target)

print(f'There are {male_cnt} male images.')
print(f'There are {female_cnt} female images.')
print(f'There are {train_images} train images.')
print(f'There are {test_images} test images.')
