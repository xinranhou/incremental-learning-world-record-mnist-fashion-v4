from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs


class Fashion(data.Dataset):  #https://github.com/zalandoresearch/fashion-mnist
    download_folder = 'download'
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, pickup_group=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if not self.__make_sure_processed():
            raise RuntimeError('Dataset not found. Please directly download to %s or make sure that program can download form %s.'%(os.path.join(self.root, self.download_folder), self.urls))
        if self.train:
            self.train_data, self.train_labels = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
            self.train_data, self.train_labels = self.__pickup_group(pickup_group, self.train_data, self.train_labels)
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data, self.test_labels = self.__pickup_group(pickup_group, self.test_data, self.test_labels)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __pickup_group(self, pickup_group, data, labels):
        if pickup_group:
            data_pickup = []
            labels_pickup = []
            for i in range(len(labels)):
                if labels[i].numpy() in pickup_group:
                    data_pickup.append(data[i])
                    labels_pickup.append(labels[i])
            return data_pickup, labels_pickup
        else:
            return data, labels

    def __make_sure_processed(self):
        if os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)):
            return True
        else:
            if self.__make_sure_unziped():
                return self.__unzip_raw_to_processed()
            else:
                return False
                
    def __make_sure_unziped(self):
        from six.moves import urllib
        import gzip
        for url in self.urls:
            raw_file = os.path.join(self.root, self.raw_folder, url.rpartition('/')[2]).replace('.gz', '')
            if not os.path.exists(raw_file):
                download_file = os.path.join(self.root, self.download_folder, url.rpartition('/')[2])
                if not os.path.exists(download_file):
                    os.makedirs(os.path.join(self.root, self.download_folder))
                    with open(download_file, 'wb') as f:
                        f.write(urllib.request.urlopen(url).read())
                    if not os.path.exists(download_file):
                        return False  
                local_raw_folder = os.path.join(self.root, self.raw_folder)              
                if not os.path.exists(local_raw_folder): 
                    os.makedirs(local_raw_folder)
                with open(raw_file, 'wb') as out_f, gzip.GzipFile(download_file) as zip_f:
                    out_f.write(zip_f.read())   
                    if not os.path.exists(raw_file):
                        return False
                    #os.unlink(file_path)
        return True      

    def __unzip_raw_to_processed(self):
        local_processed_folder = os.path.join(self.root, self.processed_folder)              
        if not os.path.exists(local_processed_folder): 
            os.makedirs(local_processed_folder)
        training_set = (
            self.__read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            self.__read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            self.__read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            self.__read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        return True

    def __get_int(self, b):
        return int(codecs.encode(b, 'hex'), 16)

    def __read_label_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.__get_int(data[:4]) == 2049
            length = self.__get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            return torch.from_numpy(parsed).view(length).long()

    def __read_image_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.__get_int(data[:4]) == 2051
            length = self.__get_int(data[4:8])
            num_rows = self.__get_int(data[8:12])
            num_cols = self.__get_int(data[12:16])
            images = []
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            return torch.from_numpy(parsed).view(length, num_rows, num_cols)
