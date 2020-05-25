import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_off
import numpy as np
import scipy.io as sio

# the name of directory of off models and corr point index
model_dir = 'data_mesh'
# the name of directory of wavelets and input descriptors
data_dir = 'data_wavelet'


class SCAPE_WAVELET(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """


    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(SCAPE_WAVELET, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']


    def process(self):
        path = osp.join(self.root, model_dir, 'mesh{0:03d}.off')
        txt_path = osp.join(self.root, data_dir, 'mesh{0:03d}.txt')

        train_list = []
        test_list = []
        for i in range(0,72):
            if i == 51:
                continue
            data = read_off(path.format(i))
            data.y = torch.tensor([i % 10], dtype=torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data.x = torch.tensor(np.loadtxt(txt_path.format(i)), dtype=torch.float32)

            Winn = sio.loadmat(txt_path.format(i)[:-3] + 'mat')
            data.V = torch.tensor(Winn['V'], dtype=torch.float32)
            data.A = torch.tensor(Winn['A'], dtype=torch.float32)
            data.D = torch.tensor(Winn['D'], dtype=torch.float32)
            data.clk = torch.tensor(Winn['clk'], dtype=torch.float32)

            data.name = int(path.format(i).split('/')[-1].split('.')[0][4:7])
            data.flag = 0
            data.map = torch.tensor(np.loadtxt(path.format(i)[:-3] + 'vts') - 1, dtype=torch.long)

            if i < 62 and i != 51:
                train_list.append(data)
            elif i != 51:
                test_list.append(data)

        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(test_list), self.processed_paths[1])
