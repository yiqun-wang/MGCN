import os
import os.path as osp
import argparse
import numpy as np
import gc
import torch
import torch.nn.functional as F
from datasets.scape_wavelet import SCAPE_WAVELET
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DataListLoader
from nn.losses import loss_HardNet
from nn.mgconv import MGConv

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_cpu', '--uc', dest='use_cpu',default=False, action='store_true',
                    help='bool value, use gpu or not')
parser.add_argument('--gpuid', '-g', default='0', type=str, metavar='N',
                    help='GPU id to run')

parser.add_argument('--learning_rate_softmax', '--lrs', default=0.001, type=float,
                    help='the learning rate')
parser.add_argument('--weight_decay_softmax', '--wds', default=1e-4, type=float,
                    help='the weight decay')
parser.add_argument('--learning_rate_hardloss', '--lrh', default=5e-5, type=float,
                    help='the learning rate')
parser.add_argument('--weight_decay_hardloss', '--wdh', default=5e-5, type=float,
                    help='the weight decay')

parser.add_argument('--epoch_softmax', '--es', default=200, type=int,metavar='N',
                    help='the number of training iterations with softmax loss')
parser.add_argument('--epoch_hardloss', '--eh', default=100, type=int,metavar='N',
                    help='the number of training iterations with hardnet loss')

parser.add_argument('--input_desc_dims', '--idd', default=128, type=int,
                    help='the number of dimensions in input descriptors')
parser.add_argument('--output_desc_dims', '--odd', default=256, type=int,
                    help='the number of dimensions in output descriptors')

parser.add_argument('--wavelet_scales', '--ws', default=16, type=int,
                    help='the number of wavelet scales.')
parser.add_argument('--n_corr_points', default=5000, type=int,
                    help='the number of corresponding points')

parser.add_argument('--save_freq', '--sf', default=100, type=int,
                    help=r'save the current trained model every {save_freq} iterations')


parser.add_argument('--saving_name', '--sn', default='mgcn_scape', type=str,
                    help='the name of trained models and the name of directory to save output descriptors')
parser.add_argument('--loading_name', '--ln', default='mgcn_scape-300', type=str,
                    help='the name of loaded model and the name of directory to generate descriptors using the loaded model')

parser.add_argument('--load', '-l', dest='load',default=False, action='store_true',
                    help='bool value, load variables from saved model or not')
parser.add_argument('--generate_desc', '--gd', dest='generate_desc',default=False, action='store_true',
                    help='bool value, generating descriptors using loaded model')

args = parser.parse_args()

USE_GPU = not args.use_cpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.benchmark = True

LOAD = args.load # True
GEN = args.generate_desc # True
TRIPLET = False
EPOCH_softmax = args.epoch_softmax
EPOCH_hardloss = args.epoch_hardloss
K = args.wavelet_scales + 1
SAVE_NAME = args.saving_name
CPOINT_NAME = args.loading_name
LEARNING_RATE=args.learning_rate_softmax
WEIGHT_DECAY=args.weight_decay_softmax

path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', 'Scape')
path_output = osp.join(osp.abspath('.'), 'outputs', SAVE_NAME)
if not os.path.exists(path_output):
    os.makedirs(path_output)
LOG_FOUT = open(path_output + '/log.out', 'w')

pre_transform = T.FaceToEdge()
train_dataset = SCAPE_WAVELET(path, True, None, pre_transform)
test_dataset = SCAPE_WAVELET(path, False, None, pre_transform)
train_loader =  DataLoader(train_dataset, batch_size=1, shuffle=True)
train_loader_tri =  DataListLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
d = test_dataset[0]
d.num_nodes = args.n_corr_points


class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        # self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1)) # + self.eps
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


class FMaxMin(torch.nn.Module):
    def __init__(self):
        super(FMaxMin,self).__init__()
    def forward(self, x):
        min = torch.min(x, dim=0)[0]
        max = torch.max(x, dim=0)[0]
        ran = max - min
        x= (x - min.unsqueeze(0).expand_as(x)) / ran.unsqueeze(0).expand_as(x)
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = MGConv(args.input_desc_dims, 96, K=K, bias=False)
        self.conv2 = MGConv(96, 96, K=K, bias=False)
        self.conv3 = MGConv(96, 96, K=K, bias=False)
        self.conv4 = MGConv(96, 96, K=K, bias=False)
        self.conv5 = MGConv(96, 96, K=K, bias=False)
        self.conv6 = MGConv(96, 128, K=K, bias=False)
        self.fc1 = torch.nn.Linear(128, args.output_desc_dims)
        self.fc2 = torch.nn.Linear(args.output_desc_dims, d.num_nodes)

    def forward(self, data):
        x, V, A, D, clk = data.x, data.V, data.A, data.D, data.clk

        list = []
        for k in range(31, -1, -(32//(K-1))):
            Win = torch.mm(torch.mm(torch.mm(V, torch.diag(clk[:, k])), torch.t(V)), torch.diag(torch.squeeze(A))) ** 2
            Win = torch.nn.functional.normalize(Win, p=2, dim=0) ** 2
            list.append(Win)
        Win = torch.stack(list, dim=0)
        torch.cuda.empty_cache()

        x = FMaxMin()(x)
        x = self.conv1(x, Win)
        x = F.elu(x)
        x = FMaxMin()(x)
        x = self.conv2(x, Win)
        x = F.elu(x)
        x = FMaxMin()(x)
        x = self.conv3(x, Win)
        x = F.elu(x)
        x = FMaxMin()(x)
        x = self.conv4(x, Win)
        x = F.elu(x)
        x = FMaxMin()(x)
        x = self.conv5(x, Win)
        x = F.elu(x)
        x = FMaxMin()(x)
        x = self.conv6(x, Win)
        x = F.elu(x)
        x = FMaxMin()(x)
        des = F.elu(self.fc1(x))
        x = F.dropout(des, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), des


device_type = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
device = torch.device(device_type)
model = Net().to(device)
target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY) # 0.01  softmax: lr=0.001, weight_decay=5e-4 triplet: lr=0.0001, weight_decay=5e-5  cheb lr=0.0005, weight_decay=1e-4


def train(epoch):
    model.train()

    if TRIPLET:
        loss_value = 0.0
        count = 0
        for data in train_loader_tri:
            if len(data)==2:
                data2 = data.copy()
                data_a = data2[0]  # Batch.from_data_list([])
                data_p = data2[1]  # Batch.from_data_list([data[1]])
                out_a = model(data_a.to(device))[1][data_a.map, :]
                out_p = model(data_p.to(device))[1][data_p.map, :]
                optimizer.zero_grad()
                loss = loss_HardNet(out_a, out_p, anchor_swap=True) # =False , batch_reduce='average')
                loss.backward()
                optimizer.step()
                loss_value = loss_value + loss.item()
                count = count + 1
                gc.collect()

        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_value / count))
        LOG_FOUT.write('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_value / count) + '\n')
        LOG_FOUT.flush()

    else:
        loss_value = 0.0
        count = 0
        flag = True
        for data in train_loader:
            optimizer.zero_grad()
            if flag:
                x, des = model(data.to(device))  # , nloss
                loss=F.nll_loss(x[data.map, :], target) # + 1e-2*nloss
                loss.backward()
            optimizer.step()
            loss_value = loss_value + loss.item()
            count = count + 1
            gc.collect()

        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_value / count))
        LOG_FOUT.write('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_value / count) + '\n')
        LOG_FOUT.flush()


if LOAD:
    model.load_state_dict(torch.load(osp.join(osp.abspath('.'), 'checkpoints', CPOINT_NAME + '.pth'), map_location=device_type))
EPOCH = EPOCH_softmax + EPOCH_hardloss
for epoch in range(1, EPOCH+1):
    if GEN:
        path_gen = osp.join(osp.abspath('.'), 'outputs', CPOINT_NAME)
        if not os.path.exists(path_gen):
            os.makedirs(path_gen)
        txt_path = osp.join(path_gen, 'mesh{0:03d}.txt')
        for data in test_loader:
            desc = model(data.to(device))[1]
            descriptor = desc.cpu().detach().numpy()
            i = int(data.name.cpu().detach())
            np.savetxt(txt_path.format(i), descriptor, fmt='%.6e')
            torch.cuda.empty_cache()
            gc.collect()
        break
    if epoch > EPOCH_softmax:
        TRIPLET = True
        LEARNING_RATE = args.learning_rate_hardloss
        WEIGHT_DECAY = args.weight_decay_hardloss
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train(epoch)
    if epoch % args.save_freq == 0:
        torch.save(model.state_dict(), osp.join(osp.abspath('.'), 'checkpoints', SAVE_NAME + str(-epoch) + '.pth'))
        txt_path = osp.join(osp.abspath('.'), 'outputs', SAVE_NAME, 'mesh{0:03d}.txt')
        for data in test_loader:
            desc = model(data.to(device))[1]
            descriptor = desc.cpu().detach().numpy()
            i = int(data.name.cpu().detach())
            np.savetxt(txt_path.format(i), descriptor, fmt='%.6e')

LOG_FOUT.close()
