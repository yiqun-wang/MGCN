import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import uniform, ones, glorot, normal


class MGConv(torch.nn.Module):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Number of scales.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, bias=True, number=0):
        super(MGConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        self.number = number
        self.K = K

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # normal(self.weight, 0, 0.1)
        glorot(self.weight)
        # ones(self.weight)
        if self.bias is not None:
            ones(self.bias)

    def forward(self, x, Win):
        for i in range(self.weight.size(0)):
            if i == 0:
                out = torch.matmul(x, self.weight[0])
            else:
                WWW = torch.matmul(torch.t(Win[i-1]), x)
                out += torch.matmul(WWW, self.weight[i])

        torch.cuda.empty_cache()

        return out


    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))
