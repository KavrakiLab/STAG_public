import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

class Model_3DCNN(nn.Module):

    # num_filters=[64,128,256] or [96,128,128]
    def __init__(self, feat_dim=24, output_dim=1, num_filters=[64,128,256]):
        super(Model_3DCNN, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.num_filters = num_filters

        self.conv_block1 = self.__conv_layer_set__(self.feat_dim, self.num_filters[0], 7, 2, 3)
        self.res_block1 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)
        self.res_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)

        self.conv_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[1], 7, 3, 3)
        self.max_pool2 = nn.MaxPool3d(2)

        self.conv_block3 = self.__conv_layer_set__(self.num_filters[1], self.num_filters[2], 5, 2, 2)
        self.max_pool3 = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(2048, 100)
        torch.nn.init.normal_(self.fc1.weight, 0, 1)
        self.fc1_bn = nn.BatchNorm1d(100, affine=False)
        self.fc2 = nn.Linear(100, 1)
        torch.nn.init.normal_(self.fc2.weight, 0, 1)
        self.relu = nn.ReLU()
        self.drop=nn.Dropout(p=0.5)

    def __conv_layer_set__(self, in_c, out_c, k_size, stride, padding):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_c))
        return conv_layer

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.conv_block1(x)
        y = self.res_block1(x)
        y = y + x
        y = self.res_block2(y)
        y = y + x
        x = self.conv_block2(y)
        x = self.max_pool2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
