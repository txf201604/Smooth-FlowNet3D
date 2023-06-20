import numpy as np
import torch 
import torch.nn as nn 

import torch.nn.functional as F
from pointnet2 import pointnet2_utils

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x
    
    
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points) # [B, npoint, nsample, C+D]

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) #BxWxKxN
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1) #BxNxWxK * BxNxKxC => BxNxWxC -> BxNx(W*C)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvK(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvK, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.agg = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.linear = nn.Linear(out_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        C = points.shape[1]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # [B, npoint, nsample, C+D]
        kernel = self.kernel(new_points.permute(0, 3, 1, 2))
        kernel = self.relu(kernel) # B, Out*In, N, S

        aggregation = torch.matmul(input = kernel.permute(0, 2, 1, 3), other = new_points.permute(0, 1, 2, 3))
        
        aggregation = self.relu(self.agg(aggregation.permute(0, 3, 1, 2))).squeeze(1)
        # B, N, S, C

        new_points = self.linear(aggregation)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class SetAbstract(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, mlp2=None, use_leaky = True):
        super(SetAbstract, self).__init__()
        self.npoint = npoint
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
                last_channel = out_channel
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  self.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointAtten(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointAtten, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # B, N, S, C

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) # B, 16, S, N

        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1) # B, N, C, S x B, N, S, 16

        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost


class CrossLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayer,self).__init__()
        # self.fe1_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel,in_channel], pooling=pooling, corr_func=corr_func)
        # self.fe2_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel, out_channel], pooling=pooling, corr_func=corr_func)
        # self.flow = nn.Conv1d(out_channel, 3, 1)

        self.nsample = nsample
        self.bn = bn
        self.mlp1_convs = nn.ModuleList()
        if bn:
            self.mlp1_bns = nn.ModuleList()
        last_channel = in_channel  * 2 + 3
        for out_channel in mlp1:
            self.mlp1_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp1_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2 is not None:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            last_channel = mlp1[-1] * 2 + 3
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, mlp_convs, mlp_bns):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(mlp_convs):
            if self.bn:
                bn = mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.mlp2_convs, self.mlp2_bns if self.bn else None)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross_t2(feat2_new)

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        # 3 nearest neightbor & use 1/dist as the weights
        knn_idx = knn_point(3, xyz1_to_2, xyz2) # group flow 1 around points 2
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3

        # 3 nearest neightbor from dense around sparse & use 1/dist as the weights the same
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow 

class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])

class SceneFlowEstimatorResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        new_points = torch.cat([feats, cost_volume], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return new_points, flow


class SA_Layer(nn.Module):
    def __init__(self, channels, gp):
        super().__init__()
        mid_channels = channels
        self.gp = gp
        assert mid_channels % 4 == 0
        self.q_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""
        x: B x C x N
        """
        bs, ch, nums = x.size()
        x_q = self.q_conv(x)  # B x C x N
        x_q = x_q.reshape(bs, self.gp, ch // self.gp, nums)
        x_q = x_q.permute(0, 1, 3, 2)  # B x gp x num x C'

        x_k = self.k_conv(x)  # B x C x N
        x_k = x_k.reshape(bs, self.gp, ch // self.gp, nums)  # B x gp x C' x nums

        x_v = self.v_conv(x)
        energy = torch.matmul(x_q, x_k)  # B x gp x N x N
        energy = torch.sum(energy, dim=1, keepdim=False)

        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        x_r = torch.matmul(x_v, attn)
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Flow_Fine_Offset(nn.Module):
    def __init__(self, channels):
        super(Flow_Fine_Offset, self).__init__()
        self.query = nn.Conv1d(channels, channels, 1, bias=False)
        self.value = nn.Conv1d(channels, channels, 1, bias=False)
        self.mlp1 = nn.Conv1d(channels, channels, 1)
        self.mlp2 = nn.Conv1d(channels, 64, 1, bias=False)
        self.mlp3 = nn.Conv1d(channels, 3, 1, bias=False)

    def forward(self, x):
        x_q = self.query(x)
        x_k = x_q.permute(0, 2, 1)
        x_v = self.value(x)

        energy = torch.matmul(x_q, x_k)
        attention = F.softmax(energy, dim = -1)
        weights = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        fea_sa = torch.matmul(weights, x_v)
        fea_sa = self.mlp1(fea_sa)
        fea_sa = self.mlp2(fea_sa)
        sa_offset = self.mlp3(fea_sa)

        return sa_offset

class HCRF(nn.Module):
    def __init__(self, iter_nums = 3, pairwise_radius = 0.1, pairwise_nsample = 12):
        super(HCRF, self).__init__()

        self.param1 = torch.nn.Parameter(torch.zeros(1))
        self.param2 = torch.nn.Parameter(torch.zeros(1))
        self.param3 = torch.nn.Parameter(1.7 * torch.ones(1))

        self.iter_nums = iter_nums
        self.pairwise_radius = pairwise_radius
        self.pairwise_nsample = pairwise_nsample

    def square_distance(self, src, dst):
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def compute_neighboring_index(self, pc1):
        """
        Find neighboring points for each point

        Returns:
            pairwise_group_idx: B, N, pairwise_nsample
                                neighboring index for each point
            pairwise_valid_mask: B, N, pairwise_nsample
                                mask to indicate whether the distance between each neighboring point and its center
                                point is less than the maximum searching radius or not
        """

        sqrdists = square_distance(pc1, pc1)
        selected_dists, pairwise_group_idx = torch.topk(sqrdists, self.pairwise_nsample, dim=-1, largest=False,
                                                        sorted=False)
        selected_dists = torch.sqrt(selected_dists)

        pairwise_valid_mask = selected_dists < self.pairwise_radius
        pairwise_valid_mask[:, :, 0] = False
        return pairwise_group_idx.int(), pairwise_valid_mask

    def build_pos_kernel(self, pc1, pairwise_group_idx, scale = 0.2):
        """
        Compute position similarity between each neighboring point and its center point
        """

        neighboring_pc1 = pointnet2_utils.grouping_operation(pc1.contiguous(), pairwise_group_idx)
        difference = torch.sum((neighboring_pc1 - pc1.unsqueeze(-1)) ** 2, 1)
        similarity = torch.exp(- difference / scale)
        return similarity

    def build_feats_kernel(self, pc1, norm1, pairwise_group_idx, scale_pos = 0.2, scale_norm = 1.0):
        """
        Compute position and surface normal similarity between each neighboring point and its center point
        """
        neighboring_pc1 = pointnet2_utils.grouping_operation(pc1.contiguous(), pairwise_group_idx)
        difference_pos = torch.sum((neighboring_pc1 - pc1.unsqueeze(-1)) ** 2, 1)

        norm1 = torch.abs(norm1).permute(0, 2, 1)
        neighboring_norm = pointnet2_utils.grouping_operation(norm1.contiguous(), pairwise_group_idx)
        difference_norm = torch.sum((neighboring_norm - norm1.unsqueeze(-1)) ** 2, 1)


        similarity = torch.exp(- difference_pos / scale_pos - difference_norm / scale_norm)
        return similarity


    def forward(self, pc1, flow_pred, voxel_label_1, norm1):
        """
        HCRF to refine the coarse scene flow predictions from DNNS

        Args:
            pc1: B, 3, N
                point coordinates for point cloud 1
            flow_pred: B, 3, N
                coarse scene flow predictions for point cloud 1
            voxel_label_1: B, N
                supervoxel segmentation results (the supervoxel indices) for each point in point cloud 1
            norm1: B, N, 3
                surface normals for point cloud 1

        Returns:
            mean: refined scene flow floe predictions
        """
        norm1 = norm1.permute(0, 2, 1)

        # model parameters for the potential terms
        param1 = torch.exp(self.param1)
        param2 = torch.exp(self.param2)
        param3 = torch.exp(self.param3)

        # compute the neighboring index
        pairwise_group_idx, pairwise_valid_mask = self.compute_neighboring_index(pc1.permute(0, 2, 1))

        # compute the pairwise similarity
        sim1 = self.build_pos_kernel(pc1, pairwise_group_idx)
        sim2 = self.build_feats_kernel(pc1, norm1, pairwise_group_idx)

        # compute the pairwise weight
        pairwise_weight = (param1 * sim1 + param2 * sim2) * pairwise_valid_mask

        # compute the normalization parameter
        std = 1 / (1 + (torch.sum(pairwise_weight , -1) / (torch.sum(pairwise_valid_mask, -1) + 0.00001)) + param3)

        # initialize the mean with coarse scene flow predictions
        mean = flow_pred
        unary = flow_pred

        for index_iter in range(self.iter_nums):
            # unary term
            part1 = unary

            # pairwise term: message passing from neighboring points
            neighboring_mean = pointnet2_utils.grouping_operation(mean.contiguous(), pairwise_group_idx)
            part2 = torch.sum(pairwise_weight.unsqueeze(1) * neighboring_mean, -1) / (torch.sum(pairwise_valid_mask.unsqueeze(1), -1) + 0.00001)

            # High-Order term: Message passing from supervoxel
            part3 = param3 * Rigid(pc1, mean, voxel_label_1)

            mean = (part1 + part2 + part3) * std.unsqueeze(1)

        return mean


def Rigid(pc1, flow_pred, voxel_label_1):
    """
    Compute the rigid motion parameters for each supervoxel
    Produce the high-order message for each point according to the rigid motion parameters of its supervoxel

    Args:
        pc1: B, 3, N
            point coordinates for point cloud 1
        flow_pred: B, 3, N
            intermediate scene flow prediction for point cloud 1
        voxel_label_1:
            supervoxel segmentation results (the supervoxel indexes) for each point in point cloud 1

    Returns:
        message: B, 3, N
            the high-order message for each point computed by the rigid motion assumption
    """
    batch_size = pc1.size(0)
    message = torch.zeros_like(flow_pred)
    for index in range(batch_size):
        message[index, :, :] = rigid_motion_one_sample(pc1[index, :, :], flow_pred[index, :, :], voxel_label_1[index, :])
    return message

def rigid_motion_one_sample(pc1, flow_pred, voxel_label):
    """
    Compute the high-order message in one point cloud
    """

    num_voxel = torch.max(voxel_label) + 1
    message = torch.zeros_like(flow_pred)

    for index_voxel in range(num_voxel):

        target = index_voxel
        mask_one_channel = torch.where(voxel_label == target)
        mask = mask_one_channel[0]

        # Only apply the high-order term to supervoxel whose point number is larger than 20
        if len(mask) > 20:

            voxel_pc1 = pc1[:, mask]
            voxel_flow_pred = flow_pred[:, mask]
            voxel_warped_pc1 = voxel_pc1 + voxel_flow_pred

            [R, t] = compute_rigid_motion_parameter(voxel_pc1, voxel_warped_pc1)
            voxel_message = torch.matmul(R, voxel_pc1) + t - voxel_pc1

            message[:, mask] = voxel_message
        else:
            message[:, mask] = flow_pred[:, mask]

        return message

def compute_rigid_motion_parameter(pc1, pc2):
    """
    Compute the rigid motion parameter for each supervoxel
    """

    pc1_mean = torch.mean(pc1, dim=1, keepdim=True)
    pc2_mean = torch.mean(pc2, dim=1, keepdim=True)

    pc1_moved = pc1 - pc1_mean
    pc2_moved = pc2 - pc2_mean

    X = pc1_moved
    Y = pc2_moved
    inner = torch.matmul(X, Y.transpose(1, 0))

    [U, S, V] = torch.svd(inner)
    R = torch.matmul(V, U.transpose(1, 0))
    t = pc2_mean - torch.matmul(R, pc1_mean)
    return R, t
