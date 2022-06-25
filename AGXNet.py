import timm
import torch
from torch import nn

class AGXNet(nn.Module):
    """AGXNet: a cascade of two networks (anatomy and observation)."""
    def __init__(self, args):
        super(AGXNet, self).__init__()
        self.args = args

        # define feature extractors
        self.net1 = timm.create_model(args.arch, pretrained=self.args.pretrained, features_only=True)
        self.net2 = timm.create_model(args.arch, pretrained=self.args.pretrained, features_only=True)

        # define pooling layers
        if self.args.pool1 == 'average':
            self.pool1 = nn.AdaptiveAvgPool2d(1)
        elif self.args.pool1 == 'max':
            self.pool1 = nn.AdaptiveMaxPool2d(1)
        elif self.args.pool1 == 'log-sum-exp':
            self.pool1 = self.logsumexp_pooling
        else:
            raise Exception('Invalid pooling layer type.')

        if self.args.pool2 == 'average':
            self.pool2 = nn.AdaptiveAvgPool2d(1)
        elif self.args.pool2 == 'max':
            self.pool2 = nn.AdaptiveMaxPool2d(1)
        elif self.args.pool2 == 'log-sum-exp':
            self.pool2 = self.logsumexp_pooling
        else:
            raise Exception('Invalid pooling layer type.')

        self.fc1 = nn.Linear(1024, args.N_landmarks_spec)
        self.fc2 = nn.Linear(1024, args.N_selected_obs)

    def forward(self, x, adj_mtx, annealing, mode):

        f1 = self.net1(x)[-1] # b * 1024 * h * w
        if mode == 'train':
            # dropout some channels to prevent overfitting
            mask = self.dropout_mask(f1)
            f1_drop = torch.einsum('bchw,c->bchw', f1, mask) # b * c * h * w
            f1_p = self.pool1(f1_drop)
            logit1 = self.fc1(f1_p.squeeze())  # b * a
            cam1 = torch.einsum('bchw, ac -> bahw', f1_drop, self.fc1.weight)  # b * a * h * w
            cam1_norm = self.normalize_cam1(cam1)
        else:
            f1_p = self.pool1(f1)
            logit1 = self.fc1(f1_p.squeeze())  # b * a
            cam1 = torch.einsum('bchw, ac -> bahw', f1, self.fc1.weight)  # b * a * h * w
            cam1_norm = self.normalize_cam1(cam1)

        f2 = self.net2(x)[-1] # b * 1024 * h * w
        if self.args.nets_dep == 'dep':
            # net2 needs to aggregate the CAMs from net1
            cam1_agg = torch.einsum('bahw, bao -> bohw', cam1_norm, adj_mtx) # b * o * h * w
            cam1_agg_norm = self.normalize_cam1(cam1_agg)
            f2 = torch.einsum('bchw, bohw -> bochw', f2, 1 + annealing * self.args.beta * cam1_agg_norm) # b * o * c * h * w
            #f2 = torch.einsum('bchw, bohw -> bochw', f2, cam1_agg_norm)  # b * o * c * h * w
            cam2 = torch.einsum('bochw, oc -> bohw', f2, self.fc2.weight) # b * o * h * w
            cam2_norm = self.normalize_cam1(cam2) # b * o * h * w
            [b, o, c, h, w] = f2.shape
            f2_p = self.pool2(f2.reshape(-1, c, h, w)) # (b * o) * c * h * w -> (b * o) * c * 1 * 1
            f2_p = f2_p.reshape(b, o, c, 1, 1) # b * o * c * 1 * 1
            f2_p = f2_p.reshape(b, o, c)  # b * o * c
            logit2 = torch.einsum('boc, oc -> bo', f2_p, self.fc2.weight) # b * o

        elif self.args.nets_dep == 'indep':
            f2_p = self.pool2(f2)
            logit2 = self.fc2(f2_p.squeeze())  # b * o
            cam2 = torch.einsum('bchw, oc -> bohw', f2, self.fc2.weight)  # b * o * h * w
            cam1_agg_norm = torch.zeros_like(cam2) # b * o * h * w
            cam2_norm = self.normalize_cam1(cam2) # b * o * h * w

        else:
            raise Exception('Invalid net dep.')

        return logit1, logit2, cam1_norm, cam1_agg_norm, cam2_norm


    def logsumexp_pooling(self, x):
        [n, c, _, _] = x.shape
        x = x.reshape(n, c, -1)
        x_max = torch.abs(x.max(2, keepdim=True)[0])
        x_p = x_max + (1 / self.args.gamma) * torch.log(torch.mean(torch.exp(self.args.gamma * (x-x_max)), 2)).unsqueeze(-1)
        x_p = x_p.reshape(n, c, 1, 1)
        return x_p

    def normalize_cam1(self, cam1):
        if self.args.cam1_norm == 'norm':
            [b, a, h, w] = cam1.shape
            cam1_norm = cam1.view(b, a, -1)
            cam1_norm -= cam1_norm.min(2, keepdim=True)[0]
            cam1_norm /= (cam1_norm.max(2, keepdim=True)[0] + 1e-12) # pervent from dividing 0
            cam1_norm = cam1_norm.view(b, a, h, w)
        elif self.args.cam1_norm == 'sigmoid':
            cam1_norm = torch.sigmoid(cam1)
        else:
            raise Exception('Invalid CAM 1 normalization method.')
        return cam1_norm

    def dropout_mask(self, f1):
        # method 1: random drop
        if self.args.dropout_method == 'random':
            mask = torch.cuda.FloatTensor(f1.shape[1]).uniform_() > self.args.dropout_rate
        elif self.args.dropout_method == 'proportional':
            # f1 : b * c * h * w
            f1_gap = f1.mean(axis=(0, 2, 3)) # c
            c_prop = f1_gap/f1_gap.sum()
            mask = torch.cuda.FloatTensor(f1.shape[1]).uniform_() > c_prop.data
        return mask