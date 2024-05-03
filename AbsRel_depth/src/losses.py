import torch
from torch import Tensor
from torch.nn import Module, Parameter, L1Loss
from torch.nn.functional import interpolate, conv2d


class Gradient2D(Module):
    def __init__(self):
        super(Gradient2D, self).__init__()
        kernel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        self.weight_x = Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x: Tensor):
        grad_x = conv2d(x, self.weight_x)
        grad_y = conv2d(x, self.weight_y)
        return grad_x, grad_y


# class HuberLoss(object):
#     def __init__(
#             self,
#             seta: float = 1.0,
#             reduction: str = 'mean',
#     ) -> None:
#         super(HuberLoss, self).__init__()
#         self.seta = seta
#         self.reduction = reduction
#
#     def loss(self, output: Tensor, target: Tensor) -> Tensor:
#         delta = output - target
#         abs_delta = torch.abs(delta)
#         with torch.no_grad():
#             cond = torch.zeros_like(delta)
#             cond[abs_delta <= self.seta] = 1.0
#         loss_metric = cond * (0.5 * delta ** 2) + (1.0 - cond) * (self.seta * (abs_delta - 0.5 * self.seta))
#         if self.reduction == 'sum':
#             loss = torch.sum(loss_metric)
#         else:
#             loss = torch.mean(loss_metric)
#         return loss
#
#
# class RuBerLoss(object):
#     def __init__(
#             self,
#             reduction: str = 'mean',
#             eps=1e-6
#     ) -> None:
#         super(RuBerLoss, self).__init__()
#         self.reduction = reduction
#         self.eps = eps
#
#     def loss(self, output: Tensor, target: Tensor) -> Tensor:
#         abs_delta = torch.abs(output - target)
#         c = 0.3 * torch.max(abs_delta)
#         loss_metric = torch.where(abs_delta <= c, abs_delta,
#                                   torch.sqrt(torch.abs(2 * c * abs_delta - c ** 2) + self.eps))
#         if self.reduction == 'sum':
#             loss = torch.sum(loss_metric)
#         else:
#             loss = torch.mean(loss_metric)
#         return loss
#
#
# class FNBerHuLoss(object):
#     def __init__(
#             self,
#             n: Tensor,
#             reduction: str = 'mean',
#             eps=1e-6
#     ) -> None:
#         super(FNBerHuLoss, self).__init__()
#         self.n = n
#         self.reduction = reduction
#         self.eps = eps
#
#     def loss(self, output: Tensor, target: Tensor) -> Tensor:
#         abs_delta = torch.abs(output - target)
#         c = 0.2 * torch.max(abs_delta)
#         loss_metric = torch.where(abs_delta <= c, abs_delta,
#                                   (abs_delta ** self.n + (self.n - 1) * c ** self.n) /
#                                   (self.n * c ** (self.n - 1) + self.eps))
#         if self.reduction == 'sum':
#             loss = torch.sum(loss_metric)
#         else:
#             loss = torch.mean(loss_metric)
#         return loss
#
#
# class L1Loss(Module):
#     def __init__(self, reduction: str = 'mean'):
#         super(L1Loss, self).__init__()
#         self.reduction = reduction
#
#     def forward(self, output: Tensor, target: Tensor):
#         residual = torch.abs(output - target)
#         if self.reduction == 'sum':
#             loss = torch.sum(residual, dim=(1, 2, 3))
#         else:
#             loss = torch.mean(residual, dim=(1, 2, 3))
#         return loss


class WeightedDataLoss(Module):
    def __init__(
            self,
    ) -> None:
        super(WeightedDataLoss, self).__init__()
        self.loss_fun = L1Loss(reduction='sum')
        self.eps = 1e-6

    def forward(self, output: Tensor, target: Tensor, hole: Tensor) -> Tensor:
        assert output.shape == target.shape, "output size != target size"
        sampled_output = hole * output + (1.0 - hole) * target
        number_valid = torch.sum(hole) + self.eps
        loss = self.loss_fun(sampled_output, target)
        return loss / number_valid


class WeightedMSGradLoss(Module):
    def __init__(
            self,
            k: int = 4,
            sobel: bool = True,
    ):
        super(WeightedMSGradLoss, self).__init__()
        if sobel:
            self.grad_fun = Gradient2D().cuda()
        else:
            self.grad_fun = L1Loss(reduction='sum')
        self.eps = 1e-6
        self.k = k
        self.sobel = sobel

    def __gradient_loss__(self, residual: Tensor) -> Tensor:
        if self.sobel:
            loss_x, loss_y = self.grad_fun(residual)
            loss_x = torch.sum(torch.abs(loss_x))
            loss_y = torch.sum(torch.abs(loss_y))
        else:
            loss_x = self.grad_fun(residual[:, :, 1:, :], residual[:, :, :-1, :])
            loss_y = self.grad_fun(residual[:, :, :, 1:], residual[:, :, :, :-1])

        loss = loss_x + loss_y
        return loss

    def forward(self, output: Tensor, target: Tensor, hole_target: Tensor) -> Tensor:
        assert output.shape == target.shape, "sampled_output size != target size"
        sampled_output = hole_target * output + (1.0 - hole_target) * target
        residual = sampled_output - target
        number_valid = torch.sum(hole_target) + self.eps
        loss = 0.
        for i in range(self.k):
            scale_factor = 1.0 / (2 ** i)
            if i == 0:
                k_residual = residual
            else:
                k_residual = interpolate(residual, scale_factor=scale_factor, recompute_scale_factor=True)
            loss += self.__gradient_loss__(k_residual)
        return loss / number_valid
