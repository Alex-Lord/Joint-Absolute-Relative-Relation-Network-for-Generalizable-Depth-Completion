import torch
from torch import Tensor
from torch.nn import Module, Parameter, L1Loss, MSELoss
from torch.nn.functional import interpolate, conv2d
import sys

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

# 修改后的损失计算模块
class Loss_for_Prob(Module):
    def __init__(self, epsilon=1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, output, target, hole, p):
        # 孔洞区域混合
        sampled_output = hole * output + (1.0 - hole) * target
        
        # 计算加权相对损失
        error = (sampled_output - target) ** 2
        weighted_error = (1 - p) * error / (2 * self.epsilon**2 + 1e-6)
        
        # 仅孔洞区域参与计算
        masked_loss = hole * weighted_error
        num_valid = torch.sum(hole) + 1e-6
        return torch.sum(masked_loss) / num_valid


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
        out = loss / number_valid
        # if torch.isnan(out).any():
        #     if torch.isnan(target).any() or torch.isinf(target).any():
        #         print("Target中存在inf或nan")
        #     if torch.isnan(output).any() or torch.isinf(output).any():
        #         print("output中存在inf或nan")
        #     if torch.isnan(sampled_output).any() or torch.isinf(sampled_output).any():
        #         print("sampled_output中存在inf或nan")
        #     print(f'sampled_output_no_zeros={torch.is_nonzero(sampled_output.all())}, number_valid={number_valid}, loss_fun={loss}')
        #     print(f'type(out)={type(out.item())}')
        return out

class WeightedDataLossL2(Module):
    def __init__(
            self,
    ) -> None:
        super(WeightedDataLossL2, self).__init__()
        self.loss_fun = MSELoss(reduction='sum')
        self.eps = 1e-6

    def forward(self, output: Tensor, target: Tensor, hole: Tensor) -> Tensor:
        assert output.shape == target.shape, "output size != target size"
        sampled_output = hole * output + (1.0 - hole) * target
        number_valid = torch.sum(hole) + self.eps
        loss = self.loss_fun(sampled_output, target)
        out = loss / number_valid
        # if torch.isnan(out).any():
        #     if torch.isnan(target).any() or torch.isinf(target).any():
        #         print("Target中存在inf或nan")
        #     if torch.isnan(output).any() or torch.isinf(output).any():
        #         print("output中存在inf或nan")
        #     if torch.isnan(sampled_output).any() or torch.isinf(sampled_output).any():
        #         print("sampled_output中存在inf或nan")
        #     print(f'sampled_output_no_zeros={torch.is_nonzero(sampled_output.all())}, number_valid={number_valid}, loss_fun={loss}')
        #     print(f'type(out)={type(out.item())}')
        return out


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
            out = loss / number_valid
            # if torch.isnan(out).any():
            #     print(f'k_residual_no_zeros={torch.is_nonzero(k_residual.all())}, number_valid={number_valid}, loss_fun={loss}')
        return out

class MaskedProbExpLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, targets):
        means = out[:, :1, :, :]
        cout = out[:, 1:2, :, :]

        res = torch.exp(cout)  # Residual term
        regl = torch.log(cout+1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss