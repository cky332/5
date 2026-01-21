"""
基于迁移的黑盒对抗攻击
Transfer-based Black-box Adversarial Attack for VIP5 Multimodal Recommender System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from typing import List, Dict, Optional, Tuple
import clip
from PIL import Image
from torchvision import transforms


class TransferAttack:
    """
    基于迁移的集成攻击框架

    核心思想: 在多个替代CLIP模型上生成对抗样本，利用对抗样本的迁移性攻击黑盒目标系统

    攻击流程:
    1. 加载多个开源CLIP模型作为替代模型
    2. 在替代模型上计算梯度
    3. 使用集成策略融合多个模型的梯度
    4. 应用各种增强技术提高迁移性
    5. 生成对抗样本
    """

    def __init__(
        self,
        epsilon: float = 8/255,
        step_size: float = 2/255,
        num_iterations: int = 100,
        momentum: float = 1.0,
        diversity_prob: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Args:
            epsilon: 最大扰动幅度 (L∞范数)
            step_size: 每次迭代的步长
            num_iterations: 迭代次数
            momentum: 动量系数
            diversity_prob: 输入多样化概率
            device: 计算设备
        """
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.diversity_prob = diversity_prob
        self.device = device

        # 加载替代模型
        self.surrogate_models = self._load_surrogate_models()

    def _load_surrogate_models(self) -> List[Dict]:
        """
        加载多个CLIP变体作为替代模型

        选择原则:
        - 架构多样性: 包含ViT和ResNet两类
        - 规模多样性: 包含不同大小的模型
        - 与目标系统相似: VIP5使用CLIP特征
        """
        models = []
        model_configs = [
            ('ViT-B/32', 'vit_b32'),    # 与VIP5默认配置相同
            ('ViT-B/16', 'vit_b16'),
            ('ViT-L/14', 'vit_l14'),
            ('RN50', 'rn50'),
            ('RN101', 'rn101'),
        ]

        for model_name, short_name in model_configs:
            try:
                model, preprocess = clip.load(model_name, device=self.device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

                models.append({
                    'model': model,
                    'preprocess': preprocess,
                    'name': short_name,
                    'weight': 1.0  # 可调整各模型权重
                })
                print(f"Loaded surrogate model: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

        return models

    def input_diversity(self, x: torch.Tensor, resize_range: Tuple[int, int] = (224, 256)) -> torch.Tensor:
        """
        输入多样化变换 (DI-FGSM)

        通过随机变换增加对抗样本的多样性，提高迁移性

        变换类型:
        - 随机缩放
        - 随机填充
        - 随机裁剪
        """
        if torch.rand(1).item() >= self.diversity_prob:
            return x

        batch_size, c, h, w = x.shape
        min_size, max_size = resize_range

        # 随机选择新尺寸
        new_size = torch.randint(min_size, max_size, (1,)).item()

        # 缩放
        x_resized = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=False)

        # 随机填充或裁剪回原尺寸
        if new_size > h:
            # 随机裁剪
            start_h = torch.randint(0, new_size - h, (1,)).item()
            start_w = torch.randint(0, new_size - w, (1,)).item()
            x_out = x_resized[:, :, start_h:start_h+h, start_w:start_w+w]
        else:
            # 随机填充
            pad_h = h - new_size
            pad_w = w - new_size
            pad_top = torch.randint(0, pad_h + 1, (1,)).item()
            pad_left = torch.randint(0, pad_w + 1, (1,)).item()
            x_out = F.pad(x_resized, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))

        return x_out

    def translation_invariant(self, grad: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        平移不变性变换 (TI-FGSM)

        通过对梯度进行高斯平滑，提高对抗样本的平移不变性
        """
        # 创建高斯核
        kernel = self._gaussian_kernel(kernel_size, nsig=3)
        kernel = kernel.to(grad.device)

        # 对每个通道应用高斯滤波
        grad_smooth = F.conv2d(
            grad,
            kernel.expand(grad.shape[1], 1, kernel_size, kernel_size),
            padding=kernel_size // 2,
            groups=grad.shape[1]
        )

        return grad_smooth

    def _gaussian_kernel(self, kernel_size: int, nsig: float = 3) -> torch.Tensor:
        """生成高斯核"""
        x = np.arange(kernel_size) - kernel_size // 2
        kern1d = np.exp(-x**2 / (2 * (nsig**2)))
        kern2d = np.outer(kern1d, kern1d)
        kernel = kern2d / kern2d.sum()
        return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def scale_invariant(self, x: torch.Tensor, num_copies: int = 5) -> List[torch.Tensor]:
        """
        尺度不变性变换 (SI-FGSM)

        生成多个不同尺度的输入副本
        """
        copies = [x]
        for i in range(1, num_copies):
            scale = 1.0 / (2 ** i)
            scaled = F.interpolate(x * scale, size=x.shape[-2:], mode='bilinear', align_corners=False)
            copies.append(scaled)
        return copies

    def compute_loss(
        self,
        adv_image: torch.Tensor,
        target_features: torch.Tensor,
        attack_mode: str = 'targeted'
    ) -> torch.Tensor:
        """
        计算集成损失

        Args:
            adv_image: 对抗图像
            target_features: 目标特征 (高排名商品的特征)
            attack_mode: 'targeted' (接近目标) 或 'untargeted' (远离原始)
        """
        total_loss = 0
        total_weight = 0

        for surrogate in self.surrogate_models:
            model = surrogate['model']
            weight = surrogate['weight']

            # 输入多样化
            diverse_input = self.input_diversity(adv_image)

            # 提取特征
            with torch.enable_grad():
                adv_features = model.encode_image(diverse_input)

                # 归一化特征
                adv_features = F.normalize(adv_features, dim=-1)
                target_norm = F.normalize(target_features, dim=-1)

                # 计算相似度损失
                if attack_mode == 'targeted':
                    # 最大化与目标的相似度 (最小化负相似度)
                    loss = -F.cosine_similarity(adv_features, target_norm, dim=-1).mean()
                else:
                    # 最小化与原始的相似度
                    loss = F.cosine_similarity(adv_features, target_norm, dim=-1).mean()

                total_loss += weight * loss
                total_weight += weight

        return total_loss / total_weight

    def attack(
        self,
        original_image: torch.Tensor,
        target_features: torch.Tensor,
        attack_mode: str = 'targeted'
    ) -> torch.Tensor:
        """
        执行迁移攻击

        Args:
            original_image: 原始图像 [B, C, H, W], 范围 [0, 1]
            target_features: 目标特征向量 (想要接近的特征)
            attack_mode: 攻击模式

        Returns:
            对抗图像
        """
        # 初始化对抗样本
        adv_image = original_image.clone().detach().to(self.device)
        adv_image.requires_grad = True

        # 动量缓存
        momentum_grad = torch.zeros_like(original_image).to(self.device)

        # 目标特征
        target_features = target_features.to(self.device)

        for iteration in range(self.num_iterations):
            # 清零梯度
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            # 计算损失
            loss = self.compute_loss(adv_image, target_features, attack_mode)

            # 反向传播
            loss.backward()

            # 获取梯度
            grad = adv_image.grad.data.clone()

            # 平移不变性处理
            grad = self.translation_invariant(grad)

            # 归一化梯度 (L1范数)
            grad = grad / (torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)

            # 动量累积
            momentum_grad = self.momentum * momentum_grad + grad

            # 更新对抗样本
            adv_image = adv_image.detach() - self.step_size * momentum_grad.sign()

            # 投影到epsilon球内
            perturbation = adv_image - original_image
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)

            # 投影到有效图像范围
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image.requires_grad = True

            # 打印进度
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iterations}, Loss: {loss.item():.4f}")

        return adv_image.detach()

    def get_high_rank_features(self, high_rank_images: List[torch.Tensor]) -> torch.Tensor:
        """
        提取高排名商品的平均特征作为攻击目标

        Args:
            high_rank_images: 高排名商品图像列表

        Returns:
            平均特征向量
        """
        all_features = []

        for image in high_rank_images:
            image = image.to(self.device)
            model_features = []

            for surrogate in self.surrogate_models:
                model = surrogate['model']
                with torch.no_grad():
                    features = model.encode_image(image)
                    features = F.normalize(features, dim=-1)
                    model_features.append(features)

            # 取多个模型的平均
            avg_features = torch.stack(model_features).mean(dim=0)
            all_features.append(avg_features)

        # 取所有高排名商品的平均
        target_features = torch.cat(all_features, dim=0).mean(dim=0, keepdim=True)

        return target_features


class EnsembleTransferAttack(TransferAttack):
    """
    增强版集成迁移攻击

    集成多种迁移增强技术:
    - MI-FGSM: 动量迭代
    - DI-FGSM: 输入多样化
    - TI-FGSM: 平移不变性
    - SI-FGSM: 尺度不变性
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_scale_invariant = True
        self.num_scale_copies = 5

    def attack(
        self,
        original_image: torch.Tensor,
        target_features: torch.Tensor,
        attack_mode: str = 'targeted'
    ) -> torch.Tensor:
        """增强版迁移攻击"""
        adv_image = original_image.clone().detach().to(self.device)
        adv_image.requires_grad = True

        momentum_grad = torch.zeros_like(original_image).to(self.device)
        target_features = target_features.to(self.device)

        for iteration in range(self.num_iterations):
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            total_grad = torch.zeros_like(original_image).to(self.device)

            # 尺度不变性: 多尺度计算
            if self.use_scale_invariant:
                scaled_copies = self.scale_invariant(adv_image, self.num_scale_copies)

                for scaled_input in scaled_copies:
                    scaled_input.requires_grad = True

                    # 计算每个尺度的损失
                    loss = self.compute_loss(scaled_input, target_features, attack_mode)
                    loss.backward()

                    # 累积梯度
                    if scaled_input.grad is not None:
                        total_grad += scaled_input.grad.data.clone()

                total_grad /= len(scaled_copies)
            else:
                loss = self.compute_loss(adv_image, target_features, attack_mode)
                loss.backward()
                total_grad = adv_image.grad.data.clone()

            # 平移不变性
            total_grad = self.translation_invariant(total_grad)

            # 归一化
            total_grad = total_grad / (torch.sum(torch.abs(total_grad), dim=(1, 2, 3), keepdim=True) + 1e-8)

            # 动量
            momentum_grad = self.momentum * momentum_grad + total_grad

            # 更新
            adv_image = adv_image.detach() - self.step_size * momentum_grad.sign()

            # 投影
            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image.requires_grad = True

            if (iteration + 1) % 20 == 0:
                print(f"[Ensemble] Iteration {iteration + 1}/{self.num_iterations}")

        return adv_image.detach()


class FeatureCollisionAttack(TransferAttack):
    """
    特征碰撞攻击

    目标: 使攻击图像的特征与指定目标特征尽可能接近
    应用: 使低排名商品的特征接近高排名商品
    """

    def __init__(self, feature_layers: List[str] = None, **kwargs):
        """
        Args:
            feature_layers: 要攻击的特征层 (中间层攻击)
        """
        super().__init__(**kwargs)
        self.feature_layers = feature_layers or ['final']

    def extract_intermediate_features(
        self,
        model: nn.Module,
        image: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """提取中间层特征"""
        features = {}

        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        # 注册钩子
        handles = []
        for name, module in model.visual.named_modules():
            if name == layer_name:
                handles.append(module.register_forward_hook(hook_fn(name)))

        # 前向传播
        with torch.enable_grad():
            _ = model.encode_image(image)

        # 移除钩子
        for handle in handles:
            handle.remove()

        return features.get(layer_name, None)

    def attack_intermediate_layer(
        self,
        original_image: torch.Tensor,
        target_image: torch.Tensor,
        layer_name: str = 'transformer.resblocks.11'
    ) -> torch.Tensor:
        """
        中间层特征攻击

        攻击中间层而非最终特征可以提高迁移性
        """
        adv_image = original_image.clone().detach().to(self.device)
        adv_image.requires_grad = True

        # 提取目标图像的中间层特征
        target_intermediate_features = {}
        for surrogate in self.surrogate_models:
            model = surrogate['model']
            with torch.no_grad():
                feat = self.extract_intermediate_features(model, target_image, layer_name)
                if feat is not None:
                    target_intermediate_features[surrogate['name']] = feat

        momentum_grad = torch.zeros_like(original_image).to(self.device)

        for iteration in range(self.num_iterations):
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            total_loss = 0

            for surrogate in self.surrogate_models:
                model = surrogate['model']
                name = surrogate['name']

                if name not in target_intermediate_features:
                    continue

                # 提取当前中间层特征
                adv_feat = self.extract_intermediate_features(model, adv_image, layer_name)

                if adv_feat is not None:
                    target_feat = target_intermediate_features[name]
                    # L2损失
                    loss = F.mse_loss(adv_feat, target_feat)
                    total_loss += loss

            if total_loss > 0:
                total_loss.backward()

                grad = adv_image.grad.data.clone()
                grad = grad / (torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)

                momentum_grad = self.momentum * momentum_grad + grad
                adv_image = adv_image.detach() - self.step_size * momentum_grad.sign()

                perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
                adv_image = torch.clamp(original_image + perturbation, 0, 1)
                adv_image.requires_grad = True

        return adv_image.detach()


def demo():
    """演示攻击流程"""
    # 初始化攻击器
    attacker = EnsembleTransferAttack(
        epsilon=8/255,
        step_size=2/255,
        num_iterations=100,
        momentum=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 模拟数据
    # 原始商品图像 (要攻击的目标商品)
    original_image = torch.rand(1, 3, 224, 224)

    # 高排名商品图像 (想要模仿的目标)
    high_rank_images = [torch.rand(1, 3, 224, 224) for _ in range(5)]

    # 提取目标特征
    target_features = attacker.get_high_rank_features(high_rank_images)

    # 执行攻击
    adv_image = attacker.attack(original_image, target_features, attack_mode='targeted')

    print(f"Original image shape: {original_image.shape}")
    print(f"Adversarial image shape: {adv_image.shape}")
    print(f"Max perturbation: {(adv_image - original_image).abs().max().item():.6f}")


if __name__ == '__main__':
    demo()
