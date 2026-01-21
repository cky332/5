"""
语义对抗攻击
Semantic Adversarial Attack for Multimodal Recommendation

核心思想: 利用CLIP的多模态对齐能力，使商品图像与正面文本语义对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import clip
from PIL import Image


class SemanticAttack:
    """
    语义对齐攻击

    攻击策略:
    1. 文本-图像对齐攻击: 使图像特征接近正面评价文本
    2. 类别迁移攻击: 使图像特征接近高销量类别
    3. 属性注入攻击: 使图像隐含特定属性
    """

    def __init__(
        self,
        epsilon: float = 8/255,
        step_size: float = 2/255,
        num_iterations: int = 100,
        device: str = 'cuda'
    ):
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.device = device

        # 加载CLIP模型
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device=device)
        self.clip_model.eval()

        # 预定义的正面文本模板
        self.positive_templates = [
            "a high quality product photo",
            "a best seller product",
            "a highly recommended item",
            "a premium quality product",
            "a popular trending product",
            "a professional product photography",
            "a luxury product image",
            "an excellent product with great reviews",
            "a top rated product",
            "a must have product",
        ]

        # 商品属性文本
        self.attribute_templates = {
            'quality': [
                "high quality", "premium", "luxury", "professional",
                "excellent", "superior", "top-tier", "first-class"
            ],
            'popularity': [
                "best seller", "trending", "popular", "hot item",
                "most wanted", "customer favorite", "top pick"
            ],
            'value': [
                "great value", "worth buying", "good deal",
                "recommended purchase", "smart choice"
            ],
            'aesthetics': [
                "beautiful design", "elegant", "stylish",
                "modern look", "attractive appearance"
            ]
        }

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """编码文本列表"""
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def text_image_alignment_attack(
        self,
        original_image: torch.Tensor,
        positive_texts: Optional[List[str]] = None,
        use_ensemble: bool = True
    ) -> torch.Tensor:
        """
        文本-图像对齐攻击

        使图像特征与正面文本描述对齐

        Args:
            original_image: 原始商品图像 [B, C, H, W]
            positive_texts: 正面文本列表
            use_ensemble: 是否使用多个正面文本的集成

        Returns:
            对抗图像
        """
        if positive_texts is None:
            positive_texts = self.positive_templates

        # 编码正面文本
        text_features = self.encode_texts(positive_texts)

        if use_ensemble:
            # 使用所有正面文本的平均特征
            target_features = text_features.mean(dim=0, keepdim=True)
        else:
            # 随机选择一个
            idx = np.random.randint(len(positive_texts))
            target_features = text_features[idx:idx+1]

        target_features = F.normalize(target_features, dim=-1)

        # 执行攻击
        adv_image = original_image.clone().to(self.device)
        adv_image.requires_grad = True

        momentum = torch.zeros_like(original_image).to(self.device)

        for i in range(self.num_iterations):
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            # 计算图像特征
            image_features = self.clip_model.encode_image(adv_image)
            image_features = F.normalize(image_features, dim=-1)

            # 最大化与正面文本的相似度
            similarity = F.cosine_similarity(image_features, target_features)
            loss = -similarity.mean()  # 最大化相似度 = 最小化负相似度

            loss.backward()

            # 获取梯度
            grad = adv_image.grad.data.clone()
            grad = grad / (torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)

            # 动量更新
            momentum = 0.9 * momentum + grad

            # 更新对抗样本
            adv_image = adv_image.detach() - self.step_size * momentum.sign()

            # 投影
            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image.requires_grad = True

            if (i + 1) % 20 == 0:
                with torch.no_grad():
                    new_features = self.clip_model.encode_image(adv_image)
                    new_features = F.normalize(new_features, dim=-1)
                    new_sim = F.cosine_similarity(new_features, target_features).item()
                print(f"Iteration {i+1}, Similarity: {new_sim:.4f}")

        return adv_image.detach()

    def attribute_injection_attack(
        self,
        original_image: torch.Tensor,
        target_attributes: List[str],
        attribute_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        属性注入攻击

        使图像隐含特定的正面属性

        Args:
            original_image: 原始图像
            target_attributes: 目标属性类别列表 ('quality', 'popularity', 'value', 'aesthetics')
            attribute_weights: 各属性的权重

        Returns:
            对抗图像
        """
        if attribute_weights is None:
            attribute_weights = {attr: 1.0 for attr in target_attributes}

        # 收集所有目标属性的文本
        all_texts = []
        text_weights = []

        for attr in target_attributes:
            if attr in self.attribute_templates:
                for text in self.attribute_templates[attr]:
                    all_texts.append(f"a {text} product")
                    text_weights.append(attribute_weights.get(attr, 1.0))

        text_weights = torch.tensor(text_weights, device=self.device)
        text_weights = text_weights / text_weights.sum()

        # 编码文本
        text_features = self.encode_texts(all_texts)

        # 加权平均
        target_features = (text_features * text_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        target_features = F.normalize(target_features, dim=-1)

        # 执行攻击
        adv_image = original_image.clone().to(self.device)
        adv_image.requires_grad = True

        for i in range(self.num_iterations):
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            image_features = self.clip_model.encode_image(adv_image)
            image_features = F.normalize(image_features, dim=-1)

            loss = -F.cosine_similarity(image_features, target_features).mean()
            loss.backward()

            grad = adv_image.grad.data
            adv_image = adv_image.detach() - self.step_size * grad.sign()

            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image.requires_grad = True

        return adv_image.detach()

    def category_transfer_attack(
        self,
        original_image: torch.Tensor,
        source_category: str,
        target_category: str
    ) -> torch.Tensor:
        """
        类别迁移攻击

        使图像看起来属于更受欢迎的类别

        Args:
            original_image: 原始图像
            source_category: 原始类别
            target_category: 目标类别

        Returns:
            对抗图像
        """
        # 创建类别文本
        source_text = f"a photo of {source_category}"
        target_text = f"a photo of {target_category}"

        source_features = self.encode_texts([source_text])
        target_features = self.encode_texts([target_text])

        adv_image = original_image.clone().to(self.device)
        adv_image.requires_grad = True

        for i in range(self.num_iterations):
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            image_features = self.clip_model.encode_image(adv_image)
            image_features = F.normalize(image_features, dim=-1)

            # 双重目标: 接近目标类别，远离原始类别
            sim_target = F.cosine_similarity(image_features, target_features)
            sim_source = F.cosine_similarity(image_features, source_features)

            loss = -sim_target.mean() + 0.5 * sim_source.mean()
            loss.backward()

            grad = adv_image.grad.data
            adv_image = adv_image.detach() - self.step_size * grad.sign()

            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image.requires_grad = True

        return adv_image.detach()


class MultimodalFusionAttack:
    """
    多模态融合层攻击

    针对VIP5的JointEncoder进行攻击
    目标: 操纵图像特征与文本特征融合后的表示
    """

    def __init__(
        self,
        epsilon: float = 8/255,
        device: str = 'cuda'
    ):
        self.epsilon = epsilon
        self.device = device

        # 加载多个视觉编码器作为替代
        self.encoders = self._load_encoders()

    def _load_encoders(self) -> List[Dict]:
        """加载多个视觉编码器"""
        encoders = []

        # CLIP模型
        for name in ['ViT-B/32', 'ViT-B/16', 'RN50']:
            try:
                model, preprocess = clip.load(name, device=self.device)
                model.eval()
                encoders.append({
                    'name': name,
                    'model': model,
                    'preprocess': preprocess,
                    'type': 'clip'
                })
            except:
                pass

        return encoders

    def attention_manipulation_attack(
        self,
        original_image: torch.Tensor,
        user_context_embedding: torch.Tensor,
        desired_attention_pattern: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        注意力操纵攻击

        目标: 使图像在与用户上下文融合时获得更高的注意力权重

        Args:
            original_image: 原始图像
            user_context_embedding: 用户上下文嵌入
            desired_attention_pattern: 期望的注意力分布

        Returns:
            对抗图像
        """
        adv_image = original_image.clone().to(self.device)
        adv_image.requires_grad = True

        for i in range(100):
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            total_loss = 0

            for encoder in self.encoders:
                if encoder['type'] == 'clip':
                    model = encoder['model']

                    # 提取图像特征
                    image_features = model.encode_image(adv_image)

                    # 计算与用户上下文的注意力
                    attention_scores = torch.matmul(
                        image_features,
                        user_context_embedding.transpose(-1, -2)
                    )

                    # 最大化注意力分数
                    loss = -attention_scores.mean()
                    total_loss += loss

            total_loss.backward()

            grad = adv_image.grad.data
            adv_image = adv_image.detach() - 0.01 * grad.sign()

            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image.requires_grad = True

        return adv_image.detach()

    def cross_modal_alignment_attack(
        self,
        original_image: torch.Tensor,
        target_text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        跨模态对齐攻击

        使图像特征与特定文本嵌入高度对齐

        Args:
            original_image: 原始图像
            target_text_embedding: 目标文本嵌入 (来自T5编码器)

        Returns:
            对抗图像
        """
        adv_image = original_image.clone().to(self.device)
        adv_image.requires_grad = True

        # 将文本嵌入转换为视觉空间的近似
        # 由于无法直接获取T5嵌入，使用CLIP作为桥接
        for encoder in self.encoders:
            if encoder['type'] == 'clip':
                model = encoder['model']
                break

        for i in range(100):
            if adv_image.grad is not None:
                adv_image.grad.zero_()

            image_features = model.encode_image(adv_image)
            image_features = F.normalize(image_features, dim=-1)

            target_norm = F.normalize(target_text_embedding, dim=-1)

            # L2距离损失
            loss = F.mse_loss(image_features, target_norm)
            loss.backward()

            grad = adv_image.grad.data
            adv_image = adv_image.detach() - 0.01 * grad.sign()

            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            adv_image.requires_grad = True

        return adv_image.detach()


class GenerativeSemanticAttack(nn.Module):
    """
    生成式语义攻击

    训练一个生成器网络，自动为任意商品图像生成语义对抗扰动
    """

    def __init__(
        self,
        epsilon: float = 8/255,
        device: str = 'cuda'
    ):
        super().__init__()
        self.epsilon = epsilon
        self.device = device

        # 扰动生成器
        self.generator = self._build_generator()

        # 加载CLIP用于训练
        self.clip_model, _ = clip.load('ViT-B/32', device=device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _build_generator(self) -> nn.Module:
        """构建扰动生成器"""
        generator = nn.Sequential(
            # 编码器
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 残差块
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # 解码器
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        return generator.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成对抗图像"""
        perturbation = self.generator(x) * self.epsilon
        adv_image = torch.clamp(x + perturbation, 0, 1)
        return adv_image

    def train_step(
        self,
        images: torch.Tensor,
        positive_text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        训练一步

        Args:
            images: 商品图像批次
            positive_text_features: 正面文本特征
            negative_text_features: 负面文本特征（可选）

        Returns:
            损失字典
        """
        # 生成对抗图像
        adv_images = self.forward(images)

        # 计算对抗图像特征
        adv_features = self.clip_model.encode_image(adv_images)
        adv_features = F.normalize(adv_features, dim=-1)

        # 正面对齐损失
        positive_sim = F.cosine_similarity(
            adv_features.unsqueeze(1),
            positive_text_features.unsqueeze(0),
            dim=-1
        )
        alignment_loss = -positive_sim.mean()

        # 负面远离损失
        if negative_text_features is not None:
            negative_sim = F.cosine_similarity(
                adv_features.unsqueeze(1),
                negative_text_features.unsqueeze(0),
                dim=-1
            )
            repulsion_loss = negative_sim.mean()
        else:
            repulsion_loss = torch.tensor(0.0).to(self.device)

        # 扰动正则化
        perturbation = adv_images - images
        perturbation_loss = (perturbation.abs().max(dim=1)[0]).mean()

        # 总损失
        total_loss = alignment_loss + 0.5 * repulsion_loss + 0.1 * perturbation_loss

        return {
            'total_loss': total_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'repulsion_loss': repulsion_loss.item(),
            'perturbation_loss': perturbation_loss.item()
        }


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


def demo():
    """演示语义攻击"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化攻击器
    attacker = SemanticAttack(
        epsilon=8/255,
        num_iterations=100,
        device=device
    )

    # 模拟商品图像
    original_image = torch.rand(1, 3, 224, 224).to(device)

    print("=" * 50)
    print("Text-Image Alignment Attack")
    print("=" * 50)
    adv_image_1 = attacker.text_image_alignment_attack(
        original_image,
        positive_texts=["a best seller product", "highly recommended"]
    )
    print(f"Perturbation L∞: {(adv_image_1 - original_image).abs().max().item():.6f}")

    print("\n" + "=" * 50)
    print("Attribute Injection Attack")
    print("=" * 50)
    adv_image_2 = attacker.attribute_injection_attack(
        original_image,
        target_attributes=['quality', 'popularity']
    )
    print(f"Perturbation L∞: {(adv_image_2 - original_image).abs().max().item():.6f}")

    print("\n" + "=" * 50)
    print("Category Transfer Attack")
    print("=" * 50)
    adv_image_3 = attacker.category_transfer_attack(
        original_image,
        source_category="cheap toy",
        target_category="premium electronics"
    )
    print(f"Perturbation L∞: {(adv_image_3 - original_image).abs().max().item():.6f}")


if __name__ == '__main__':
    demo()
