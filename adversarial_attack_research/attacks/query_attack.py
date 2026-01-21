"""
基于查询的黑盒对抗攻击
Query-based Black-box Adversarial Attack

适用场景: 可以查询目标系统获取排名分数或排序结果，但无法获取梯度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple, List
from abc import ABC, abstractmethod


class QueryAttackBase(ABC):
    """查询攻击基类"""

    def __init__(
        self,
        epsilon: float = 8/255,
        max_queries: int = 10000,
        device: str = 'cuda'
    ):
        self.epsilon = epsilon
        self.max_queries = max_queries
        self.device = device
        self.query_count = 0

    @abstractmethod
    def attack(
        self,
        original_image: torch.Tensor,
        query_fn: Callable,
        target_rank: int = 1
    ) -> torch.Tensor:
        """执行攻击"""
        pass

    def reset_query_count(self):
        self.query_count = 0


class NESAttack(QueryAttackBase):
    """
    Natural Evolution Strategies (NES) 攻击

    核心思想: 使用进化策略估计梯度
    - 在当前点周围采样
    - 根据查询结果加权平均得到梯度估计

    优点:
    - 可并行化
    - 对噪声鲁棒

    适用场景: 可获取连续的排名分数
    """

    def __init__(
        self,
        sigma: float = 0.001,
        learning_rate: float = 0.01,
        num_samples: int = 50,
        antithetic: bool = True,  # 使用对称采样
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.antithetic = antithetic

    def estimate_gradient(
        self,
        image: torch.Tensor,
        query_fn: Callable
    ) -> torch.Tensor:
        """
        使用NES估计梯度

        梯度估计公式:
        ∇f(x) ≈ (1/nσ) Σ[f(x + σu_i) - f(x - σu_i)] * u_i

        其中 u_i 是随机采样的方向向量
        """
        grad_estimate = torch.zeros_like(image)
        num_directions = self.num_samples // 2 if self.antithetic else self.num_samples

        for _ in range(num_directions):
            # 随机方向
            noise = torch.randn_like(image)

            if self.antithetic:
                # 对称采样 (antithetic sampling) - 减少方差
                pos_image = torch.clamp(image + self.sigma * noise, 0, 1)
                neg_image = torch.clamp(image - self.sigma * noise, 0, 1)

                score_pos = query_fn(pos_image)
                self.query_count += 1

                score_neg = query_fn(neg_image)
                self.query_count += 1

                # 梯度估计
                grad_estimate += (score_pos - score_neg) * noise
            else:
                # 单向采样
                perturbed_image = torch.clamp(image + self.sigma * noise, 0, 1)
                base_score = query_fn(image)
                perturbed_score = query_fn(perturbed_image)
                self.query_count += 2

                grad_estimate += (perturbed_score - base_score) * noise / self.sigma

        # 平均
        if self.antithetic:
            grad_estimate /= (2 * self.sigma * num_directions)
        else:
            grad_estimate /= num_directions

        return grad_estimate

    def attack(
        self,
        original_image: torch.Tensor,
        query_fn: Callable,
        target_rank: int = 1,
        maximize_score: bool = True
    ) -> torch.Tensor:
        """
        执行NES攻击

        Args:
            original_image: 原始图像
            query_fn: 查询函数，返回排名分数 (分数越高排名越靠前)
            target_rank: 目标排名
            maximize_score: 是否最大化分数

        Returns:
            对抗图像
        """
        self.reset_query_count()

        adv_image = original_image.clone().to(self.device)
        best_image = adv_image.clone()
        best_score = query_fn(adv_image)
        self.query_count += 1

        print(f"Initial score: {best_score:.4f}")

        while self.query_count < self.max_queries:
            # 估计梯度
            grad = self.estimate_gradient(adv_image, query_fn)

            # 更新方向
            if maximize_score:
                update = self.learning_rate * grad.sign()
            else:
                update = -self.learning_rate * grad.sign()

            # 更新图像
            adv_image = adv_image + update

            # 投影到epsilon球
            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)

            # 评估
            current_score = query_fn(adv_image)
            self.query_count += 1

            if (maximize_score and current_score > best_score) or \
               (not maximize_score and current_score < best_score):
                best_score = current_score
                best_image = adv_image.clone()

            if self.query_count % 500 == 0:
                print(f"Queries: {self.query_count}, Best score: {best_score:.4f}")

        print(f"Final score: {best_score:.4f}, Total queries: {self.query_count}")
        return best_image


class ZOSignSGDAttack(QueryAttackBase):
    """
    Zeroth-Order Sign SGD 攻击

    核心思想: 使用随机坐标梯度估计
    - 每次只扰动一个坐标
    - 使用符号函数更新

    优点:
    - 查询效率高
    - 适合高维空间

    参考: "ZO-signSGD: A Zeroth-Order Framework" (Cheng et al., 2019)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        delta: float = 0.01,
        batch_size: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.delta = delta
        self.batch_size = batch_size

    def estimate_gradient_sign(
        self,
        image: torch.Tensor,
        query_fn: Callable
    ) -> torch.Tensor:
        """估计梯度符号"""
        grad_sign = torch.zeros_like(image)

        # 随机选择坐标
        total_coords = image.numel()
        coords = np.random.choice(total_coords, self.batch_size, replace=False)

        flat_image = image.flatten()

        for coord in coords:
            # 正向扰动
            pos_image = flat_image.clone()
            pos_image[coord] += self.delta
            pos_image = torch.clamp(pos_image.view_as(image), 0, 1)

            # 负向扰动
            neg_image = flat_image.clone()
            neg_image[coord] -= self.delta
            neg_image = torch.clamp(neg_image.view_as(image), 0, 1)

            # 查询
            score_pos = query_fn(pos_image)
            score_neg = query_fn(neg_image)
            self.query_count += 2

            # 梯度符号
            if score_pos > score_neg:
                grad_sign.flatten()[coord] = 1
            elif score_pos < score_neg:
                grad_sign.flatten()[coord] = -1

        return grad_sign

    def attack(
        self,
        original_image: torch.Tensor,
        query_fn: Callable,
        target_rank: int = 1,
        maximize_score: bool = True
    ) -> torch.Tensor:
        """执行ZO-SignSGD攻击"""
        self.reset_query_count()

        adv_image = original_image.clone().to(self.device)
        best_image = adv_image.clone()
        best_score = query_fn(adv_image)
        self.query_count += 1

        while self.query_count < self.max_queries:
            # 估计梯度符号
            grad_sign = self.estimate_gradient_sign(adv_image, query_fn)

            # 更新
            if maximize_score:
                adv_image = adv_image + self.learning_rate * grad_sign
            else:
                adv_image = adv_image - self.learning_rate * grad_sign

            # 投影
            perturbation = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + perturbation, 0, 1)

            # 评估
            current_score = query_fn(adv_image)
            self.query_count += 1

            if (maximize_score and current_score > best_score) or \
               (not maximize_score and current_score < best_score):
                best_score = current_score
                best_image = adv_image.clone()

        return best_image


class BoundaryAttack(QueryAttackBase):
    """
    边界攻击 (Decision-based Attack)

    核心思想: 只需要决策边界信息（硬标签）
    - 从对抗样本出发
    - 沿边界游走，逐渐靠近原始图像

    适用场景: 只能获取排名顺序，无法获取分数

    参考: "Decision-Based Adversarial Attacks" (Brendel et al., 2018)
    """

    def __init__(
        self,
        initial_delta: float = 0.1,
        delta_decay: float = 0.9,
        step_adapt: float = 1.5,
        orthogonal_step: float = 0.01,
        source_step: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.initial_delta = initial_delta
        self.delta_decay = delta_decay
        self.step_adapt = step_adapt
        self.orthogonal_step = orthogonal_step
        self.source_step = source_step

    def is_adversarial(
        self,
        image: torch.Tensor,
        query_fn: Callable,
        target_rank: int
    ) -> bool:
        """检查是否满足对抗条件（排名达到目标）"""
        rank = query_fn(image)
        self.query_count += 1
        return rank <= target_rank

    def find_initial_adversarial(
        self,
        original_image: torch.Tensor,
        query_fn: Callable,
        target_rank: int
    ) -> Optional[torch.Tensor]:
        """
        找到初始对抗样本

        策略: 从随机噪声开始，逐渐混合原图直到找到边界点
        """
        # 尝试随机噪声
        for _ in range(100):
            random_image = torch.rand_like(original_image)
            if self.is_adversarial(random_image, query_fn, target_rank):
                print("Found initial adversarial with random noise")
                return random_image

        # 二分搜索
        low, high = 0.0, 1.0
        for _ in range(20):
            mid = (low + high) / 2
            mixed = mid * original_image + (1 - mid) * torch.rand_like(original_image)

            if self.is_adversarial(mixed, query_fn, target_rank):
                low = mid
            else:
                high = mid

        return mid * original_image + (1 - mid) * torch.rand_like(original_image)

    def orthogonal_perturbation(
        self,
        delta: float,
        current_sample: torch.Tensor,
        original_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        生成正交扰动

        正交于从对抗样本到原始样本的方向
        """
        # 从当前样本到原始样本的方向
        diff = original_sample - current_sample
        diff_norm = torch.norm(diff)

        # 随机扰动
        perturbation = torch.randn_like(current_sample)

        # 正交化: 去除沿diff方向的分量
        perturbation = perturbation - (torch.sum(perturbation * diff) / (diff_norm ** 2 + 1e-8)) * diff

        # 归一化
        perturbation = perturbation / (torch.norm(perturbation) + 1e-8)

        # 缩放
        perturbation = delta * perturbation * diff_norm

        return perturbation

    def step_toward_source(
        self,
        current_sample: torch.Tensor,
        original_sample: torch.Tensor,
        step_size: float
    ) -> torch.Tensor:
        """向原始样本方向移动"""
        diff = original_sample - current_sample
        diff_norm = torch.norm(diff)

        step = step_size * diff / (diff_norm + 1e-8)

        return current_sample + step

    def attack(
        self,
        original_image: torch.Tensor,
        query_fn: Callable,
        target_rank: int = 1
    ) -> torch.Tensor:
        """
        执行边界攻击

        Args:
            original_image: 原始图像
            query_fn: 查询函数，返回当前排名
            target_rank: 目标排名

        Returns:
            对抗图像
        """
        self.reset_query_count()

        # 找到初始对抗样本
        adv_image = self.find_initial_adversarial(original_image, query_fn, target_rank)

        if adv_image is None:
            print("Failed to find initial adversarial sample")
            return original_image

        best_image = adv_image.clone()
        best_distance = torch.norm(adv_image - original_image).item()

        delta = self.initial_delta
        step_success = 0

        while self.query_count < self.max_queries:
            # 正交步
            perturbation = self.orthogonal_perturbation(
                delta * self.orthogonal_step,
                adv_image,
                original_image
            )
            candidate = adv_image + perturbation
            candidate = torch.clamp(candidate, 0, 1)

            # 检查是否仍然对抗
            if self.is_adversarial(candidate, query_fn, target_rank):
                # 向原图方向移动
                candidate = self.step_toward_source(
                    candidate,
                    original_image,
                    delta * self.source_step
                )
                candidate = torch.clamp(candidate, 0, 1)

                if self.is_adversarial(candidate, query_fn, target_rank):
                    adv_image = candidate
                    step_success += 1

                    # 更新最佳结果
                    current_distance = torch.norm(adv_image - original_image).item()
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_image = adv_image.clone()

            # 自适应步长
            if step_success > 10:
                delta *= self.step_adapt
                step_success = 0
            elif self.query_count % 100 == 0:
                delta *= self.delta_decay

            if self.query_count % 500 == 0:
                print(f"Queries: {self.query_count}, Best L2: {best_distance:.4f}")

        # 检查是否在epsilon范围内
        final_perturbation = best_image - original_image
        if torch.norm(final_perturbation, p=float('inf')) > self.epsilon:
            # 投影到L∞球
            final_perturbation = torch.clamp(final_perturbation, -self.epsilon, self.epsilon)
            best_image = original_image + final_perturbation

        print(f"Final L2 distance: {best_distance:.4f}, Total queries: {self.query_count}")
        return best_image


class SimBA(QueryAttackBase):
    """
    Simple Black-box Attack (SimBA)

    核心思想: 沿随机方向贪婪搜索
    - 每次选择一个随机方向
    - 如果该方向能改善目标，则接受

    优点:
    - 实现简单
    - 查询效率较高

    参考: "Simple Black-box Adversarial Attacks" (Guo et al., 2019)
    """

    def __init__(
        self,
        step_size: float = 0.02,
        freq_dims: int = 28,  # DCT频率维度
        use_dct: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.step_size = step_size
        self.freq_dims = freq_dims
        self.use_dct = use_dct

    def dct_basis(self, size: int) -> torch.Tensor:
        """生成DCT基"""
        n = size
        basis = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    basis[k, i] = 1 / np.sqrt(n)
                else:
                    basis[k, i] = np.sqrt(2/n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
        return basis

    def sample_random_direction(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """采样随机方向"""
        if self.use_dct:
            # 在DCT域采样低频方向
            _, c, h, w = shape
            direction = torch.zeros(shape)

            for ch in range(c):
                # 随机选择频率分量
                freq_h = torch.randint(0, self.freq_dims, (1,)).item()
                freq_w = torch.randint(0, self.freq_dims, (1,)).item()

                # 创建DCT基向量
                dct_h = self.dct_basis(h)
                dct_w = self.dct_basis(w)

                direction[0, ch] = torch.outer(dct_h[freq_h], dct_w[freq_w])

            # 归一化
            direction = direction / (torch.norm(direction) + 1e-8)
        else:
            # 像素空间随机方向
            direction = torch.randn(shape)
            direction = direction / (torch.norm(direction) + 1e-8)

        return direction

    def attack(
        self,
        original_image: torch.Tensor,
        query_fn: Callable,
        target_rank: int = 1,
        maximize_score: bool = True
    ) -> torch.Tensor:
        """执行SimBA攻击"""
        self.reset_query_count()

        adv_image = original_image.clone().to(self.device)
        best_score = query_fn(adv_image)
        self.query_count += 1

        print(f"Initial score: {best_score:.4f}")

        num_improvements = 0

        while self.query_count < self.max_queries:
            # 采样随机方向
            direction = self.sample_random_direction(adv_image.shape).to(self.device)

            # 尝试正方向
            pos_image = torch.clamp(adv_image + self.step_size * direction, 0, 1)
            # 投影到epsilon球
            perturbation = torch.clamp(pos_image - original_image, -self.epsilon, self.epsilon)
            pos_image = original_image + perturbation

            pos_score = query_fn(pos_image)
            self.query_count += 1

            improved = False

            if (maximize_score and pos_score > best_score) or \
               (not maximize_score and pos_score < best_score):
                best_score = pos_score
                adv_image = pos_image
                improved = True
                num_improvements += 1
            else:
                # 尝试负方向
                neg_image = torch.clamp(adv_image - self.step_size * direction, 0, 1)
                perturbation = torch.clamp(neg_image - original_image, -self.epsilon, self.epsilon)
                neg_image = original_image + perturbation

                neg_score = query_fn(neg_image)
                self.query_count += 1

                if (maximize_score and neg_score > best_score) or \
                   (not maximize_score and neg_score < best_score):
                    best_score = neg_score
                    adv_image = neg_image
                    improved = True
                    num_improvements += 1

            if self.query_count % 500 == 0:
                print(f"Queries: {self.query_count}, Score: {best_score:.4f}, "
                      f"Improvements: {num_improvements}")

        print(f"Final score: {best_score:.4f}, Total improvements: {num_improvements}")
        return adv_image


class SquareAttack(QueryAttackBase):
    """
    Square Attack

    核心思想: 使用随机正方形扰动
    - 每次在图像上添加随机位置的正方形扰动
    - 渐进式减小正方形大小

    优点:
    - 不需要梯度估计
    - 对防御更鲁棒

    参考: "Square Attack: a query-efficient black-box attack" (Andriushchenko et al., 2020)
    """

    def __init__(
        self,
        p_init: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.p_init = p_init

    def get_square_size(self, iteration: int, total_iterations: int) -> float:
        """根据迭代进度计算正方形大小"""
        # 渐进式减小
        p = self.p_init * (1 - iteration / total_iterations) ** 0.5
        return max(p, 0.01)

    def random_square_perturbation(
        self,
        image: torch.Tensor,
        p: float
    ) -> torch.Tensor:
        """生成随机正方形扰动"""
        _, c, h, w = image.shape

        # 正方形大小
        s = int(p * min(h, w))
        s = max(s, 1)

        # 随机位置
        top = torch.randint(0, h - s + 1, (1,)).item()
        left = torch.randint(0, w - s + 1, (1,)).item()

        # 随机扰动值
        perturbation = torch.zeros_like(image)
        noise = torch.sign(torch.randn(1, c, s, s)) * self.epsilon
        perturbation[:, :, top:top+s, left:left+s] = noise.to(image.device)

        return perturbation

    def attack(
        self,
        original_image: torch.Tensor,
        query_fn: Callable,
        target_rank: int = 1,
        maximize_score: bool = True
    ) -> torch.Tensor:
        """执行Square Attack"""
        self.reset_query_count()

        # 初始化: 随机角点扰动
        adv_image = original_image.clone().to(self.device)
        adv_image = torch.clamp(
            adv_image + self.epsilon * torch.sign(torch.randn_like(adv_image)),
            0, 1
        )
        adv_image = torch.clamp(adv_image, original_image - self.epsilon, original_image + self.epsilon)

        best_score = query_fn(adv_image)
        self.query_count += 1

        total_iterations = self.max_queries

        while self.query_count < self.max_queries:
            # 计算当前正方形大小
            p = self.get_square_size(self.query_count, total_iterations)

            # 生成扰动
            delta = self.random_square_perturbation(adv_image, p)

            # 候选图像
            candidate = adv_image + delta
            candidate = torch.clamp(candidate, 0, 1)
            candidate = torch.clamp(candidate, original_image - self.epsilon, original_image + self.epsilon)

            # 评估
            score = query_fn(candidate)
            self.query_count += 1

            if (maximize_score and score > best_score) or \
               (not maximize_score and score < best_score):
                best_score = score
                adv_image = candidate

            if self.query_count % 500 == 0:
                print(f"Queries: {self.query_count}, Score: {best_score:.4f}, p: {p:.4f}")

        return adv_image


def demo():
    """演示各种查询攻击"""
    # 模拟查询函数 (实际使用时替换为真实系统查询)
    def mock_query_fn(image: torch.Tensor) -> float:
        """模拟排名分数查询"""
        # 简单模拟: 基于图像均值
        return image.mean().item() + np.random.normal(0, 0.01)

    original_image = torch.rand(1, 3, 224, 224)

    print("=" * 50)
    print("Testing NES Attack")
    print("=" * 50)
    nes_attacker = NESAttack(
        epsilon=8/255,
        max_queries=1000,
        sigma=0.001,
        num_samples=50,
        device='cpu'
    )
    adv_nes = nes_attacker.attack(original_image, mock_query_fn, maximize_score=True)
    print(f"NES perturbation L∞: {(adv_nes - original_image).abs().max().item():.6f}")

    print("\n" + "=" * 50)
    print("Testing SimBA Attack")
    print("=" * 50)
    simba_attacker = SimBA(
        epsilon=8/255,
        max_queries=1000,
        step_size=0.02,
        device='cpu'
    )
    adv_simba = simba_attacker.attack(original_image, mock_query_fn, maximize_score=True)
    print(f"SimBA perturbation L∞: {(adv_simba - original_image).abs().max().item():.6f}")


if __name__ == '__main__':
    demo()
