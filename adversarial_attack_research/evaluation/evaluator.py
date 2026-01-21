"""
对抗攻击评估框架
Evaluation Framework for Adversarial Attacks on VIP5
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
import json
from pathlib import Path


@dataclass
class AttackResult:
    """单次攻击结果"""
    original_rank: int
    adversarial_rank: int
    rank_change: int
    perturbation_linf: float
    perturbation_l2: float
    ssim: float
    lpips: float
    success: bool  # 排名是否提升
    num_queries: int = 0  # 查询攻击使用


@dataclass
class EvaluationMetrics:
    """综合评估指标"""
    # 攻击成功率
    attack_success_rate: float  # 排名提升的比例
    targeted_success_rate: Dict[int, float]  # Top-K成功率

    # 排名变化
    avg_rank_improvement: float
    median_rank_improvement: float
    max_rank_improvement: int

    # 扰动大小
    avg_perturbation_linf: float
    avg_perturbation_l2: float

    # 图像质量
    avg_ssim: float
    avg_lpips: float

    # 效率
    avg_queries: float  # 查询攻击


class AdversarialEvaluator:
    """
    对抗攻击评估器

    评估维度:
    1. 攻击效果: 排名提升程度
    2. 隐蔽性: 扰动大小和视觉质量
    3. 效率: 查询次数（查询攻击）
    4. 迁移性: 跨模型攻击效果
    """

    def __init__(
        self,
        recommender_query_fn: Callable,
        device: str = 'cuda'
    ):
        """
        Args:
            recommender_query_fn: 推荐系统查询函数
                输入: 图像张量
                输出: 排名 (int) 或 排名分数 (float)
            device: 计算设备
        """
        self.query_fn = recommender_query_fn
        self.device = device

        # LPIPS感知距离计算器
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()

    def evaluate_single(
        self,
        original_image: torch.Tensor,
        adversarial_image: torch.Tensor,
        num_queries: int = 0
    ) -> AttackResult:
        """
        评估单个对抗样本

        Args:
            original_image: 原始图像 [1, C, H, W]
            adversarial_image: 对抗图像 [1, C, H, W]
            num_queries: 查询次数（查询攻击）

        Returns:
            AttackResult
        """
        # 确保在正确设备上
        original_image = original_image.to(self.device)
        adversarial_image = adversarial_image.to(self.device)

        # 获取排名
        original_rank = self.query_fn(original_image)
        adversarial_rank = self.query_fn(adversarial_image)

        # 计算排名变化
        rank_change = original_rank - adversarial_rank  # 正值表示排名提升

        # 计算扰动
        perturbation = adversarial_image - original_image
        perturbation_linf = perturbation.abs().max().item()
        perturbation_l2 = torch.norm(perturbation).item()

        # 计算SSIM
        orig_np = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
        adv_np = adversarial_image.squeeze().permute(1, 2, 0).cpu().numpy()
        ssim_value = ssim(orig_np, adv_np, channel_axis=2, data_range=1.0)

        # 计算LPIPS
        with torch.no_grad():
            # LPIPS期望输入范围[-1, 1]
            orig_lpips = original_image * 2 - 1
            adv_lpips = adversarial_image * 2 - 1
            lpips_value = self.lpips_fn(orig_lpips, adv_lpips).item()

        return AttackResult(
            original_rank=original_rank,
            adversarial_rank=adversarial_rank,
            rank_change=rank_change,
            perturbation_linf=perturbation_linf,
            perturbation_l2=perturbation_l2,
            ssim=ssim_value,
            lpips=lpips_value,
            success=rank_change > 0,
            num_queries=num_queries
        )

    def evaluate_batch(
        self,
        original_images: List[torch.Tensor],
        adversarial_images: List[torch.Tensor],
        query_counts: Optional[List[int]] = None,
        target_ranks: List[int] = [1, 5, 10, 20]
    ) -> Tuple[EvaluationMetrics, List[AttackResult]]:
        """
        批量评估

        Args:
            original_images: 原始图像列表
            adversarial_images: 对抗图像列表
            query_counts: 查询次数列表
            target_ranks: 目标排名列表（用于计算Top-K成功率）

        Returns:
            综合指标和单个结果列表
        """
        if query_counts is None:
            query_counts = [0] * len(original_images)

        results = []
        for orig, adv, queries in zip(original_images, adversarial_images, query_counts):
            result = self.evaluate_single(orig, adv, queries)
            results.append(result)

        # 计算综合指标
        num_samples = len(results)

        # 攻击成功率
        attack_success_rate = sum(r.success for r in results) / num_samples

        # Top-K成功率
        targeted_success_rate = {}
        for k in target_ranks:
            success_count = sum(1 for r in results if r.adversarial_rank <= k)
            targeted_success_rate[k] = success_count / num_samples

        # 排名变化统计
        rank_changes = [r.rank_change for r in results]
        avg_rank_improvement = np.mean(rank_changes)
        median_rank_improvement = np.median(rank_changes)
        max_rank_improvement = max(rank_changes)

        # 扰动统计
        avg_perturbation_linf = np.mean([r.perturbation_linf for r in results])
        avg_perturbation_l2 = np.mean([r.perturbation_l2 for r in results])

        # 图像质量统计
        avg_ssim = np.mean([r.ssim for r in results])
        avg_lpips = np.mean([r.lpips for r in results])

        # 查询统计
        avg_queries = np.mean([r.num_queries for r in results])

        metrics = EvaluationMetrics(
            attack_success_rate=attack_success_rate,
            targeted_success_rate=targeted_success_rate,
            avg_rank_improvement=avg_rank_improvement,
            median_rank_improvement=median_rank_improvement,
            max_rank_improvement=max_rank_improvement,
            avg_perturbation_linf=avg_perturbation_linf,
            avg_perturbation_l2=avg_perturbation_l2,
            avg_ssim=avg_ssim,
            avg_lpips=avg_lpips,
            avg_queries=avg_queries
        )

        return metrics, results

    def generate_report(
        self,
        metrics: EvaluationMetrics,
        results: List[AttackResult],
        attack_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告

        Args:
            metrics: 综合指标
            results: 单个结果列表
            attack_name: 攻击方法名称
            output_path: 输出路径（可选）

        Returns:
            报告字符串
        """
        report = f"""
========================================
对抗攻击评估报告
Attack Method: {attack_name}
========================================

1. 攻击效果
----------------------------------------
攻击成功率 (排名提升): {metrics.attack_success_rate:.2%}

Top-K 成功率:
"""
        for k, rate in sorted(metrics.targeted_success_rate.items()):
            report += f"  Top-{k}: {rate:.2%}\n"

        report += f"""
排名变化:
  平均提升: {metrics.avg_rank_improvement:.2f}
  中位数提升: {metrics.median_rank_improvement:.2f}
  最大提升: {metrics.max_rank_improvement}

2. 隐蔽性
----------------------------------------
扰动大小:
  平均 L∞: {metrics.avg_perturbation_linf:.6f}
  平均 L2: {metrics.avg_perturbation_l2:.6f}

图像质量:
  平均 SSIM: {metrics.avg_ssim:.4f}
  平均 LPIPS: {metrics.avg_lpips:.4f}

3. 效率
----------------------------------------
平均查询次数: {metrics.avg_queries:.0f}

4. 详细结果分布
----------------------------------------
"""
        # 添加分布统计
        rank_changes = [r.rank_change for r in results]
        report += f"排名变化分布:\n"
        report += f"  提升 > 10: {sum(1 for r in rank_changes if r > 10)} ({sum(1 for r in rank_changes if r > 10)/len(rank_changes):.1%})\n"
        report += f"  提升 5-10: {sum(1 for r in rank_changes if 5 <= r <= 10)} ({sum(1 for r in rank_changes if 5 <= r <= 10)/len(rank_changes):.1%})\n"
        report += f"  提升 1-5: {sum(1 for r in rank_changes if 1 <= r < 5)} ({sum(1 for r in rank_changes if 1 <= r < 5)/len(rank_changes):.1%})\n"
        report += f"  无变化: {sum(1 for r in rank_changes if r == 0)} ({sum(1 for r in rank_changes if r == 0)/len(rank_changes):.1%})\n"
        report += f"  下降: {sum(1 for r in rank_changes if r < 0)} ({sum(1 for r in rank_changes if r < 0)/len(rank_changes):.1%})\n"

        report += """
========================================
"""

        if output_path:
            Path(output_path).write_text(report)

            # 同时保存JSON格式
            json_data = {
                'attack_name': attack_name,
                'metrics': {
                    'attack_success_rate': metrics.attack_success_rate,
                    'targeted_success_rate': metrics.targeted_success_rate,
                    'avg_rank_improvement': metrics.avg_rank_improvement,
                    'median_rank_improvement': metrics.median_rank_improvement,
                    'max_rank_improvement': metrics.max_rank_improvement,
                    'avg_perturbation_linf': metrics.avg_perturbation_linf,
                    'avg_perturbation_l2': metrics.avg_perturbation_l2,
                    'avg_ssim': metrics.avg_ssim,
                    'avg_lpips': metrics.avg_lpips,
                    'avg_queries': metrics.avg_queries
                },
                'results': [
                    {
                        'original_rank': r.original_rank,
                        'adversarial_rank': r.adversarial_rank,
                        'rank_change': r.rank_change,
                        'perturbation_linf': r.perturbation_linf,
                        'ssim': r.ssim,
                        'success': r.success
                    } for r in results
                ]
            }
            json_path = output_path.replace('.txt', '.json')
            Path(json_path).write_text(json.dumps(json_data, indent=2))

        return report


class TransferabilityEvaluator:
    """
    迁移性评估器

    评估对抗样本在不同模型间的迁移效果
    """

    def __init__(
        self,
        source_models: List[Dict],
        target_models: List[Dict],
        device: str = 'cuda'
    ):
        """
        Args:
            source_models: 源模型列表 (用于生成对抗样本)
            target_models: 目标模型列表 (用于评估迁移性)
        """
        self.source_models = source_models
        self.target_models = target_models
        self.device = device

    def evaluate_transferability(
        self,
        original_images: List[torch.Tensor],
        adversarial_images: List[torch.Tensor],
        source_model_name: str
    ) -> Dict[str, Dict]:
        """
        评估迁移性

        Returns:
            各目标模型的攻击效果
        """
        results = {}

        for target in self.target_models:
            target_name = target['name']
            model = target['model']

            # 在目标模型上评估
            original_features = []
            adversarial_features = []

            with torch.no_grad():
                for orig, adv in zip(original_images, adversarial_images):
                    orig = orig.to(self.device)
                    adv = adv.to(self.device)

                    orig_feat = model.encode_image(orig)
                    adv_feat = model.encode_image(adv)

                    original_features.append(orig_feat)
                    adversarial_features.append(adv_feat)

            # 计算特征变化
            original_features = torch.cat(original_features, dim=0)
            adversarial_features = torch.cat(adversarial_features, dim=0)

            # 特征相似度变化
            orig_sim = F.cosine_similarity(
                original_features[:-1], original_features[1:], dim=-1
            ).mean()

            adv_sim = F.cosine_similarity(
                adversarial_features[:-1], adversarial_features[1:], dim=-1
            ).mean()

            # 原始-对抗相似度
            orig_adv_sim = F.cosine_similarity(
                original_features, adversarial_features, dim=-1
            ).mean()

            results[target_name] = {
                'original_consistency': orig_sim.item(),
                'adversarial_consistency': adv_sim.item(),
                'original_adversarial_similarity': orig_adv_sim.item(),
                'feature_shift': 1 - orig_adv_sim.item()  # 特征偏移程度
            }

        return {
            'source_model': source_model_name,
            'target_results': results
        }


class RobustnessEvaluator:
    """
    鲁棒性评估器

    评估对抗样本对各种防御/预处理的鲁棒性
    """

    def __init__(
        self,
        recommender_query_fn: Callable,
        device: str = 'cuda'
    ):
        self.query_fn = recommender_query_fn
        self.device = device

    def apply_defense(
        self,
        image: torch.Tensor,
        defense_type: str
    ) -> torch.Tensor:
        """
        应用防御/预处理

        Args:
            image: 输入图像
            defense_type: 防御类型

        Returns:
            处理后的图像
        """
        image = image.to(self.device)

        if defense_type == 'jpeg_compression':
            # JPEG压缩
            quality = 75
            from io import BytesIO
            img_pil = Image.fromarray(
                (image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_compressed = Image.open(buffer)
            img_tensor = torch.from_numpy(
                np.array(img_compressed).astype(np.float32) / 255
            ).permute(2, 0, 1).unsqueeze(0)
            return img_tensor.to(self.device)

        elif defense_type == 'gaussian_blur':
            # 高斯模糊
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=3, sigma=1.0)
            return blur(image)

        elif defense_type == 'median_filter':
            # 中值滤波
            from scipy.ndimage import median_filter
            img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
            filtered = median_filter(img_np, size=3)
            return torch.from_numpy(filtered).permute(2, 0, 1).unsqueeze(0).to(self.device)

        elif defense_type == 'bit_depth_reduction':
            # 位深度缩减
            bits = 4
            image = torch.round(image * (2**bits - 1)) / (2**bits - 1)
            return image

        elif defense_type == 'random_resize':
            # 随机缩放
            scale = np.random.uniform(0.9, 1.1)
            h, w = image.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)
            resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear')
            return F.interpolate(resized, size=(h, w), mode='bilinear')

        return image

    def evaluate_robustness(
        self,
        original_images: List[torch.Tensor],
        adversarial_images: List[torch.Tensor],
        defenses: List[str] = None
    ) -> Dict[str, Dict]:
        """
        评估对抗样本对各种防御的鲁棒性

        Returns:
            各防御下的攻击效果
        """
        if defenses is None:
            defenses = [
                'jpeg_compression',
                'gaussian_blur',
                'median_filter',
                'bit_depth_reduction',
                'random_resize'
            ]

        results = {'no_defense': {'success_rate': 0, 'avg_rank_change': 0}}

        # 无防御基线
        for orig, adv in zip(original_images, adversarial_images):
            orig_rank = self.query_fn(orig)
            adv_rank = self.query_fn(adv)
            if adv_rank < orig_rank:
                results['no_defense']['success_rate'] += 1
            results['no_defense']['avg_rank_change'] += orig_rank - adv_rank

        n = len(original_images)
        results['no_defense']['success_rate'] /= n
        results['no_defense']['avg_rank_change'] /= n

        # 各防御方法
        for defense in defenses:
            results[defense] = {'success_rate': 0, 'avg_rank_change': 0}

            for orig, adv in zip(original_images, adversarial_images):
                # 应用防御
                adv_defended = self.apply_defense(adv, defense)

                orig_rank = self.query_fn(orig)
                adv_rank = self.query_fn(adv_defended)

                if adv_rank < orig_rank:
                    results[defense]['success_rate'] += 1
                results[defense]['avg_rank_change'] += orig_rank - adv_rank

            results[defense]['success_rate'] /= n
            results[defense]['avg_rank_change'] /= n

        return results


def demo():
    """演示评估流程"""
    # 模拟推荐系统查询函数
    def mock_query_fn(image: torch.Tensor) -> int:
        """模拟返回排名"""
        return np.random.randint(1, 100)

    device = 'cpu'

    # 初始化评估器
    evaluator = AdversarialEvaluator(mock_query_fn, device=device)

    # 模拟数据
    original_images = [torch.rand(1, 3, 224, 224) for _ in range(10)]
    adversarial_images = [img + 0.01 * torch.randn_like(img) for img in original_images]
    adversarial_images = [torch.clamp(img, 0, 1) for img in adversarial_images]

    # 批量评估
    metrics, results = evaluator.evaluate_batch(
        original_images,
        adversarial_images,
        target_ranks=[1, 5, 10, 20]
    )

    # 生成报告
    report = evaluator.generate_report(
        metrics,
        results,
        attack_name="Demo Attack",
        output_path="/home/user/5/adversarial_attack_research/evaluation/demo_report.txt"
    )
    print(report)


if __name__ == '__main__':
    demo()
