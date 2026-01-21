"""
对抗攻击实验主程序
Main Experiment Runner for Adversarial Attacks on VIP5

使用方法:
    python run_experiments.py --attack transfer --dataset toys --epsilon 0.03
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

# 添加项目路径
sys.path.append('/home/user/5/src')
sys.path.append('/home/user/5/adversarial_attack_research')

from attacks.transfer_attack import TransferAttack, EnsembleTransferAttack, FeatureCollisionAttack
from attacks.query_attack import NESAttack, ZOSignSGDAttack, BoundaryAttack, SimBA, SquareAttack
from attacks.semantic_attack import SemanticAttack, GenerativeSemanticAttack
from evaluation.evaluator import AdversarialEvaluator, TransferabilityEvaluator, RobustnessEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='VIP5对抗攻击实验')

    # 攻击方法
    parser.add_argument('--attack', type=str, default='transfer',
                        choices=['transfer', 'ensemble_transfer', 'feature_collision',
                                 'nes', 'simba', 'square', 'boundary',
                                 'semantic', 'attribute', 'all'],
                        help='攻击方法')

    # 数据集
    parser.add_argument('--dataset', type=str, default='toys',
                        choices=['toys', 'beauty', 'sports', 'clothing'],
                        help='数据集名称')

    # 攻击参数
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='最大扰动幅度')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='迭代次数')
    parser.add_argument('--max_queries', type=int, default=10000,
                        help='最大查询次数(查询攻击)')

    # 实验参数
    parser.add_argument('--num_samples', type=int, default=100,
                        help='测试样本数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')

    # 输出
    parser.add_argument('--output_dir', type=str,
                        default='/home/user/5/adversarial_attack_research/results',
                        help='输出目录')

    return parser.parse_args()


class VIP5AttackExperiment:
    """
    VIP5攻击实验框架

    整合所有攻击方法和评估流程
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device if torch.cuda.is_available() else 'cpu'

        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self._init_recommender()
        self._init_attacks()
        self._init_evaluator()

    def _init_recommender(self):
        """
        初始化推荐系统接口

        注意: 在实际使用中，需要加载训练好的VIP5模型
        这里提供模拟接口用于演示
        """
        # 模拟推荐系统查询函数
        # 实际使用时替换为真实模型
        def mock_recommender_query(image: torch.Tensor) -> int:
            """
            模拟推荐系统查询

            输入: 商品图像
            输出: 该商品在推荐列表中的排名
            """
            # 这里应该是真实模型的推理
            # 返回随机排名用于演示
            return np.random.randint(1, 100)

        self.recommender_query = mock_recommender_query

        print(f"Recommender system initialized (mock mode)")
        print(f"Note: Replace with actual VIP5 model for real experiments")

    def _init_attacks(self):
        """初始化攻击方法"""
        self.attacks = {}

        # 迁移攻击
        if self.args.attack in ['transfer', 'all']:
            self.attacks['transfer'] = TransferAttack(
                epsilon=self.args.epsilon,
                num_iterations=self.args.num_iterations,
                device=self.device
            )

        if self.args.attack in ['ensemble_transfer', 'all']:
            self.attacks['ensemble_transfer'] = EnsembleTransferAttack(
                epsilon=self.args.epsilon,
                num_iterations=self.args.num_iterations,
                device=self.device
            )

        if self.args.attack in ['feature_collision', 'all']:
            self.attacks['feature_collision'] = FeatureCollisionAttack(
                epsilon=self.args.epsilon,
                num_iterations=self.args.num_iterations,
                device=self.device
            )

        # 查询攻击
        if self.args.attack in ['nes', 'all']:
            self.attacks['nes'] = NESAttack(
                epsilon=self.args.epsilon,
                max_queries=self.args.max_queries,
                device=self.device
            )

        if self.args.attack in ['simba', 'all']:
            self.attacks['simba'] = SimBA(
                epsilon=self.args.epsilon,
                max_queries=self.args.max_queries,
                device=self.device
            )

        if self.args.attack in ['square', 'all']:
            self.attacks['square'] = SquareAttack(
                epsilon=self.args.epsilon,
                max_queries=self.args.max_queries,
                device=self.device
            )

        if self.args.attack in ['boundary', 'all']:
            self.attacks['boundary'] = BoundaryAttack(
                epsilon=self.args.epsilon,
                max_queries=self.args.max_queries,
                device=self.device
            )

        # 语义攻击
        if self.args.attack in ['semantic', 'all']:
            self.attacks['semantic'] = SemanticAttack(
                epsilon=self.args.epsilon,
                num_iterations=self.args.num_iterations,
                device=self.device
            )

        print(f"Initialized attacks: {list(self.attacks.keys())}")

    def _init_evaluator(self):
        """初始化评估器"""
        self.evaluator = AdversarialEvaluator(
            self.recommender_query,
            device=self.device
        )

    def load_data(self):
        """
        加载测试数据

        注意: 实际使用时需要加载真实的商品图像数据
        """
        # 模拟数据
        # 实际使用时从数据集加载
        images = []
        for i in range(self.args.num_samples):
            # 随机生成图像（实际应从数据集加载）
            img = torch.rand(1, 3, 224, 224)
            images.append(img)

        print(f"Loaded {len(images)} test images (mock data)")
        return images

    def get_high_rank_targets(self, num_targets: int = 10):
        """
        获取高排名商品作为攻击目标

        返回高排名商品的特征，用于特征碰撞攻击
        """
        # 模拟高排名商品图像
        # 实际使用时从推荐结果中获取
        high_rank_images = [torch.rand(1, 3, 224, 224) for _ in range(num_targets)]

        print(f"Selected {num_targets} high-rank products as targets")
        return high_rank_images

    def run_transfer_attack(self, images, attack_name='transfer'):
        """运行迁移攻击"""
        attack = self.attacks[attack_name]
        adversarial_images = []

        # 获取目标特征
        high_rank_images = self.get_high_rank_targets()
        target_features = attack.get_high_rank_features(high_rank_images)

        print(f"\nRunning {attack_name} attack on {len(images)} images...")

        for i, img in enumerate(images):
            adv_img = attack.attack(img, target_features, attack_mode='targeted')
            adversarial_images.append(adv_img)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(images)}")

        return adversarial_images

    def run_query_attack(self, images, attack_name):
        """运行查询攻击"""
        attack = self.attacks[attack_name]
        adversarial_images = []
        query_counts = []

        print(f"\nRunning {attack_name} attack on {len(images)} images...")

        for i, img in enumerate(images):
            attack.reset_query_count()

            if attack_name == 'boundary':
                adv_img = attack.attack(img, self.recommender_query, target_rank=10)
            else:
                adv_img = attack.attack(img, self.recommender_query, maximize_score=True)

            adversarial_images.append(adv_img)
            query_counts.append(attack.query_count)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(images)}, Avg queries: {np.mean(query_counts):.0f}")

        return adversarial_images, query_counts

    def run_semantic_attack(self, images):
        """运行语义攻击"""
        attack = self.attacks['semantic']
        adversarial_images = []

        print(f"\nRunning semantic attack on {len(images)} images...")

        for i, img in enumerate(images):
            adv_img = attack.text_image_alignment_attack(img)
            adversarial_images.append(adv_img)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(images)}")

        return adversarial_images

    def run_experiment(self):
        """运行完整实验"""
        print("=" * 60)
        print("VIP5 Adversarial Attack Experiment")
        print("=" * 60)
        print(f"Attack method: {self.args.attack}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Epsilon: {self.args.epsilon}")
        print(f"Num samples: {self.args.num_samples}")
        print("=" * 60)

        # 加载数据
        images = self.load_data()

        # 运行各攻击方法
        all_results = {}

        for attack_name in self.attacks.keys():
            print(f"\n{'='*40}")
            print(f"Running: {attack_name}")
            print('='*40)

            # 执行攻击
            if attack_name in ['transfer', 'ensemble_transfer', 'feature_collision']:
                adversarial_images = self.run_transfer_attack(images, attack_name)
                query_counts = None
            elif attack_name in ['nes', 'simba', 'square', 'boundary']:
                adversarial_images, query_counts = self.run_query_attack(images, attack_name)
            elif attack_name == 'semantic':
                adversarial_images = self.run_semantic_attack(images)
                query_counts = None
            else:
                continue

            # 评估
            metrics, results = self.evaluator.evaluate_batch(
                images,
                adversarial_images,
                query_counts
            )

            # 生成报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"{attack_name}_{self.args.dataset}_{timestamp}.txt"

            report = self.evaluator.generate_report(
                metrics,
                results,
                attack_name=attack_name,
                output_path=str(report_path)
            )
            print(report)

            all_results[attack_name] = {
                'metrics': metrics,
                'report_path': str(report_path)
            }

        # 保存综合结果
        summary = {
            'experiment_config': vars(self.args),
            'timestamp': datetime.now().isoformat(),
            'attacks': {}
        }

        for name, result in all_results.items():
            metrics = result['metrics']
            summary['attacks'][name] = {
                'success_rate': metrics.attack_success_rate,
                'avg_rank_improvement': metrics.avg_rank_improvement,
                'avg_perturbation_linf': metrics.avg_perturbation_linf,
                'avg_ssim': metrics.avg_ssim,
                'top_k_success': metrics.targeted_success_rate
            }

        summary_path = self.output_dir / f"summary_{self.args.dataset}_{timestamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print(f"\nExperiment completed. Results saved to: {self.output_dir}")
        return all_results


def main():
    args = parse_args()

    experiment = VIP5AttackExperiment(args)
    results = experiment.run_experiment()

    # 打印摘要
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for attack_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{attack_name}:")
        print(f"  Success Rate: {metrics.attack_success_rate:.2%}")
        print(f"  Avg Rank Improvement: {metrics.avg_rank_improvement:.2f}")
        print(f"  Top-10 Success: {metrics.targeted_success_rate.get(10, 0):.2%}")


if __name__ == '__main__':
    main()
