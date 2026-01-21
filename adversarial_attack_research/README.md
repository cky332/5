# 黑盒对抗攻击研究方案：针对多模态推荐系统的商品排名提升

## 研究背景

本研究针对VIP5多模态推荐系统，探索通过图像扰动提升目标商品排名的黑盒攻击方法。该研究属于AI安全领域的对抗鲁棒性研究，旨在揭示多模态推荐系统的潜在安全漏洞，为后续防御措施的开发提供理论基础。

## 1. 攻击目标分析

### 1.1 系统架构理解

```
用户查询/历史 + 商品图像
        ↓
┌───────────────────────────────────┐
│     CLIP Vision Encoder          │  ← 图像特征提取 (黑盒)
│     (ViT-B/32, ViT-L/14等)       │
└───────────────────────────────────┘
        ↓ 视觉特征 [B, V_W_L, feat_dim]
┌───────────────────────────────────┐
│     VisualEmbedding              │
│     MLP投影到模型空间             │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│     JointEncoder                 │  ← 多模态融合
│     文本嵌入 + 视觉嵌入 + 类别嵌入  │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│     T5 Decoder                   │  ← 生成推荐结果
│     Seq2Seq生成                   │
└───────────────────────────────────┘
        ↓
    推荐商品排名
```

### 1.2 攻击目标

- **主要目标**: 通过修改目标商品的图像，使其在推荐列表中排名上升
- **攻击类型**:
  - 定向攻击: 将目标商品推至Top-K
  - 非定向攻击: 扰乱整体排序，降低推荐质量

### 1.3 约束条件

- **黑盒约束**: 无法获取模型参数和梯度
- **感知约束**: 扰动需要对人眼不可见 (L∞ ≤ ε, 通常 ε = 8/255 或 16/255)
- **查询约束**: 限制查询次数（模拟真实场景）

---

## 2. 黑盒攻击方法设计

### 2.1 方法一：基于迁移的攻击 (Transfer-based Attack)

**核心思想**: 利用替代模型生成对抗样本，迁移到目标黑盒模型

```
┌─────────────────────────────────────────────────────┐
│                 替代模型选择                          │
├─────────────────────────────────────────────────────┤
│ 1. 开源CLIP模型 (与目标系统架构相似)                   │
│ 2. 其他视觉编码器 (ResNet, ViT变体)                   │
│ 3. 集成多个模型提高迁移性                             │
└─────────────────────────────────────────────────────┘
```

**攻击流程**:

```python
# 伪代码
def transfer_attack(target_image, surrogate_models, epsilon=8/255):
    """
    基于迁移的对抗攻击
    """
    adv_image = target_image.clone()

    for model in surrogate_models:
        # 在替代模型上计算梯度
        loss = compute_feature_similarity_loss(model, adv_image, target_features)
        grad = compute_gradient(loss, adv_image)

        # 集成梯度
        ensemble_grad += grad / len(surrogate_models)

    # 更新对抗样本
    adv_image = adv_image + epsilon * sign(ensemble_grad)
    adv_image = clip(adv_image, 0, 1)

    return adv_image
```

**提高迁移性的技术**:
- **输入变换**: 随机缩放、平移、旋转
- **模型集成**: 多个不同架构的替代模型
- **中间层攻击**: 攻击中间层特征而非最终输出
- **动量攻击 (MI-FGSM)**: 添加动量项稳定梯度方向

### 2.2 方法二：基于查询的攻击 (Query-based Attack)

**核心思想**: 通过观察模型输出估计梯度

#### 2.2.1 基于得分的攻击 (Score-based)

如果能获取推荐得分/排名分数：

```python
def score_based_attack(target_image, query_fn, epsilon=8/255, num_queries=10000):
    """
    基于得分的黑盒攻击 - 使用自然进化策略(NES)估计梯度
    """
    adv_image = target_image.clone()
    sigma = 0.001  # 采样标准差

    for _ in range(num_iterations):
        # 估计梯度
        grad_estimate = torch.zeros_like(adv_image)

        for _ in range(num_samples):
            # 随机方向采样
            noise = torch.randn_like(adv_image)

            # 正向和反向查询
            score_pos = query_fn(adv_image + sigma * noise)
            score_neg = query_fn(adv_image - sigma * noise)

            # 累积梯度估计
            grad_estimate += (score_pos - score_neg) * noise / (2 * sigma)

        grad_estimate /= num_samples

        # 更新对抗样本
        adv_image = adv_image + step_size * sign(grad_estimate)
        adv_image = project_to_epsilon_ball(adv_image, target_image, epsilon)

    return adv_image
```

#### 2.2.2 基于决策的攻击 (Decision-based)

只能获取排名顺序（硬标签）：

```python
def decision_based_attack(target_image, query_fn, target_rank, epsilon=8/255):
    """
    基于决策的黑盒攻击 - 边界攻击(Boundary Attack)
    """
    # 初始化：找到一个对抗样本（随机噪声直到排名改变）
    adv_image = find_initial_adversarial(target_image, query_fn, target_rank)

    for step in range(max_steps):
        # 沿边界随机游走
        perturbation = orthogonal_perturbation(adv_image, target_image)
        candidate = adv_image + perturbation

        # 向原图方向移动
        candidate = candidate + step_toward_original(candidate, target_image)

        # 检查是否仍然对抗
        if query_fn(candidate) <= target_rank:
            adv_image = candidate

    return adv_image
```

### 2.3 方法三：基于特征的攻击 (Feature-based Attack)

**核心思想**: 针对CLIP视觉特征空间进行攻击

```
攻击策略:
1. 特征碰撞攻击: 使目标商品图像特征接近高排名商品
2. 特征逃逸攻击: 使目标商品图像特征远离低排名商品
3. 语义对齐攻击: 使图像特征与正面文本描述对齐
```

```python
def feature_collision_attack(target_image, high_rank_image, surrogate_clip):
    """
    特征碰撞攻击：使目标图像特征接近高排名商品
    """
    target_features = surrogate_clip.encode_image(high_rank_image)

    adv_image = target_image.clone()
    adv_image.requires_grad = True

    for _ in range(iterations):
        current_features = surrogate_clip.encode_image(adv_image)

        # 最小化与目标特征的距离
        loss = -cosine_similarity(current_features, target_features)

        grad = torch.autograd.grad(loss, adv_image)[0]
        adv_image = adv_image - step_size * grad.sign()
        adv_image = clip_perturbation(adv_image, target_image, epsilon)

    return adv_image
```

### 2.4 方法四：生成式对抗扰动 (Generative Adversarial Perturbation)

**核心思想**: 训练生成器网络产生通用扰动

```
┌────────────────────────────────────────────┐
│           Generator Network                │
│         (输入: 原始图像)                    │
│         (输出: 对抗扰动)                    │
└────────────────────────────────────────────┘
                    ↓
        adv_image = image + G(image)
                    ↓
┌────────────────────────────────────────────┐
│      Surrogate CLIP Models                 │
│      (计算特征相似度损失)                   │
└────────────────────────────────────────────┘
```

```python
class AdversarialGenerator(nn.Module):
    def __init__(self, epsilon=8/255):
        super().__init__()
        self.epsilon = epsilon
        self.encoder = ResNetEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        features = self.encoder(x)
        perturbation = self.decoder(features)
        # 限制扰动幅度
        perturbation = torch.tanh(perturbation) * self.epsilon
        return x + perturbation

def train_generator(generator, surrogate_models, dataloader):
    """训练对抗扰动生成器"""
    optimizer = Adam(generator.parameters(), lr=1e-4)

    for images, high_rank_features in dataloader:
        adv_images = generator(images)

        # 多模型集成损失
        total_loss = 0
        for model in surrogate_models:
            adv_features = model.encode_image(adv_images)
            # 最大化与高排名特征的相似度
            loss = -cosine_similarity(adv_features, high_rank_features)
            total_loss += loss

        # 添加感知损失保持视觉质量
        perceptual_loss = lpips(images, adv_images)
        total_loss += lambda_perceptual * perceptual_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## 3. 针对VIP5系统的具体攻击策略

### 3.1 攻击入口分析

VIP5系统的图像处理流程：

```python
# 来自 modeling_vip5.py 的 VisualEmbedding
class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        # 图像特征维度映射
        # vitb32: 512 → d_model (768 for T5-base)
        # vitl14: 768 → d_model
        self.visual_fc = nn.Linear(config.feat_dim * config.n_vis_tokens,
                                   config.n_vis_tokens * config.d_model)
```

**攻击关键点**:
1. **CLIP特征层**: 攻击CLIP提取的视觉特征
2. **投影层后**: 攻击MLP投影后的特征
3. **融合层**: 攻击多模态融合后的表示

### 3.2 任务特定攻击

#### Sequential推荐攻击 (A任务)

```
目标: 使目标商品出现在"下一个购买"预测中

攻击策略:
1. 收集用户购买序列中的商品图像特征
2. 使目标商品图像特征与序列末端商品相似
3. 利用时序相关性提升被推荐概率
```

#### Direct推荐攻击 (B任务)

```
目标: 在候选集排序中提升目标商品排名

攻击策略:
1. 分析高排名商品的共同视觉特征
2. 通过特征迁移使目标商品获得类似特征
3. 或者降低竞争商品的视觉质量
```

#### Explanation推荐攻击 (C任务)

```
目标: 使模型生成对目标商品的正面解释

攻击策略:
1. 将图像特征与正面评价文本对齐
2. 利用CLIP的多模态对齐能力
3. 使图像触发正面语义生成
```

### 3.3 实现代码框架

```python
# adversarial_attack.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip

class VIP5BlackBoxAttack:
    """VIP5多模态推荐系统黑盒攻击框架"""

    def __init__(self, epsilon=8/255, attack_type='transfer'):
        self.epsilon = epsilon
        self.attack_type = attack_type

        # 加载替代模型
        self.surrogate_models = self._load_surrogate_models()

    def _load_surrogate_models(self):
        """加载多个CLIP替代模型用于迁移攻击"""
        models = []
        model_names = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101']

        for name in model_names:
            model, preprocess = clip.load(name)
            models.append({
                'model': model.eval(),
                'preprocess': preprocess,
                'name': name
            })

        return models

    def transfer_attack(self, image, target_features, iterations=100):
        """基于迁移的集成攻击"""
        adv_image = image.clone().requires_grad_(True)

        # 动量
        momentum = torch.zeros_like(image)
        mu = 1.0  # 动量系数

        for i in range(iterations):
            total_grad = torch.zeros_like(image)

            for surrogate in self.surrogate_models:
                model = surrogate['model']

                # 输入多样化
                transformed = self._input_diversity(adv_image)

                # 计算特征
                features = model.encode_image(transformed)

                # 特征碰撞损失
                loss = -F.cosine_similarity(
                    features, target_features, dim=-1
                ).mean()

                # 计算梯度
                grad = torch.autograd.grad(loss, adv_image)[0]
                total_grad += grad

            # 平均梯度
            total_grad /= len(self.surrogate_models)

            # 动量更新
            total_grad = total_grad / torch.norm(total_grad, p=1)
            momentum = mu * momentum + total_grad

            # 更新对抗样本
            adv_image = adv_image - self.epsilon / iterations * momentum.sign()

            # 投影到epsilon球内
            perturbation = adv_image - image
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(image + perturbation, 0, 1)
            adv_image = adv_image.detach().requires_grad_(True)

        return adv_image.detach()

    def _input_diversity(self, image, p=0.5):
        """输入多样化增强迁移性"""
        if torch.rand(1) < p:
            # 随机缩放
            scale = torch.randint(224, 256, (1,)).item()
            image = F.interpolate(image, size=(scale, scale), mode='bilinear')
            image = F.interpolate(image, size=(224, 224), mode='bilinear')
        return image

    def query_attack(self, image, query_fn, num_queries=10000):
        """基于查询的NES攻击"""
        adv_image = image.clone()
        sigma = 0.01
        lr = 0.01

        best_score = query_fn(image)
        best_adv = image.clone()

        queries_used = 0

        while queries_used < num_queries:
            # 估计梯度
            grad_estimate = torch.zeros_like(image)

            for _ in range(50):  # 每次迭代采样50个方向
                noise = torch.randn_like(image)

                # 正向查询
                pos_image = torch.clamp(adv_image + sigma * noise, 0, 1)
                score_pos = query_fn(pos_image)
                queries_used += 1

                # 反向查询
                neg_image = torch.clamp(adv_image - sigma * noise, 0, 1)
                score_neg = query_fn(neg_image)
                queries_used += 1

                # 累积梯度
                grad_estimate += (score_pos - score_neg) * noise

            grad_estimate /= (2 * sigma * 50)

            # 更新
            adv_image = adv_image + lr * grad_estimate.sign()
            perturbation = torch.clamp(adv_image - image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(image + perturbation, 0, 1)

            # 记录最佳结果
            current_score = query_fn(adv_image)
            queries_used += 1

            if current_score > best_score:
                best_score = current_score
                best_adv = adv_image.clone()

        return best_adv

    def semantic_attack(self, image, positive_texts):
        """语义对齐攻击：使图像特征与正面文本对齐"""
        adv_image = image.clone().requires_grad_(True)

        for surrogate in self.surrogate_models:
            model = surrogate['model']

            # 编码正面文本
            text_tokens = clip.tokenize(positive_texts)
            text_features = model.encode_text(text_tokens)
            text_features = text_features.mean(dim=0, keepdim=True)

            for _ in range(100):
                image_features = model.encode_image(adv_image)

                # 最大化图像与正面文本的相似度
                loss = -F.cosine_similarity(image_features, text_features).mean()

                grad = torch.autograd.grad(loss, adv_image)[0]
                adv_image = adv_image - 0.01 * grad.sign()

                perturbation = torch.clamp(adv_image - image, -self.epsilon, self.epsilon)
                adv_image = torch.clamp(image + perturbation, 0, 1)
                adv_image = adv_image.detach().requires_grad_(True)

        return adv_image.detach()


class UniversalPerturbationGenerator(nn.Module):
    """通用对抗扰动生成器"""

    def __init__(self, epsilon=8/255):
        super().__init__()
        self.epsilon = epsilon

        # U-Net风格的生成器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)
        perturbation = self.decoder(features)
        perturbation = perturbation * self.epsilon
        return torch.clamp(x + perturbation, 0, 1)
```

---

## 4. 实验设计

### 4.1 数据集

使用VIP5支持的数据集：
- Amazon Toys
- Amazon Beauty
- Amazon Sports
- Amazon Clothing

### 4.2 评估指标

```python
class AttackEvaluator:
    """攻击效果评估"""

    def __init__(self, model, original_rankings):
        self.model = model
        self.original_rankings = original_rankings

    def evaluate(self, adv_images, target_items):
        results = {
            'attack_success_rate': 0,  # 排名提升成功率
            'avg_rank_change': 0,       # 平均排名变化
            'top_k_rate': {},           # Top-K命中率
            'perturbation_norm': 0,     # 扰动大小
            'ssim': 0,                  # 结构相似度
            'lpips': 0,                 # 感知相似度
        }

        for adv_img, target in zip(adv_images, target_items):
            # 获取新排名
            new_ranking = self.model.get_ranking(adv_img)
            old_ranking = self.original_rankings[target]

            # 计算指标
            rank_change = old_ranking - new_ranking
            results['avg_rank_change'] += rank_change

            if rank_change > 0:
                results['attack_success_rate'] += 1

            for k in [1, 5, 10, 20]:
                if new_ranking <= k:
                    results['top_k_rate'][k] = results['top_k_rate'].get(k, 0) + 1

        # 平均化
        n = len(adv_images)
        results['attack_success_rate'] /= n
        results['avg_rank_change'] /= n
        for k in results['top_k_rate']:
            results['top_k_rate'][k] /= n

        return results
```

### 4.3 实验配置

```yaml
# experiment_config.yaml
attack_methods:
  - transfer_attack:
      surrogate_models: ['ViT-B/32', 'ViT-B/16', 'RN50']
      epsilon: [4/255, 8/255, 16/255]
      iterations: [50, 100, 200]

  - query_attack:
      num_queries: [1000, 5000, 10000]
      epsilon: [8/255]

  - semantic_attack:
      positive_texts: ['high quality', 'best seller', 'highly recommended']
      epsilon: [8/255]

evaluation:
  metrics: ['attack_success_rate', 'avg_rank_change', 'top_k_rate', 'ssim', 'lpips']
  top_k: [1, 5, 10, 20]

datasets:
  - name: 'toys'
    num_samples: 1000
  - name: 'beauty'
    num_samples: 1000
```

### 4.4 基线对比

| 方法 | 类型 | 查询次数 | 迁移性 |
|------|------|---------|--------|
| FGSM | 白盒 | 0 | 低 |
| PGD | 白盒 | 0 | 中 |
| MI-FGSM | 迁移 | 0 | 高 |
| NES | 查询 | 10000+ | N/A |
| Boundary Attack | 决策 | 10000+ | N/A |
| **Ours (Ensemble Transfer)** | 迁移 | 0 | 高 |
| **Ours (Query-efficient)** | 查询 | 1000 | N/A |

---

## 5. 防御建议

本研究同时提出以下防御措施：

### 5.1 输入净化
- JPEG压缩
- 高斯模糊
- 位深度缩减

### 5.2 对抗训练
```python
def adversarial_training(model, dataloader, attack_fn):
    for images, labels in dataloader:
        # 生成对抗样本
        adv_images = attack_fn(images)

        # 混合训练
        mixed_images = torch.cat([images, adv_images])
        mixed_labels = torch.cat([labels, labels])

        # 训练
        loss = model(mixed_images, mixed_labels)
        loss.backward()
```

### 5.3 特征去噪
- 在视觉特征层添加去噪模块
- 使用自编码器重建干净特征

### 5.4 检测机制
- 基于特征分布异常检测
- 基于排名一致性检测

---

## 6. 伦理声明

本研究仅用于学术目的，旨在：
1. 揭示多模态推荐系统的安全漏洞
2. 促进更鲁棒的推荐系统设计
3. 为AI安全研究社区提供参考

研究者应遵守相关法律法规，不得将攻击方法用于非法目的。

---

## 参考文献

1. Goodfellow et al. "Explaining and Harnessing Adversarial Examples" ICLR 2015
2. Carlini & Wagner "Towards Evaluating the Robustness of Neural Networks" S&P 2017
3. Dong et al. "Boosting Adversarial Attacks with Momentum" CVPR 2018
4. Ilyas et al. "Black-box Adversarial Attacks with Limited Queries" ICML 2018
5. Gong et al. "VIP5: Towards Multimodal Foundation Models for Recommendation" EMNLP 2023
