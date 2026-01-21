"""
对抗攻击方法模块
Adversarial Attack Methods for Multimodal Recommendation Systems
"""

from .transfer_attack import (
    TransferAttack,
    EnsembleTransferAttack,
    FeatureCollisionAttack
)
from .query_attack import (
    NESAttack,
    ZOSignSGDAttack,
    BoundaryAttack,
    SimBA,
    SquareAttack
)
from .semantic_attack import (
    SemanticAttack,
    MultimodalFusionAttack,
    GenerativeSemanticAttack
)

__all__ = [
    # 迁移攻击
    'TransferAttack',
    'EnsembleTransferAttack',
    'FeatureCollisionAttack',
    # 查询攻击
    'NESAttack',
    'ZOSignSGDAttack',
    'BoundaryAttack',
    'SimBA',
    'SquareAttack',
    # 语义攻击
    'SemanticAttack',
    'MultimodalFusionAttack',
    'GenerativeSemanticAttack'
]
