# FAS-Aug パッケージ初期化ファイル

"""
FAS-Aug: Face Anti-Spoofing Augmentation package
 
この パッケージは顔認証のスプーフィング攻撃検出のためのデータ拡張機能を提供します。
"""

__version__ = "1.0.0"
__author__ = "FAS-Aug Team"

# 主要なクラスとメソッドをインポート
try:
    from .data.FAS_Augmentations import FAS_Augmentations
    from .data.transform import random_parse_policies, MultiAugmentation, get_basetransform
    
    __all__ = [
        'FAS_Augmentations',
        'random_parse_policies', 
        'MultiAugmentation',
        'get_basetransform'
    ]
except ImportError as e:
    print(f"FAS-Aug パッケージの初期化中にエラーが発生しました: {e}")
    __all__ = [] 