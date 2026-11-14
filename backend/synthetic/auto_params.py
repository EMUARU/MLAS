#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic.auto_params

依樣本規模之自動參數機制：CTGAN/TVAE/TabDDPM 三生成器的分段參數規則。

"""

# 依樣本規模之自動參數分段規則（無外部相依）


def _auto_epochs_ctgan(n: int, base_epochs: int, use_auto_params: bool = True) -> int:
    """CTGAN 自動 epochs 調整。

    【論文依據】GAN 對抗訓練需要足夠的迭代:
    - 小數據集: 更多 epochs (收斂更慢)
    - 大數據集: 較少 epochs (已有足夠樣本)
    """
    if not use_auto_params:
        return int(base_epochs)
    base = int(base_epochs)
    if n < 50:
        return int(min(base * 15, 5000))
    if n < 100:
        return int(min(base * 10, 4000))
    if n < 300:
        return int(min(base * 5, 3000))
    if n < 500:
        return int(min(base * 3, 2000))
    if n < 1000:
        return int(min(base * 2, 1500))
    if n < 2000:
        return int(min(base * 1.5, 1000))
    return int(min(base, 800))


def _auto_dims_ctgan(n: int, base_generator_dim, use_auto_params: bool = True) -> tuple:
    """CTGAN 自動網路維度調整。

    【論文依據】網路容量應與數據複雜度匹配:
    - 小數據集: 較小網路防止過擬合
    - 大數據集: 較大網路捕捉複雜模式
    """
    # 解析 generator_dim 的值
    if isinstance(base_generator_dim, (list, tuple)):
        base_dim = int(base_generator_dim[0])
    else:
        base_dim = int(base_generator_dim)

    if not use_auto_params:
        return (base_dim, base_dim)

    if n < 50:
        return (32, 32)
    if n < 100:
        return (64, 64)
    if n < 300:
        return (128, 128)
    if n < 500:
        return (192, 192)
    if n < 1000:
        return (256, 256)
    return (min(base_dim, 512), min(base_dim, 512))


def _auto_discriminator_steps_ctgan(n: int, base_steps: int, use_auto_params: bool = True) -> int:
    """CTGAN 自動 discriminator_steps 調整。

    【論文依據】discriminator 訓練步數影響穩定性:
    - 小數據集: 更多步數讓 discriminator 更強
    - 大數據集: 標準 1 步
    """
    if not use_auto_params:
        return int(base_steps)
    if n < 100:
        return 3
    if n < 500:
        return 2
    return 1


def _auto_batch_ctgan(n: int, base_batch_size: int) -> int:
    """CTGAN 自動 batch_size 調整 (取為 10 的倍數)。

    依據：SDV CTGANSynthesizer 要求 batch_size 須能被 pac 整除；
          pac 預設為 10，故此處取 10 的倍數以相容預設行為。
    """
    bs = int(base_batch_size)
    if n <= 0:
        return bs
    max_allowed = max(10, (n // 2) // 10 * 10)
    result = min(bs, max_allowed)
    result = max(10, result // 10 * 10)
    return int(result)


def _get_batch_sizes_to_try(n: int, preferred_bs: int) -> list:
    """獲取安全的 batch_size 嘗試列表 (從大到小，10 的倍數)。"""
    candidates = set()
    if preferred_bs > 0:
        safe_pref = min(preferred_bs, n - 1)
        safe_pref = safe_pref // 10 * 10
        if safe_pref >= 10:
            candidates.add(safe_pref)
    for divisor in [2, 3, 4, 5, 10]:
        val = (n // divisor) // 10 * 10
        if val >= 10:
            candidates.add(val)
    candidates.add(10)
    return sorted(candidates, reverse=True)


def _auto_epochs_tvae(n: int, base_epochs: int, use_auto_params: bool = True) -> int:
    """TVAE 自動 epochs 調整。

    【論文依據】VAE 訓練收斂較慢，需要更多 epochs:
    - 小數據集: 更多 epochs 確保收斂
    - 大數據集: 適當減少避免過擬合
    """
    if not use_auto_params:
        return int(base_epochs)
    base = int(base_epochs)
    if n < 50:
        return int(min(base * 20, 10000))
    if n < 100:
        return int(min(base * 15, 8000))
    if n < 300:
        return int(min(base * 10, 6000))
    if n < 500:
        return int(min(base * 7, 5000))
    if n < 1000:
        return int(min(base * 5, 4000))
    if n < 2000:
        return int(min(base * 3, 3000))
    return int(min(base * 2, 2000))


def _auto_dims_tvae(n: int, base_compress_dim, use_auto_params: bool = True) -> tuple:
    """TVAE 自動網路維度調整。

    【論文依據】VAE 網路結構應與數據複雜度匹配。
    """
    if isinstance(base_compress_dim, (list, tuple)):
        base_dim = int(base_compress_dim[0])
    else:
        base_dim = int(base_compress_dim)

    if not use_auto_params:
        return (base_dim, base_dim)

    if n < 50:
        return (32, 32)
    if n < 100:
        return (64, 64)
    if n < 300:
        return (64, 64)
    if n < 500:
        return (96, 96)
    if n < 1000:
        return (128, 128)
    return (min(base_dim, 256), min(base_dim, 256))


def _auto_batch_tvae(n: int, base_batch_size: int) -> int:
    """TVAE 自動 batch_size 調整 (取為 10 的倍數，與 CTGAN 對齊)。"""
    bs = int(base_batch_size)
    if n <= 0:
        return bs
    max_allowed = max(10, (n // 2) // 10 * 10)
    result = min(bs, max_allowed)
    result = max(10, result // 10 * 10)
    return int(result)


def _get_batch_sizes_to_try_tvae(n: int, preferred_bs: int) -> list:
    """TVAE 之 batch_size 嘗試列表（與 CTGAN 邏輯相同，獨立函數以便未來分化）。"""
    candidates = set()
    if preferred_bs > 0:
        safe_pref = min(preferred_bs, n - 1)
        safe_pref = safe_pref // 10 * 10
        if safe_pref >= 10:
            candidates.add(safe_pref)
    for divisor in [2, 3, 4, 5, 10]:
        val = (n // divisor) // 10 * 10
        if val >= 10:
            candidates.add(val)
    candidates.add(10)
    return sorted(candidates, reverse=True)


def _auto_n_iter_tabddpm(n: int, base_n_iter: int, use_auto_params: bool = True) -> int:
    """TabDDPM 自動 n_iter 調整。

    【論文依據】Kotelnikov et al. ICML 2023 Table 1 建議: 5000、10000 或 20000。
    """
    if not use_auto_params:
        return int(base_n_iter)
    base = int(base_n_iter)
    if n < 1000:
        return max(base, 5000)
    return max(base, 10000)


def _auto_batch_tabddpm(n: int, base_batch_size: int, n_features: int = 0,
                         use_auto_params: bool = True) -> int:
    """TabDDPM 自動 batch_size 調整。

    【論文依據】Table 1 建議 Cat{256, 4096}；高維資料需縮減以避免 GPU OOM。
    """
    if not use_auto_params:
        return int(base_batch_size)
    bs = int(base_batch_size)
    if n <= 0:
        return bs
    # 高維資料：縮減以避免 GPU 記憶體溢出
    if n_features > 80:
        if n < 100:
            return min(bs, 32)
        if n < 500:
            return min(bs, 64)
        if n < 1000:
            return min(bs, 128)
        return min(bs, 256)
    # 一般情況
    if n < 50:
        return min(bs, max(32, n))
    if n < 100:
        return min(bs, max(64, n // 2))
    if n < 256:
        return min(bs, 128)
    if n < 1000:
        return min(bs, 256)
    return min(bs, 4096)


def _auto_timesteps_tabddpm(user_timesteps: int, use_auto_params: bool = True) -> int:
    """TabDDPM 自動 timesteps 調整。

    【論文依據】Table 1 建議 Cat{100, 1000}。將非標準值對齊至最近建議值。
    """
    if not use_auto_params:
        return int(user_timesteps)
    if user_timesteps <= 550:
        return 100
    return 1000


# ============================================================
# 分群訓練 CTGAN（對照論文 CTGAN_pos.ipynb / CTGAN_neg.ipynb）
# ============================================================
