"""
單元測試：synthetic.auto_params（依樣本數自動調整訓練參數）

涵蓋：各分段邊界、關閉自動參數時回傳原值、batch_size 為 10 的倍數且不超過資料量、
TabDDPM 高維資料縮減 batch 與 timesteps 對齊論文建議值。
"""

import pytest

from synthetic.auto_params import (
    _auto_batch_ctgan,
    _auto_batch_tabddpm,
    _auto_batch_tvae,
    _auto_dims_ctgan,
    _auto_dims_tvae,
    _auto_discriminator_steps_ctgan,
    _auto_epochs_ctgan,
    _auto_epochs_tvae,
    _auto_n_iter_tabddpm,
    _auto_timesteps_tabddpm,
    _get_batch_sizes_to_try,
    _get_batch_sizes_to_try_tvae,
)


@pytest.mark.parametrize("n, expected", [
    (49, 4500), (50, 3000), (99, 3000), (100, 1500), (299, 1500),
    (300, 900), (499, 900), (500, 600), (999, 600), (1000, 450), (1999, 450), (2000, 300),
])
def test_ctgan_epochs_segments(n, expected):
    assert _auto_epochs_ctgan(n, 300) == expected


def test_ctgan_epochs_are_capped():
    assert _auto_epochs_ctgan(10, 1000) == 5000


@pytest.mark.parametrize("n, expected", [
    (49, 6000), (99, 4500), (299, 3000), (499, 2100), (999, 1500), (1999, 900), (5000, 600),
])
def test_tvae_epochs_segments(n, expected):
    assert _auto_epochs_tvae(n, 300) == expected


@pytest.mark.parametrize("n, expected", [
    (10, (32, 32)), (60, (64, 64)), (200, (128, 128)), (400, (192, 192)), (800, (256, 256)), (5000, (256, 256)),
])
def test_ctgan_dims_segments(n, expected):
    assert _auto_dims_ctgan(n, 256) == expected


def test_ctgan_dims_cap_at_512_and_accept_list():
    assert _auto_dims_ctgan(5000, [1024, 1024]) == (512, 512)


@pytest.mark.parametrize("n, expected", [
    (10, (32, 32)), (60, (64, 64)), (200, (64, 64)), (400, (96, 96)), (800, (128, 128)), (5000, (256, 256)),
])
def test_tvae_dims_segments(n, expected):
    assert _auto_dims_tvae(n, 512) == expected


@pytest.mark.parametrize("n, expected", [(50, 3), (99, 3), (100, 2), (499, 2), (500, 1)])
def test_ctgan_discriminator_steps(n, expected):
    assert _auto_discriminator_steps_ctgan(n, 1) == expected


@pytest.mark.parametrize("fn, args, base", [
    (_auto_epochs_ctgan, (100, 123), 123),
    (_auto_epochs_tvae, (100, 123), 123),
    (_auto_discriminator_steps_ctgan, (100, 5), 5),
    (_auto_n_iter_tabddpm, (100, 777), 777),
    (_auto_timesteps_tabddpm, (321,), 321),
])
def test_disabled_auto_params_return_base(fn, args, base):
    assert fn(*args, use_auto_params=False) == base


def test_disabled_auto_dims_return_base_pair():
    assert _auto_dims_ctgan(10, 200, use_auto_params=False) == (200, 200)
    assert _auto_dims_tvae(10, 200, use_auto_params=False) == (200, 200)


@pytest.mark.parametrize("fn", [_auto_batch_ctgan, _auto_batch_tvae])
@pytest.mark.parametrize("n, base", [(15, 500), (100, 500), (1000, 500), (5000, 500), (100, 37)])
def test_batch_is_multiple_of_10_and_fits_data(fn, n, base):
    bs = fn(n, base)
    assert bs % 10 == 0
    assert bs >= 10
    assert bs <= max(10, n // 2)


@pytest.mark.parametrize("fn", [_auto_batch_ctgan, _auto_batch_tvae])
def test_batch_with_no_data_returns_base(fn):
    assert fn(0, 500) == 500


@pytest.mark.parametrize("fn", [_get_batch_sizes_to_try, _get_batch_sizes_to_try_tvae])
def test_batch_candidates_descending_multiples_of_10(fn):
    sizes = fn(200, 500)
    assert sizes == sorted(sizes, reverse=True)
    assert all(s % 10 == 0 and 10 <= s < 200 for s in sizes)
    assert sizes[-1] == 10


@pytest.mark.parametrize("n, base, expected", [(500, 1000, 5000), (500, 8000, 8000), (5000, 1000, 10000)])
def test_tabddpm_n_iter_minimum(n, base, expected):
    assert _auto_n_iter_tabddpm(n, base) == expected


@pytest.mark.parametrize("n, n_features, expected", [
    (40, 0, 40), (80, 0, 64), (200, 0, 128), (800, 0, 256), (5000, 0, 1024),
    (80, 100, 32), (400, 100, 64), (800, 100, 128), (5000, 100, 256),
])
def test_tabddpm_batch(n, n_features, expected):
    assert _auto_batch_tabddpm(n, 1024, n_features=n_features) == expected


@pytest.mark.parametrize("user, expected", [(50, 100), (550, 100), (551, 1000), (2000, 1000)])
def test_tabddpm_timesteps_snap_to_paper_values(user, expected):
    assert _auto_timesteps_tabddpm(user) == expected
