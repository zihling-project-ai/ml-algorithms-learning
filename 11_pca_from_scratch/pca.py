"""L08-B｜PCA 正式主線。
主線：固定切分 → Train 學 PCA → 比較維度 → 變異門檻選最少維度 → Test 還原評估。
原有小型數字推導與驗證保留於 pca_learning.py。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "component_counts": (1, 2),
    "variance_threshold": 0.95,
    "n_samples": 200,
    "shared_std": 2.0,
    "noise_std": 0.35,
}


def fit_pca(X, n_components):
    """輸入二維資料與保留方向數量，回傳中心與投影規則。

    本版共變異數除以 n，與 np.var 的預設一致；只中心化，不自動標準化。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必須是有樣本、有 Feature 的二維陣列")
    if not np.isfinite(X).all():
        raise ValueError("X 不能含有 NaN 或無限值")
    if isinstance(n_components, (bool, np.bool_)) or not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= X.shape[1]:
        raise ValueError("n_components 必須是 1 到 Feature 總數之間的整數")

    # ① 學習各欄中心，再計算每筆相對於中心的偏離。
    mean = np.mean(X, axis=0)
    centered = X - mean

    # ② 將各欄變異數及兩欄共同變動整理成矩陣。
    covariance_matrix = centered.T @ centered / len(X)

    # ③ 找出方向，並依對應變異數由大到小排序。
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # 理論上皆非負；修正浮點誤差造成的極小負值。
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # ④ 先用所有方向算比例，再取要保留的方向。
    total_variance = np.sum(eigenvalues)
    if total_variance > 0:
        ratios = eigenvalues / total_variance
    else:
        # 全部樣本相同時沒有變化；比例無定義，本版約定回傳零。
        ratios = np.zeros_like(eigenvalues)

    return {
        "mean": mean,
        "components": eigenvectors[:, :n_components],
        "explained_variance": eigenvalues[:n_components],
        "explained_variance_ratio": ratios[:n_components],
        "total_variance": total_variance,
    }


def transform_pca(X_new, model):
    """沿用訓練時的中心與方向，回傳 (新樣本數, 保留方向數)。"""
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim != 2 or X_new.shape[1] != len(model["mean"]):
        raise ValueError("X_new 必須是二維陣列，且 Feature 數須與訓練資料相同")
    if not np.isfinite(X_new).all():
        raise ValueError("X_new 不能含有 NaN 或無限值")

    # 沿用 fit_pca 學到的平均，不重新計算新資料的平均。
    centered = X_new - model["mean"]
    # 每欄是一個保留方向；每筆資料沿各方向取得一個位置。
    projected = centered @ model["components"]
    return projected


def inverse_transform_pca(projected, model):
    """將低維座標近似放回原 Feature 空間；無法恢復捨棄方向的細節。"""
    projected = np.asarray(projected, dtype=float)
    if projected.ndim != 2 or projected.shape[1] != model["components"].shape[1]:
        raise ValueError("projected 必須是二維陣列，且欄數等於保留的主成分數")
    if not np.isfinite(projected).all():
        raise ValueError("projected 不能含有 NaN 或無限值")
    # 先沿各保留方向放回原維度，再把中心移回訓練時的位置。
    reconstructed = projected @ model["components"].T + model["mean"]
    return reconstructed


def plot_pca_reconstruction(X_train, model):
    """只畫 Train；左圖看主方向，右圖看保留方向造成的近似還原。"""
    projected = transform_pca(X_train, model)
    reconstructed = inverse_transform_pca(projected, model)
    mean = model["mean"]
    # 使用同一份 Train 展示兩個主方向，沒有加入其他資料重新估計。
    full_model = fit_pca(X_train, n_components=X_train.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    low = X_train.min(axis=0) - 0.8
    high = X_train.max(axis=0) + 0.8
    for ax in axes:
        ax.set(xlabel="Feature 1", ylabel="Feature 2",
               xlim=(low[0], high[0]), ylim=(low[1], high[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    axes[0].scatter(X_train[:, 0], X_train[:, 1], s=24, alpha=0.55,
                    color="#2874a6", label="Original Train")
    for index, color in [(0, "#27844c"), (1, "#b66b30")]:
        direction = full_model["components"][:, index]
        # 線段總長是四個該方向標準差，僅用來展示方向與尺度。
        offsets = np.array([-2.0, 2.0]) * np.sqrt(full_model["explained_variance"][index])
        line = mean + offsets[:, None] * direction
        axes[0].plot(line[:, 0], line[:, 1], color=color, lw=2.5,
                     label=f"PC{index + 1} direction")
    axes[0].scatter(*mean, marker="X", s=110, c="black", label="Train mean", zorder=5)
    axes[0].set_title("Train distribution and principal directions")

    axes[1].scatter(X_train[:, 0], X_train[:, 1], s=24, alpha=0.28,
                    color="#2874a6", label="Original Train")
    for point, restored in zip(X_train, reconstructed):
        axes[1].plot([point[0], restored[0]], [point[1], restored[1]],
                     color="#999999", alpha=0.45, lw=0.7)
    axes[1].scatter(reconstructed[:, 0], reconstructed[:, 1], s=22,
                    c="#d77a24", alpha=0.8, label="Reconstructed Train", zorder=3)
    axes[1].scatter(*mean, marker="X", s=110, c="black", zorder=5, label="Train mean")
    retained_ratio = np.sum(model["explained_variance_ratio"])
    axes[1].set_title(f"Keep {model['components'].shape[1]} component(s): {retained_ratio:.2%} variance")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / "pca_projection_reconstruction.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


# ==================== 正式實驗 ====================

def main():
    # ① 建立資料
    rng = np.random.default_rng(CONFIG["data_seed"])

    # 共同變化：同一筆的兩欄都受到這個數字影響。
    shared = rng.normal(0.0, CONFIG["shared_std"], size=CONFIG["n_samples"])
    # 個別擾動：每筆兩欄各有自己的偏離。
    noise = rng.normal(
        0.0, CONFIG["noise_std"], size=(CONFIG["n_samples"], 2),
    )

    feature_1 = 10.0 + shared + noise[:, 0]
    feature_2 = 5.0 + 0.8 * shared + noise[:, 1]
    X = np.column_stack([feature_1, feature_2])

    print("資料形狀：", X.shape)
    print("共同變化形狀：", shared.shape)
    print("個別擾動形狀：", noise.shape)
    print("前 5 筆資料：\n", X[:5])
    assert X.shape == (CONFIG["n_samples"], 2)
    assert np.isfinite(X).all()

    # 此圖只觀察合成資料，不據此調整模型或超參數。
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.65, s=30, color="#2874a6")
    ax.set(xlabel="Feature 1", ylabel="Feature 2",
           title="PCA dataset: shared variation + individual noise")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / "pca_dataset.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print("資料分布圖已儲存：", output_path)


    # ==================== ② 固定切分；只用 Train 學投影規則 ====================
    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))
    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]
    X_train = X[train_indices]
    X_validation = X[validation_indices]
    X_test = X[test_indices]

    print("\nTrain：", X_train.shape)
    print("Validation：", X_validation.shape)
    print("Test：", X_test.shape)

    assert len(np.unique(np.concatenate([
        train_indices, validation_indices, test_indices,
    ]))) == len(X)

    # ==================== ③ 比較主成分數量：壓縮與還原誤差 ====================
    # 每組只從 Train 學中心與方向；Validation 只套用規則。
    # 本實驗兩欄尺度可比較，只中心化，不額外標準化。
    results = {}
    for n_components in CONFIG["component_counts"]:
        model = fit_pca(X_train, n_components=n_components)
        train_projected = transform_pca(X_train, model)
        validation_projected = transform_pca(X_validation, model)
        train_reconstructed = inverse_transform_pca(train_projected, model)
        validation_reconstructed = inverse_transform_pca(validation_projected, model)

        # 對全部樣本、全部 Feature 的平方偏差取平均。
        train_mse = np.mean((X_train - train_reconstructed) ** 2)
        validation_mse = np.mean((X_validation - validation_reconstructed) ** 2)
        retained_ratio = np.sum(model["explained_variance_ratio"])
        results[n_components] = {
            "model": model,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "retained_ratio": retained_ratio,
        }
        print(
            f"保留 {n_components} 維：Train 變異保留={retained_ratio:.2%}，"
            f"Train MSE={train_mse:.8f}，Validation MSE={validation_mse:.8f}"
        )
        print("投影形狀：", train_projected.shape, validation_projected.shape)

        assert np.allclose(model["mean"], np.mean(X_train, axis=0))
        assert train_projected.shape == (len(X_train), n_components)
        assert validation_projected.shape == (len(X_validation), n_components)
        assert train_reconstructed.shape == X_train.shape
        assert validation_reconstructed.shape == X_validation.shape
        assert np.isfinite([train_mse, validation_mse]).all()
        assert np.allclose(model["components"].T @ model["components"], np.eye(n_components))
        discarded_variance = model["total_variance"] - np.sum(model["explained_variance"])
        assert np.isclose(train_mse * X_train.shape[1], discarded_variance)
        if n_components == X_train.shape[1]:
            # 保留全部正交方向時，可在浮點精度範圍內完整還原新資料。
            assert np.allclose(validation_reconstructed, X_validation)

    # 還原誤差隨保留維度增加不會變大；但保留全部維度也沒有降維。
    # 不直接用最小 MSE 選模，改用事先設定的變異保留需求。


    # ==================== ④ 變異門檻：選達標的最少維度 ====================
    # 95% 是此實驗的需求，不是 PCA 的固定規定；只依 Train 比例選擇。
    selected_n_components = None
    for n_components in sorted(results):
        if results[n_components]["retained_ratio"] >= CONFIG["variance_threshold"]:
            selected_n_components = n_components
            break

    if selected_n_components is None:
        raise ValueError("目前候選維度沒有達到變異保留門檻")

    selected_result = results[selected_n_components]
    selected_model = selected_result["model"]
    print("\n變異保留門檻：", CONFIG["variance_threshold"])
    print("選定維度：", selected_n_components)
    print("選定 Train 變異保留比例：", selected_result["retained_ratio"])
    print("選定 Validation 還原 MSE：", selected_result["validation_mse"])

    # ==================== ⑤ Test：沿用已選定的模型 ====================
    test_projected = transform_pca(X_test, selected_model)
    test_reconstructed = inverse_transform_pca(test_projected, selected_model)
    test_mse = np.mean((X_test - test_reconstructed) ** 2)
    print("Test 降維前後：", X_test.shape, "→", test_projected.shape)
    print("Test 還原 MSE：", test_mse)
    assert test_projected.shape == (len(X_test), selected_n_components)
    assert test_reconstructed.shape == X_test.shape
    assert np.isfinite(test_mse)
    # 不使用 Test 結果回頭調整門檻或選定維度。


    # ==================== ⑥ 主方向與近似還原圖 ====================
    projection_figure = plot_pca_reconstruction(X_train, selected_model)
    print("主方向與近似還原圖已儲存：", projection_figure)


if __name__ == "__main__":
    main()
