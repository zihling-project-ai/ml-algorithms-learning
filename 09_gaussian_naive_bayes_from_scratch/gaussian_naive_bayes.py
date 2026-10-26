"""L07｜Gaussian Naive Bayes From Scratch：多特徵正式主線。
資料 → 固定切分 → Train 統計量 → Log 域預測 → Accuracy／Baseline → Test → 圖表。
公式展開與 40°C 小例子保留在 gaussian_naive_bayes_learning.py。
假設輸入為非空有限二維數值資料，訓練標籤與樣本對應，變異數下限為正。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ==================== 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "n_normal": 120,
    "n_abnormal": 80,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "variance_floor": 1e-6,
}

# ==================== 核心模型與數值穩定性 ====================
# 訓練：Prior_k=n_k/n；μ_kj=mean(X_k[:,j])；v_kj=mean((X_k[:,j]-μ_kj)²)。
# Gaussian 假設：各類每個 Feature 以常態分布近似。
# Naive 假設：給定類別後，Features 條件獨立。
# score_k = Prior_k × Π_j f(x_j|y=k)
# log_score_k = log(Prior_k) + Σ_j log f(x_j|y=k)
# posterior_k = score_k / Σ_c score_c；ŷ=分數最大者的類別。
#
# 【數值保護】變異數下限會調整過小的統計量，不是分布定義的固定常數。
# 【等價計算】Log 密度與減最大值正規化避免下溢，不增加分類假設。
# axis=0：訓練時跨樣本逐欄估計；axis=1：預測時同類各 Feature 證據加總。

def fit_gaussian_naive_bayes_multifeature(X, y, variance_floor=1e-6):
    """X：(樣本數, Features)；回傳各類 Prior 與逐欄平均、變異數。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("需要非空的二維 X 與等長一維 y")
    if not np.isfinite(X).all() or not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("X 必須有限，variance_floor 必須是有限正數")
    classes = np.unique(y)
    priors = []
    means = []
    variances = []


    for label in classes:
        mask = y == label
        members = X[mask]

        # Prior_k = n_k / n：每類一個比例。
        prior = len(members) / len(y)

        # μ_kj = (1/n_k) × Σ_i X_ij
        # axis=0：跨樣本逐欄計算，不把溫度與振動混在一起平均。
        feature_means = np.mean(members, axis=0)

        # v_kj = (1/n_k) × Σ_i (X_ij - μ_kj)²
        raw_variances = np.var(members, axis=0)
        # 【數值保護】每一欄各自套用變異數下限。
        safe_variances = np.maximum(raw_variances, variance_floor)

        priors.append(prior)
        means.append(feature_means)
        variances.append(safe_variances)


    return {
        "classes": classes,
        "priors": np.array(priors),
        "means": np.array(means),
        "variances": np.array(variances),
    }


def predict_one_gaussian_naive_bayes_multifeature(point, model):
    means = model["means"]
    variances = model["variances"]

    # 輸入 point：(Features,)；means、variances：(類別數, Features)。
    # Broadcasting：同一筆 point 分別與每一類的統計量計算。
    # 每格的公式：log f(x_j|y=k)
    # = -0.5 × [log(2πv_kj) + (x_j-μ_kj)²/v_kj]
    log_likelihoods = -0.5 * (
        np.log(2 * np.pi * variances)
        + (point - means) ** 2 / variances
    )

    # 【模型核心】條件獨立近似：score_k = Prior_k × Π_j f(x_j|y=k)
    # Π 表示相乘；取 Log 後：log_score_k = log Prior_k + Σ_j log f(x_j|y=k)。
    # log_likelihoods 每列為一類、每欄為一個 Feature。
    # axis=1：橫向加總同類的溫度、振動證據，得到每類一個分數。
    log_scores = (
        np.log(model["priors"])
        + np.sum(log_likelihoods, axis=1)
    )

    # 【數值穩定性】平移後正規化；模型規則不變。
    shifted_scores = log_scores - np.max(log_scores)
    relative_scores = np.exp(shifted_scores)
    posteriors = relative_scores / np.sum(relative_scores)

    best_index = np.argmax(log_scores)
    prediction = model["classes"][best_index]
    return prediction, posteriors


def predict_gaussian_naive_bayes_multifeature(X_new, model):
    predictions = []
    posterior_rows = []

    for point in X_new:
        prediction, posteriors = (
            predict_one_gaussian_naive_bayes_multifeature(
                point,
                model,
            )
        )

        predictions.append(prediction)
        posterior_rows.append(posteriors)

    return np.array(predictions), np.array(posterior_rows)


# ==================== 視覺化：決策區域與後驗機率 ====================

def save_decision_boundary(X_train, y_train, model):
    # 只用 Train 決定圖面範圍，Test 不參與圖表範圍選擇。
    low = X_train.min(axis=0)
    high = X_train.max(axis=0)
    padding = (high - low) * 0.08
    temperature_axis = np.linspace(low[0] - padding[0], high[0] + padding[0], 180)
    vibration_axis = np.linspace(low[1] - padding[1], high[1] + padding[1], 180)
    grid_x, grid_y = np.meshgrid(temperature_axis, vibration_axis)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    labels, posteriors = predict_gaussian_naive_bayes_multifeature(points, model)
    abnormal_index = np.flatnonzero(model["classes"] == 1)[0]
    abnormal_p = posteriors[:, abnormal_index].reshape(grid_x.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].contourf(grid_x, grid_y, labels.reshape(grid_x.shape),
                     levels=[-0.5, 0.5, 1.5], colors=["tab:blue", "tab:orange"], alpha=0.25)
    shading = axes[1].contourf(grid_x, grid_y, abnormal_p,
                              levels=np.linspace(0, 1, 21), cmap="RdBu_r", vmin=0, vmax=1)
    fig.colorbar(shading, ax=axes[1], label="Model P(Abnormal | Features)")
    for ax in axes:
        ax.contour(grid_x, grid_y, abnormal_p, levels=[0.5], colors="black", linestyles="--")
        for label, color, name in ((0,"tab:blue","Train Normal"), (1,"tab:orange","Train Abnormal")):
            members = X_train[y_train == label]
            ax.scatter(members[:, 0], members[:, 1], s=24, color=color,
                       edgecolors="black", linewidths=0.4, label=name)
        ax.set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Vibration")
    axes[0].set_title("Gaussian Naive Bayes Decision Regions")
    axes[1].set_title("Posterior Under Model Assumptions")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / "gaussian_naive_bayes_boundary.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("決策邊界圖已儲存：", output_path)


# ==================== 正式實驗入口 ====================

def main():
    # ==================== 1. 正式實驗：資料與固定切分 ====================
    # 本例刻意以各類常態分布、類內獨立的 Features 生成，符合模型假設。
    # 分布互有重疊，後續以未參與統計量估計的資料評估。

    rng_data = np.random.default_rng(CONFIG["data_seed"])

    # loc：各欄平均；scale：各欄標準差（不是變異數）。
    # 欄位順序固定為 [溫度, 振動]。
    X_normal = rng_data.normal(
        loc=[33.0, 0.03],
        scale=[3.0, 0.015],
        size=(CONFIG["n_normal"], 2),
    )
    X_abnormal = rng_data.normal(
        loc=[39.0, 0.07],
        scale=[4.0, 0.02],
        size=(CONFIG["n_abnormal"], 2),
    )

    # axis=0：上下合併樣本，不改變欄位數量。
    X = np.concatenate((X_normal, X_abnormal), axis=0)

    # 標籤代表由哪一類分布生成，不是依某個溫度門檻計算。
    # 標籤與 X 的樣本順序一一對應。
    y_experiment = np.concatenate((
        np.zeros(CONFIG["n_normal"], dtype=np.int64),
        np.ones(CONFIG["n_abnormal"], dtype=np.int64),
    ))

    # 固定樣本編號的順序，再讓 X 與標籤使用相同切片。
    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))

    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    X_train = X[train_indices]
    y_train = y_experiment[train_indices]
    X_validation = X[validation_indices]
    y_validation = y_experiment[validation_indices]
    X_test = X[test_indices]
    y_test = y_experiment[test_indices]

    print("\n【正式實驗資料】")
    print("Train：", X_train.shape, y_train.shape)
    print("Validation：", X_validation.shape, y_validation.shape)
    print("Test：", X_test.shape, y_test.shape)

    assert X.shape == (len(y_experiment), 2)
    assert len(X_train) == len(y_train)
    assert len(X_validation) == len(y_validation)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_validation) + len(X_test) == len(X)
    # 至此只完成切分，下一步只用 Train 估計統計量。


    # ==================== 2. 正式訓練與 Training／Validation 評估 ====================

    # ① 只使用 Train 估計統計量。
    experiment_model = fit_gaussian_naive_bayes_multifeature(
        X_train,
        y_train,
        variance_floor=CONFIG["variance_floor"],
    )

    # ② 同一個模型，分別預測兩份資料。
    train_predictions, train_posteriors = (
        predict_gaussian_naive_bayes_multifeature(
            X_train,
            experiment_model,
        )
    )

    validation_predictions, validation_posteriors = (
        predict_gaussian_naive_bayes_multifeature(
            X_validation,
            experiment_model,
        )
    )

    # ③ 逐筆比較是否答對，再取平均。
    train_accuracy = np.mean(train_predictions == y_train)

    validation_accuracy = np.mean(
        validation_predictions == y_validation
    )

    print("Training Accuracy：", train_accuracy)
    print("Validation Accuracy：", validation_accuracy)

    assert train_predictions.shape == y_train.shape
    assert validation_predictions.shape == y_validation.shape

    # ==================== 3. Majority-Class Baseline ====================

    train_counts = np.bincount(y_train, minlength=2)
    majority_class = np.argmax(train_counts)

    baseline_predictions = np.full(
        y_validation.shape,
        majority_class,
        dtype=np.int64,
    )

    baseline_accuracy = np.mean(
        baseline_predictions == y_validation
    )

    print("Train 各類筆數：", train_counts)
    print("Baseline 固定預測類別：", majority_class)
    print("Baseline Validation Accuracy：", baseline_accuracy)


    # ==================== 4. 最終 Test 評估 ====================
    # 設定固定，未進行超參數搜尋；Test 不參與統計量估計或調參。

    test_predictions, test_posteriors = (
        predict_gaussian_naive_bayes_multifeature(
            X_test,
            experiment_model,
        )
    )

    test_accuracy = np.mean(test_predictions == y_test)

    print("Test Accuracy：", test_accuracy)

    assert test_predictions.shape == y_test.shape

    # 圖表與主線分開，僅用 Train 決定顯示範圍與繪製樣本。
    save_decision_boundary(X_train, y_train, experiment_model)


if __name__ == "__main__":
    main()
