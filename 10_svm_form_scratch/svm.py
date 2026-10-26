"""L08｜Linear SVM 主線實驗。
資料 → 固定切分 → Train 標準化 → 比較 C → Validation 選模 → Test → 邊界圖。
輔助示範見 svm_learning.py；完整圖解與推導見 SVM_DERIVATION_NOTES.md。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==================== 1. 主線實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "learning_rate": 0.05,
    "iterations": 500,
    "C_values": (0.1, 1.0, 10.0),
}


# ==================== 2. 分類分數 ====================

def decision_function(X, weights, bias):
    """X：(樣本數, Features)，weights：(Features,)，回傳每筆分數。"""
    # 第 i 筆：s_i = Σ_j w_j x_ij + b。
    # @：每列 Features 與係數相乘加總，不是逐元素乘法。
    # 分數為 0 的位置是分類邊界；分數不是機率或一般情況下的幾何距離。
    return X @ weights + bias


# ==================== 3. 計算訓練總目標 ====================

def compute_svm_loss(X, y_signed, weights, bias, C):
    scores = decision_function(X, weights, bias)

    # y_signed 為真實類別：Normal=-1、Abnormal=+1。
    # y*s<0：分錯；0<y*s<1：分對但間隔不足；y*s>=1：達到間隔要求。
    signed_scores = y_signed * scores
    hinge_losses = np.maximum(0.0, 1.0 - signed_scores)

    # J = 0.5 × Σ_j w_j² + C × mean(Hinge Loss)。
    # 此版本以「平均」而非「總和」計算 Hinge，C 的尺度需配合定義。
    weight_penalty = 0.5 * np.sum(weights ** 2)
    mean_hinge_loss = np.mean(hinge_losses)
    total_loss = weight_penalty + C * mean_hinge_loss

    return total_loss, weight_penalty, mean_hinge_loss


# ==================== 4. 訓練：梯度與完整資料批次更新 ====================

def compute_svm_gradients(X, y_signed, weights, bias, C):
    scores = decision_function(X, weights, bias)
    active = y_signed * scores < 1
    hinge_weight_sum = np.zeros_like(weights)
    hinge_bias_sum = 0.0

    # 未達標才有 Hinge 貢獻；全部使用同一組目前參數計算。
    for point, label, is_active in zip(X, y_signed, active):
        if is_active:
            hinge_weight_sum += -label * point
            hinge_bias_sum += -label

    # weights 是 0.5*||w||² 的梯度；bias 沒有這項懲罰。
    # 平均分母為全部樣本數，與 compute_svm_loss 的定義一致。
    grad_weights = weights + C * hinge_weight_sum / len(X)
    grad_bias = C * hinge_bias_sum / len(X)
    return grad_weights, grad_bias


def fit_linear_svm(X, y_signed, C, learning_rate, iterations):
    """輸入已處理尺度的資料，回傳 weights、bias、起始與各輪總目標。"""
    weights = np.zeros(X.shape[1])
    bias = 0.0
    initial_loss, _, _ = compute_svm_loss(X, y_signed, weights, bias, C)
    loss_history = [initial_loss]

    for _ in range(iterations):
        grad_weights, grad_bias = compute_svm_gradients(
            X, y_signed, weights, bias, C,
        )
        weights = weights - learning_rate * grad_weights
        bias = bias - learning_rate * grad_bias
        loss, _, _ = compute_svm_loss(X, y_signed, weights, bias, C)
        loss_history.append(loss)

    # Hinge 在 y*s=1 不可微；此實作選 active=False，使用允許的次梯度。
    # 固定步幅次梯度法可能在轉折附近震盪，不要求每輪目標嚴格下降。
    return weights, bias, np.array(loss_history)



def plot_svm_boundary(X_train, y_train, weights, bias, train_mean, train_std, C):
    """以原始溫度、振動畫圖；查詢點沿用 Train 尺度後再算分數。"""
    temperature = np.linspace(X_train[:, 0].min() - 2, X_train[:, 0].max() + 2, 250)
    vibration = np.linspace(X_train[:, 1].min() - 0.01, X_train[:, 1].max() + 0.01, 250)
    grid_t, grid_v = np.meshgrid(temperature, vibration)
    query_points = np.column_stack([grid_t.ravel(), grid_v.ravel()])
    query_scaled = (query_points - train_mean) / train_std
    grid_scores = decision_function(query_scaled, weights, bias).reshape(grid_t.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(grid_t, grid_v, (grid_scores >= 0).astype(int),
                levels=[-0.5, 0.5, 1.5], colors=["#dcebf7", "#ffe4ce"], alpha=0.7)
    contours = ax.contour(grid_t, grid_v, grid_scores, levels=[-1, 0, 1],
                          colors=["#2874a6", "black", "#ca6f1e"],
                          linestyles=["--", "-", "--"], linewidths=[1.7, 2.2, 1.7])
    ax.clabel(contours, fmt={-1: "score = -1", 0: "score = 0", 1: "score = +1"})
    for label, color, name in [(-1, "#2874a6", "Train Normal (-1)"),
                                (1, "#e67e22", "Train Abnormal (+1)")]:
        mask = y_train == label
        ax.scatter(X_train[mask, 0], X_train[mask, 1], c=color, edgecolors="white",
                   s=45, linewidths=0.5, label=name)
    # 標記目前未達間隔要求的點；不是精確求解後的 Support Vector 判定。
    train_scores = decision_function((X_train - train_mean) / train_std, weights, bias)
    active = y_train * train_scores < 1
    ax.scatter(X_train[active, 0], X_train[active, 1], facecolors="none",
               edgecolors="black", s=95, linewidths=0.9, label="Margin violations (y * score < 1)")
    ax.set(xlabel="Temperature (C)", ylabel="Vibration",
           title=f"Linear SVM: boundary and margins (C={C:g})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / "svm_boundary_margin.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


# ==================== 5. 主線實驗 ====================

def main():
    # ==================== ① 建立資料、固定切分與標準化 ====================
    # 每列一筆設備資料；兩欄依序是溫度、振動。
    rng_data = np.random.default_rng(CONFIG["data_seed"])
    X_normal = rng_data.normal(
        loc=[33.0, 0.03], scale=[3.0, 0.015], size=(120, 2),
    )
    X_abnormal = rng_data.normal(
        loc=[39.0, 0.07], scale=[4.0, 0.02], size=(80, 2),
    )
    X = np.concatenate([X_normal, X_abnormal], axis=0)
    y = np.concatenate([
        np.full(120, -1, dtype=np.int64),
        np.full(80, 1, dtype=np.int64),
    ])

    # 同一組編號同時取出 X 與 y，保持設備與標籤對應。
    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))
    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])
    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_indices], y[validation_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # 只向 Train 學習尺度；Validation、Test 沿用同一把尺。
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std = np.where(train_std > 0, train_std, 1.0)
    X_train_scaled = (X_train - train_mean) / train_std
    X_validation_scaled = (X_validation - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    print("\n【正式資料：切分與標準化】")
    print("Train：", X_train_scaled.shape, y_train.shape)
    print("Validation：", X_validation_scaled.shape, y_validation.shape)
    print("Test：", X_test_scaled.shape, y_test.shape)
    print("Train 平均值（溫度、振動）：", train_mean)
    print("Train 標準差（溫度、振動）：", train_std)
    print("標準化後 Train 各欄平均：", np.mean(X_train_scaled, axis=0))
    print("標準化後 Train 各欄標準差：", np.std(X_train_scaled, axis=0))

    # 驗證切分無重疊、無遺漏，以及 Train 標準化結果。
    assert len(np.unique(np.concatenate([
        train_indices, validation_indices, test_indices,
    ]))) == len(X)
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(X_train_scaled, axis=0), 1.0)
    assert np.isfinite(X_validation_scaled).all()
    assert np.isfinite(X_test_scaled).all()


    # ==================== ② 比較 C，保存每組模型與結果 ====================
    results = {}

    for C in CONFIG["C_values"]:
        # 每次呼叫都從 weights=0、bias=0 開始，只用 Train 更新參數。
        weights, bias, loss_history = fit_linear_svm(
            X_train_scaled,
            y_train,
            C=C,
            learning_rate=CONFIG["learning_rate"],
            iterations=CONFIG["iterations"],
        )
        train_scores = decision_function(X_train_scaled, weights, bias)
        validation_scores = decision_function(X_validation_scaled, weights, bias)
        train_predictions = np.where(train_scores >= 0, 1, -1)
        validation_predictions = np.where(validation_scores >= 0, 1, -1)
        train_accuracy = np.mean(train_predictions == y_train)
        validation_accuracy = np.mean(validation_predictions == y_validation)

        results[C] = {
            "weights": weights,
            "bias": bias,
            "loss_history": loss_history,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        }
        print(
            f"C={C}：Train ACC={train_accuracy:.4f}，"
            f"Validation ACC={validation_accuracy:.4f}"
        )
        assert train_predictions.shape == y_train.shape
        assert validation_predictions.shape == y_validation.shape
        assert np.isfinite(loss_history).all()

    # ==================== ③ Validation 選模 ====================
    # Accuracy 越大越好；同分時選較小的 C，讓選擇規則固定。
    selected_C = CONFIG["C_values"][0]
    for C in CONFIG["C_values"]:
        candidate_accuracy = results[C]["validation_accuracy"]
        best_accuracy = results[selected_C]["validation_accuracy"]
        if candidate_accuracy > best_accuracy:
            selected_C = C
        elif candidate_accuracy == best_accuracy and C < selected_C:
            selected_C = C

    selected_result = results[selected_C]
    selected_weights = selected_result["weights"]
    selected_bias = selected_result["bias"]
    print("選定 C：", selected_C)
    print("選定 Validation Accuracy：", selected_result["validation_accuracy"])
    # 不比較不同 C 的總目標大小：每組目標的 Hinge 權重已改變。
    # 固定步幅次梯度法不保證已收斂；這是目前訓練設定下的比較。


    # ==================== ④ 選定模型的 Test 評估 ====================
    # 不重新訓練，直接使用 Validation 選出的參數。
    # X_test_scaled 已沿用 Train 的平均值與標準差。
    test_scores = decision_function(
        X_test_scaled, selected_weights, selected_bias,
    )
    test_predictions = np.where(test_scores >= 0, 1, -1)
    test_accuracy = np.mean(test_predictions == y_test)

    print("\n【選定模型的 Test 評估】")
    print("最終 C：", selected_C)
    print("Test 答對筆數：", np.sum(test_predictions == y_test), "/", len(y_test))
    print("Test Accuracy：", test_accuracy)
    assert test_predictions.shape == y_test.shape
    assert np.isfinite(test_scores).all()
    # Test 分數用於報告，不回頭據此挑選 C 或其他訓練設定。


    # ==================== ⑤ 分類邊界與 Margin ====================
    figure_path = plot_svm_boundary(
        X_train, y_train, selected_weights, selected_bias,
        train_mean, train_std, selected_C,
    )
    print("分類邊界與 Margin 圖已儲存：", figure_path)


if __name__ == "__main__":
    main()
