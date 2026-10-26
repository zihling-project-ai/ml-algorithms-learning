import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 1. 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "k_values": (1, 5, 25),
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "grid_points": 150,
}


# ==================== 2. KNN 核心函式 ====================

# ---------- 2-1. 預測單一樣本 ----------

def predict_one(X_train, y_train, query_point, k):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k 必須是整數")

    if not 1 <= k <= len(X_train):
        raise ValueError("k 必須介於 1 與訓練樣本數之間")

    # Broadcasting：每筆訓練樣本都減去同一個查詢點
    differences = X_train - query_point
    squared_differences = differences ** 2

    # axis=1每列內的 Features 相加 ,axis=0 不同樣本的同一欄相加,sum(...)所有元素全部相加
    squared_distances = np.sum(squared_differences, axis=1)
    distances = np.sqrt(squared_distances)

    # np.argsort() 找出距離由近到遠的樣本索引
    sorted_indices = np.argsort(distances, kind="stable")
    neighbor_indices = sorted_indices[:k]
    neighbor_labels = y_train[neighbor_indices]

    votes = np.bincount(neighbor_labels, minlength=2)
    # np.bincount()按整數標籤的數值安排位置
    # votes[0]  # 標籤 0 的數量 → Normal 票數
    # votes[1]  # 標籤 1 的數量 → Abnormal 票數

    prediction = np.argmax(votes)
    # np.max(votes)：最大票數是多少。
    # np.argmax(votes)：最大票數在哪個位置，也就是哪個類別勝出。

    return prediction


# ---------- 2-2. 預測多筆樣本 ----------

def predict(X_train, y_train, X_query, k):
    predictions = []

    for query_point in X_query:
        prediction = predict_one(X_train, y_train, query_point, k)
        predictions.append(prediction)

    # 在迴圈外回傳，確保所有查詢樣本都完成預測
    return np.array(predictions, dtype=np.int64)


# ---------- 2-3. 計算準確率 ----------

def accuracy(y_true, y_pred):
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("標籤必須是一維陣列")

    if y_true.shape != y_pred.shape:
        raise ValueError("真實標籤與預測標籤的形狀必須相同")

    if y_true.size == 0:
        raise ValueError("標籤不能是空陣列")

    return np.mean(y_true == y_pred)


def main():
    # ==================== 3. 圓形資料與固定切分 ====================
    # 圓形規則只用於產生真實標籤，不參與 KNN 預測。
    rng_data = np.random.default_rng(CONFIG["data_seed"])
    X = rng_data.uniform(-2, 2, size=(200, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.44).astype(np.int64)

    # 分開管理資料生成與切分的亂數狀態，讓各 K 共用相同切分。
    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(X.shape[0])
    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    # 特徵與標籤用相同索引取出，避免配對錯誤。
    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_indices], y[validation_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"Train / Validation / Test：{len(y_train)} / {len(y_validation)} / {len(y_test)}")


    # ==================== 4. 比較 K 與 Majority-Class Baseline ====================
    # Training Accuracy：對已見過的資料，預測得如何？
    # Validation Accuracy：對未見過的資料，預測得如何？
    # K=1 的訓練查詢包含自己，所以本次 Training Accuracy 為 100%。
    results = {}
    print("\nK    Training Accuracy    Validation Accuracy")
    for k in CONFIG["k_values"]:
        train_predictions = predict(X_train, y_train, X_train, k)
        validation_predictions = predict(X_train, y_train, X_validation, k)
        train_accuracy = accuracy(y_train, train_predictions)
        validation_accuracy = accuracy(y_validation, validation_predictions)
        results[k] = {"validation_accuracy": validation_accuracy}
        print(f"{k:<4} {train_accuracy:>17.4f} {validation_accuracy:>22.4f}")

    # 多數類別只能由 Train 決定，Baseline 不使用座標。
    train_class_counts = np.bincount(y_train, minlength=2)
    majority_class = np.argmax(train_class_counts)
    baseline_validation_predictions = np.full(y_validation.shape, majority_class, dtype=np.int64)
    baseline_validation_accuracy = accuracy(y_validation, baseline_validation_predictions)
    print(f"Baseline 類別：{majority_class}；Validation Accuracy：{baseline_validation_accuracy:.4f}")

    # 依本輪 Validation 比較選 K；同分選較小的 K，不使用 Test。
    selected_k = max(results, key=lambda k: (results[k]["validation_accuracy"], -k))


    # ==================== 5. 三種 K 的決策邊界 ====================
    # 網格是繪圖查詢點，不是新增訓練樣本或 Test 評估資料。
    grid_axis = np.linspace(-2, 2, CONFIG["grid_points"])
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    # ravel 攤平兩張座標表；column_stack 合成每列一個 (x₁, x₂)。
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    for k in CONFIG["k_values"]:
        grid_predictions = predict(X_train, y_train, grid_points, k)
        # reshape 將預測依原順序排回網格，與座標表對齊。
        grid_labels = grid_predictions.reshape(grid_x.shape)
        fig, ax = plt.subplots()

        ax.contourf(
            grid_x,
            grid_y,
            grid_labels,
            levels=[-0.5, 0.5, 1.5],
            colors=["tab:blue", "tab:orange"],
            alpha=0.3,
        )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"KNN Decision Regions (K={k})")
        ax.set_aspect("equal")

        ax.scatter(
            X_train[y_train == 0, 0],
            X_train[y_train == 0, 1],
            color="tab:blue",
            edgecolors="black",
            label="Train Normal",
        )

        ax.scatter(
            X_train[y_train == 1, 0],
            X_train[y_train == 1, 1],
            color="tab:orange",
            edgecolors="black",
            label="Train Abnormal",
        )
        true_boundary = plt.Circle(
            (0, 0),
            radius=1.2,
            fill=False,
            edgecolor="black",
            linestyle="--",
            label="True boundary",
        )
        ax.add_patch(true_boundary)
        ax.legend()

        figure_path = (
            Path(__file__).resolve().parent
            / f"knn_boundary_k{k}.png"
        )

        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        print("圖檔已儲存：", figure_path)
        plt.close(fig)


    # ==================== 6. 最終 Test 評估 ====================
    # K 已固定，只用原始 Train 搜尋鄰居，不用 Test 調整模型。
    test_predictions = predict(X_train, y_train, X_test, selected_k)
    test_accuracy = accuracy(y_test, test_predictions)
    print(f"最終 K：{selected_k}；Test Accuracy：{test_accuracy:.4f}")

    # 實驗解讀見 README：K 較大不保證更準。
    # (0.6, 0.8) 真實為 Normal；K=25 時票數 8 / 17，因而誤判。


if __name__ == "__main__":
    main()
