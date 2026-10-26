import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 1. 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "init_seed": 42,        # 重現初始群中心
    "n_samples": 200,
    "k_values": (2, 3, 4),
    "max_iterations": 100, # 限制最多更新次數
    "tolerance": 1e-4,      # 中心最大移動量不超過 0.0001 就停止
}


# ==================== 2. 初始化群中心 ====================

def initialize_centroids(X, k, seed):
    if not 1 <= k <= len(X):
        raise ValueError("k 必須介於 1 與樣本數之間")

    rng = np.random.default_rng(seed)

    # ① 抽出 K 個不重複的樣本編號
    selected_indices = rng.choice(len(X), size=k, replace=False)

    # ② 取出座標，當作初始中心
    centroids = X[selected_indices].copy()

    # ③ 把初始中心交回函式外
    return centroids


# ==================== 3. 分配到最近中心 ====================

def assign_cluster(X, centroids):
    cluster_ids = np.empty(len(X), dtype=np.int64)

    for i, point in enumerate(X):
        # 每個中心減去同一個樣本，得到各方向的座標差。
        differences = centroids - point

        # axis=1：每個中心的兩個平方差相加。
        squared_distances = np.sum(differences ** 2, axis=1)

        # 保存最近中心的編號，不是距離。
        cluster_ids[i] = np.argmin(squared_distances)
    return cluster_ids


# ==================== 4. 更新群中心 ====================

def update_centroids(X, cluster_ids, centroids):
    new_centroids = centroids.copy()

    for j in range(len(centroids)):
        # 取出目前分到第 j 群的全部樣本。
        members = X[cluster_ids == j]

        if len(members) > 0:
            # axis=0：各 Feature 分別跨樣本取平均。
            new_centroids[j] = np.mean(members, axis=0)

        # 空群沒有平均位置，本版本保留該群的舊中心。

    return new_centroids


# ==================== 5. 反覆分配與更新，直到停止 ====================

def fit_kmeans(X, k, seed, max_iterations, tolerance):
    # 初始化只做一次，不能放進迴圈內。
    centroids = initialize_centroids(X, k, seed)
    if max_iterations < 1 or tolerance < 0:
        raise ValueError("最大更新次數須至少為 1，停止門檻不能小於 0")

    movement_history = []

    for _ in range(max_iterations):
        # A. 根據目前中心，決定每筆資料屬於哪一群。
        cluster_ids = assign_cluster(X, centroids)

        # B. 根據這次分群，計算各群的平均位置。
        new_centroids = update_centroids(X, cluster_ids, centroids)

        # C. 比較新舊中心，得到每個中心的移動距離。
        center_movements = np.sqrt(
            np.sum((new_centroids - centroids) ** 2, axis=1)
        )

        # 用移動最多的中心代表這一輪，並保存紀錄。
        max_movement = np.max(center_movements)
        movement_history.append(max_movement)

        # D. 接受新中心，下一輪才會從新位置開始。
        centroids = new_centroids

        # E. 所有中心都移動得足夠小，就停止。
        if max_movement <= tolerance:
            break

    # 確保回傳的群組編號對應最後的中心。
    cluster_ids = assign_cluster(X, centroids)

    return centroids, cluster_ids, np.array(movement_history)


# ==================== 6. 計算 Inertia ====================

def compute_inertia(X, cluster_ids, centroids):
    # 每筆樣本取出自己的中心，形成與 X 相同形狀的陣列。
    assigned_centroids = centroids[cluster_ids]
    differences = X - assigned_centroids

    # 加總全部樣本、全部 Features 的平方差。
    # 不指定 axis，因為最後需要一個總數；也不開根號。
    return np.sum(differences ** 2)


def main():
    # ==================== 7. 同一份資料比較 K＝2、3、4 ====================
    # def 定義工具；這一區才實際呼叫工具、接收結果。
    # 等號左邊接住回傳值，括號內才是傳入參數。
    rng_data = np.random.default_rng(CONFIG["data_seed"])
    X = rng_data.uniform(-2, 2, size=(CONFIG["n_samples"], 2))

    # 不建立或使用真實標籤 y，各群編號沒有 Normal / Abnormal 意義。
    results = {}
    for k in CONFIG["k_values"]:
        centroids, cluster_ids, movement_history = fit_kmeans(
            X,
            k=k,
            seed=CONFIG["init_seed"],
            max_iterations=CONFIG["max_iterations"],
            tolerance=CONFIG["tolerance"],
        )

        # Inertia 需要樣本位置、每筆所屬群組，以及各群中心位置。
        inertia = compute_inertia(X, cluster_ids, centroids)
        cluster_counts = np.bincount(cluster_ids, minlength=k)
        converged = movement_history[-1] <= CONFIG["tolerance"]

        # 保存每個 K 的結果，畫圖時不必再執行分群。
        results[k] = {
            "centroids": centroids,
            "cluster_ids": cluster_ids,
            "movement_history": movement_history,
            "inertia": float(inertia),
            "cluster_counts": cluster_counts,
            "converged": bool(converged),
        }

        print(f"\nK＝{k}")
        print("更新輪數：", len(movement_history))
        print("最後最大移動量：", movement_history[-1])
        print("停止原因：", "中心移動量已達門檻" if converged else "已達最大更新次數")
        print("各群樣本數：", cluster_counts)
        print("Inertia：", inertia)


    # ==================== 8. 分群結果比較圖 ====================
    output_dir = Path(__file__).resolve().parent
    fig_clusters, axes = plt.subplots(1, len(CONFIG["k_values"]), figsize=(15, 4), squeeze=False)

    for ax, k in zip(axes[0], CONFIG["k_values"]):
        current = results[k]
        centers = current["centroids"]
        groups = current["cluster_ids"]

        ax.scatter(X[:, 0], X[:, 1], c=groups, cmap="tab10", s=25)
        ax.scatter(
            centers[:, 0], centers[:, 1],
            color="black", marker="X", s=180, label="Centroids",
        )
        ax.set_title(f"K={k}, Inertia={current['inertia']:.2f}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_aspect("equal")
        ax.legend()

    fig_clusters.tight_layout()
    fig_clusters.savefig(output_dir / "kmeans_clusters_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_clusters)


    # ==================== 9. 中心移動紀錄圖 ====================
    # max_movement 是單輪的停止依據；history 保存每輪變化。
    # 這不是 Inertia 曲線，不要求每輪單調下降。
    fig_movement, ax = plt.subplots()
    for k in CONFIG["k_values"]:
        movements = results[k]["movement_history"]
        rounds = np.arange(1, len(movements) + 1)
        ax.plot(rounds, movements, marker="o", label=f"K={k}")

    ax.axhline(CONFIG["tolerance"], color="black", linestyle="--", label="Tolerance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Maximum centroid movement")
    ax.set_title("K-Means Convergence")
    ax.legend()
    fig_movement.tight_layout()
    fig_movement.savefig(output_dir / "kmeans_centroid_movement.png", dpi=150, bbox_inches="tight")
    plt.close(fig_movement)
    print("\n兩張圖已儲存至：", output_dir)


if __name__ == "__main__":
    main()
