"""PCA 輔助學習：保留原有展開計算與五筆資料驗證。"""
import numpy as np

from pca import fit_pca, transform_pca, inverse_transform_pca


def run_learning_examples():
    positions_a = np.array([-2, -1, 0, 1, 2]) * np.sqrt(2)
    positions_b = np.zeros(5)

    variance_a = np.var(positions_a)
    variance_b = np.var(positions_b)

    print("方向 A 的變異數：", variance_a)
    print("方向 B 的變異數：", variance_b)

    X = np.array([
        [-2.0, -2.0],
        [-1.0, -1.0],
        [ 0.0,  0.0],
        [ 1.0,  1.0],
        [ 2.0,  2.0],
    ])

    means = np.mean(X, axis=0) # [0. 0.]
    centered = X - means
    # [[-2. -2.]
    #  [-1. -1.]
    #  [ 0.  0.]
    #  [ 1.  1.]
    #  [ 2.  2.]]
    temperature = centered[:, 0] # [-2. -1. 0. 1. 2.]
    vibration = centered[:, 1] # [-2. -1. 0. 1. 2.]

    temperature_variance = np.mean(temperature * temperature)
    vibration_variance = np.mean(vibration * vibration)
    covariance = np.mean(temperature * vibration)

    covariance_matrix = np.array([
        [temperature_variance, covariance],
        [covariance, vibration_variance],
    ])

    print(covariance_matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    print("特徵值：")
    print(eigenvalues)

    print("特徵向量：")
    print(eigenvectors)


    best_index = np.argmax(eigenvalues)
    principal_direction = eigenvectors[:, best_index]

    print("第一主成分方向：", principal_direction)

    # 每欄分別計算平均，再將中心移到原點。
    means = np.mean(X, axis=0)
    centered = X - means

    # 每筆資料沿第一主成分方向的位置。
    projected = centered @ principal_direction

    print("第一主成分方向：", principal_direction)
    print("投影後的位置：", projected)
    print("投影後的形狀：", projected.shape)
    print("投影後的變異數：", np.var(projected))


    # ① 每列一筆，兩欄為已處理尺度的 Feature。
    X = np.array([
        [-1.5, -2.5],
        [-1.5, -0.5],
        [ 0.0,  0.0],
        [ 0.5,  1.5],
        [ 2.5,  1.5],
    ])

    # ② 中心化：每欄減掉自己的平均。
    means = np.mean(X, axis=0)
    centered = X - means

    # ③ 建立共變異數矩陣。
    # 每格都是「兩欄偏離量相乘，再取平均」。
    covariance_matrix = centered.T @ centered / len(X)

    # ④ 找方向及各方向的變異數。
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # ⑤ 按變異數由大到小排列，方向也必須一起換順序。
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # ⑥ 保留第一個主方向，將二維資料投影成一維。
    components = eigenvectors[:, :1]
    projected = centered @ components

    # ⑦ 計算保留的變異比例。
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    print("共變異數矩陣：\n", covariance_matrix)
    print("排序後的特徵值：", eigenvalues)
    print("保留的主方向：\n", components)
    print("投影結果：\n", projected)
    print("降維前後形狀：", X.shape, "→", projected.shape)
    print("保留變異比例：", explained_variance_ratio[0])

    # ==================== 訓練函式：以第二組五筆資料確認 ====================
    model = fit_pca(X, n_components=1)
    print("\nfit_pca 學到的平均：", model["mean"])
    print("fit_pca 保留的方向：\n", model["components"])
    print("fit_pca 保留的變異數：", model["explained_variance"])
    print("fit_pca 保留的比例：", model["explained_variance_ratio"])

    assert model["components"].shape == (2, 1)
    assert np.allclose(model["explained_variance"], [4.0])
    assert np.allclose(model["explained_variance_ratio"], [4 / 4.4])


    # ==================== 轉換函式：沿用同一套投影規則 ====================
    projected_by_function = transform_pca(X, model)
    print("\ntransform_pca 投影結果：\n", projected_by_function)
    print("transform_pca 輸出形狀：", projected_by_function.shape)
    assert projected_by_function.shape == (5, 1)
    assert np.allclose(np.var(projected_by_function, axis=0), model["explained_variance"])

    X_new = np.array([[0.5, 1.5], [2.0, 2.0]])
    new_projected = transform_pca(X_new, model)
    print("新資料投影位置：\n", new_projected)
    assert new_projected.shape == (2, 1)
    # 方向反號仍合法，因此比較絕對值。
    assert np.allclose(np.abs(new_projected[:, 0]), [np.sqrt(2), 2 * np.sqrt(2)])


    # ==================== 近似還原：觀察丟掉的細節 ====================
    reconstructed = inverse_transform_pca(projected_by_function, model)
    errors = X - reconstructed
    reconstruction_mse = np.mean(errors ** 2)

    print("\n原始資料：\n", X)
    print("近似還原：\n", reconstructed)
    print("原始資料減還原資料：\n", errors)
    print("還原 MSE（所有樣本、所有欄位平均）：", reconstruction_mse)

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, [
        [-2.0, -2.0], [-1.0, -1.0], [0.0, 0.0],
        [1.0, 1.0], [2.0, 2.0],
    ])
    assert np.isclose(reconstruction_mse, 0.2)


if __name__ == "__main__":
    run_learning_examples()
