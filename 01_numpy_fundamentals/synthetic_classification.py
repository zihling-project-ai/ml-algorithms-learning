import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    #===================== Step1 創造隨機（x₁,x₂)特徵 ======================#

    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, size=(200, 2))

    #print("第一次取樣：", rng.random(5))
    #print("第二次取樣：", rng.random(5))
    print("前 5 筆資料：")
    print(X[:5])
    print("X.shape：", X.shape)

    '''
    備註：
    1. (-2, 2)代表：-2 <= X < 2
    2. size=(200, 2)：200 列、2 欄；每列是一筆樣本，每欄是一個 Feature。
    3. X[:5]：取索引 0 到 4，也就是前 5 筆樣本。
    4. X.shape：查看陣列各軸的長度，預期為 (200, 2)
    '''

    #===================== Step2 模擬圓形正異常樣本邊界 ======================#

    is_abnormal = X[:, 0] ** 2 + X[:, 1] ** 2 > 1.44
    y = is_abnormal.astype(np.int64)

    print("前5筆資料：", y[:5])
    print("y.shape", y.shape)
    print("y.dtype", y.dtype)

    '''
    備註：
    假定Normal：0，x₁**2 + x₂**2 <= 1.44，在圓內或圓上。
    Abnormal：1，x₁**2 + x₂**2 > 1.44，在圓外。
    是一個以原點為圓心、半徑為 1.2 的圓
    先取出每筆資料的兩個座標：
    X[:, 0]  # 所有樣本的第一個 Feature，也就是 x₁
    X[:, 1]  # 所有樣本的第二個 Feature，也就是 x₂
    本次主要是存成整數型別，不一定只能選int64
    '''

    #===================== Step3 轉成二元分類資訊 ======================#

    normal_mask = y == 0
    X_normal = X[normal_mask]

    abnormal_mask = y == 1
    X_abnormal = X[abnormal_mask]

    print("Normal資料形狀:", X_normal.shape)
    print("Abnormal資料形狀:", X_abnormal.shape)
    print("X.ndim:", X.ndim)
    print("X.dtype:", X.dtype)
    print("y.ndim:", y.ndim)

    '''
    備註：
    Mask 是篩選條件，篩選結果的內容來自 X
    這行只拿 y 的每個元素和 0 比較，沒有再計算 X 的平方：

    X：兩個座標                    y      normal_mask
    [ 1.0958, -0.2445]             0         True
    [ 1.4344,  0.7895]             1         False
    [-1.6233,  1.9025]             1         False
    ...
    因此 normal_mask.shape 是 (200,)，每個布林值回答：「這一列要不要保留？」
    X_normal = X[normal_mask]意思是「從 X 裡，取出 Mask 為 True 的那些列」。
    每列原本有兩個座標，取出後仍保留兩個座標，所以結果是 (57, 2)
    而因為X抽樣時是帶小數的數值，而 rng.uniform() 預設用 float64
    所以X.dtype答案仍是float64
    '''

    #===================== Step4 使用圖表轉成可視化 ======================#

    fig, ax = plt.subplots()

    ax.scatter(
        X_normal[:, 0],
        X_normal[:, 1],
        label="Normal",
    )

    ax.scatter(
        X_abnormal[:, 0],
        X_abnormal[:, 1],
        label="Abnormal",
        color="tab:orange",
    )

    boundary = plt.Circle(
        (0, 0),
        radius=1.2,
        fill=False,
        edgecolor="black",
        linestyle="--",
        label="Boundary (r=1.2)",
    )
    ax.add_patch(boundary)
    ax.set_aspect("equal")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()

    figure_path = Path(__file__).resolve().parent / "synthetic_classification.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print("圖檔已儲存：", figure_path)

    plt.close(fig)


    #===================== Step5 用程式驗證資料形狀與標籤 ======================#

    assert X.shape == (200, 2), "X 的形狀不符合預期"
    assert y.shape == (200,), "y 的形狀不符合預期"
    assert np.all((y == 0)|(y == 1)), "y 出現了 0、1 以外的標籤"

    radius_squared = X[:, 0] ** 2 + X[:, 1] ** 2

    assert np.all(radius_squared[y == 0] <= 1.44), "Normal 標籤有誤"
    assert np.all(radius_squared[y == 1] > 1.44), "Abnormal 標籤有誤"

    print("資料形狀與標籤驗證通過")


    #=============== Step6 重現所有資料再度確認同樣Seed結果是否相同 ==============#

    rng_check = np.random.default_rng(42)
    X_check = rng_check.uniform(-2, 2, size=(200,2))
    y_check = (X_check[:, 0] ** 2 + X_check[:, 1] ** 2 > 1.44).astype(np.int64)

    assert np.array_equal(X, X_check), "X 無法重現"
    assert np.array_equal(y, y_check), "y 無法重現"

    print("Seed 可重現性驗證通過")

if __name__ == "__main__":
    main()
