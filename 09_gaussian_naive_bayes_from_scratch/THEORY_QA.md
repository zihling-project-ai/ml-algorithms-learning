# L07｜Gaussian Naive Bayes 理論驗收與問答

已能描述 Training 到 Prediction 的資料流；Gaussian Density 細部公式標記為「認得用途、尚未熟練推導」，留待 A3 Probability／Statistics 或後續複習。

## 驗收題｜模型如何判斷新資料的類別？

訓練後已知兩類的 Prior，以及各 Feature 在各 Class 下的 Mean、Variance。如何利用這些資訊，判斷新資料比較屬於 Normal 還是 Abnormal？


## 原回答

> 看新資料在不同標籤下的密度。Prior 是正常與不正常在群體裡的比例；各類每個 Feature 有 Mean、Variance，藉由公式計算新資料的密度。最後用 Prior × Likelihood 得到各類機率，例如 Normal 5%、Abnormal 95%，就預測 Abnormal。

**A1 評估：PASS。** 已掌握 Prior、分布參數、Likelihood 與 Posterior 的核心連接；有兩個用詞與計算步驟需要修正。

## 修正一｜Gaussian Density 不等於距離

模型不是計算新資料與 Label 的距離。Label 是類別編號，不是距離計算的座標。

更精確的說法：利用各類別的 Mean、Variance，計算新 Feature Value 在該類分布下的**機率密度**，作為 Likelihood。

- Mean：該類某 Feature 的分布中心。
- Variance：該類某 Feature 的分散程度。
- Density：新數值在這個分布中的密度高度，不是直接的類別機率。

公式雖然包含與平均值的平方差，但整個 Gaussian Density 不能直接稱為距離。

## 修正二｜Prior × Likelihood 先得到分數

對多個 Features，模型先計算：

$$
\operatorname{Score}(c)=P(c)\prod_j p(x_j\mid c)
$$

| 符號 | 意義 |
|---|---|
| $c$ | 正在評估的類別，例如 Normal |
| $P(c)$ | 該類的 Prior，這次由 Train 類別比例估計 |
| $x_j$ | 新資料的第 $j$ 個 Feature 值 |
| $p(x_j\mid c)$ | 假設屬於類別 $c$，該數值在此類分布下的密度 |
| $\prod_j$ | 將同一類別下，各 Feature 的密度相乘 |

能將各 Feature 密度相乘，是使用了**給定類別後，各 Features 條件獨立**的 Naive 假設。不是說 Features 在所有資料中都一定無關。

這個 Score 是相對分數，不一定介於 0 與 1，也不一定加總為 1。

- **只要預測類別：**直接比較各類分數，選最大者。
- **要輸出 Posterior：**每類分數除以全部類別分數總和。

$$
P(c\mid x)=\frac{\operatorname{Score}(c)}{\sum_{c'}\operatorname{Score}(c')}
$$

共同分母不改變大小順序，因此兩種方式會選出相同類別。

例如以下只是正規化示意，不是本次實測：

```text
Normal 分數：0.01
Abnormal 分數：0.19
總分：0.20

Normal Posterior：0.01／0.20＝5%
Abnormal Posterior：0.19／0.20＝95%
→ 預測 Abnormal
```

## 修正後的完整資料流

```text
Training：只使用 Train
    → 計算各類 Prior
    → 計算每類、每個 Feature 的 Mean／Variance
    → 保存模型統計量

Prediction：新資料＋已訓練統計量
    → 計算各類各 Feature 的 Gaussian Likelihood
    → 同類的 Likelihood 相乘，再乘 Prior
    → 各類相對分數
        ├─ 直接比較 → 預測類別
        └─ 正規化 → Posterior → 選機率最大類別
```

預測不需要新資料的真實標籤，也不重新估計訓練統計量。程式使用 Log 等價計算與穩定正規化，以避免密度下溢；這不改變上述模型主線。

## 本次掌握與後續安排

| 項目 | 狀態 |
|---|---|
| Prior：Train 類別比例 | 已理解 |
| Mean／Variance：類別內逐 Feature 的分布參數 | 已理解 |
| Likelihood：新數值在各類分布下的密度 | 已理解，已修正距離用詞 |
| Score 與 Posterior 的差別 | 已補充正規化步驟 |
| Training → Prediction 核心流程 | A1 理論驗收 PASS |
| Gaussian Density 完整公式推導 | 認得用途，尚未熟練推導 |
| Log-Likelihood 完整推導 | 保留日後複習，不阻擋主線 |

後驗機率依賴模型假設與訓練統計，接近 1 不代表現實中絕對確定。詳細程式、實驗結果與限制另見 [README](README.md)。
