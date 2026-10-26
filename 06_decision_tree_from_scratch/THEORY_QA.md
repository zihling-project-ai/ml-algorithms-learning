# Decision Tree｜理論驗收與問答

課程：A1-P01-L05｜Decision Tree From Scratch

首次回答的 Gini 算術與 Bootstrap 概念經修正後完成確認；原始碼獨立審查尚未進行。本紀錄保留題目、回答狀況與修正重點。

## Q1｜Gini Impurity

**題目：**某節點有 10 筆資料，Normal 6 筆、Abnormal 4 筆。Gini 是多少？全部為 Normal 時又是多少？

**回答狀況：**列式正確，但將 $0.4^2$ 算成 $0.04$，因此誤得 $0.6$；全同類時 Gini 為 0 的概念正確。

正確計算：

$$
Gini=1-\sum_k p_k^2
=1-0.6^2-0.4^2
=1-0.36-0.16
=0.48
$$

其中 $p_k$ 是節點中第 $k$ 類的比例。全部為 Normal 時：

$$
Gini=1-1^2-0^2=0
$$

二元分類的 Gini 介於 0 與 0.5：0 表示標籤全同類；兩類各半時為 0.5，混雜程度最高。純節點不必再為提高分類純度而切分。

## Q2｜Best Split Selection

**題目：**同樣 10 筆資料，下列切法哪個比較好？

| Split | 左節點 | 右節點 |
|---|---|---|
| A | 5 Normal、0 Abnormal | 1 Normal、4 Abnormal |
| B | 3 Normal、2 Abnormal | 3 Normal、2 Abnormal |

**回答狀況：**選 A 正確；原理由是左邊已純、右邊再切一層即可。後半句無法保證，仍須看剩餘 Features 與資料分布。

真正比較依據是左右節點的**加權 Gini**：

$$
Gini_{split}
=\frac{n_L}{n}Gini_L+\frac{n_R}{n}Gini_R
$$

其中 $n_L$、$n_R$ 是左右筆數，$n=n_L+n_R$。

| Split | 左 Gini | 右 Gini | 加權 Gini |
|---|---:|---:|---:|
| A | 0 | $1-0.2^2-0.8^2=0.32$ | $0.5(0)+0.5(0.32)=0.16$ |
| B | 0.48 | 0.48 | 0.48 |

A 的加權 Gini 較低，因此優先選 A，並非只看其中一邊是否已純。

## Q3｜Max Depth、Overfitting 與 Generalization

**題目：**Depth 1 的 Training／Validation Accuracy 為 69.17%／72.5%；Depth 10 為 100%／92.5%。為什麼不能只憑 Training 100% 認定最佳模型？

**回答狀況：**知道要觀察 Validation，但曾把單棵 Decision Tree 與 Random Forest 的有放回抽樣混淆，也把一次 Validation 高分直接理解為泛化能力高。

修正重點：

- 深層樹能建立更多條件，也可能記住訓練資料的細節與雜訊；Training 高分不保證未見資料同樣高分。
- 應依 Validation 比較候選設定，選定後才用 Test 作最終評估。
- 本次 Depth 10 在候選中 Validation 最高，選它合理；不能據此宣稱其他切分或真實場域也一定最佳。
- Validation 只有 40 筆，一筆即占 2.5 個百分點，結果受樣本組成影響。
- Training 100% 本身也不足以證明已過度擬合，需結合未見資料表現判斷。

### Decision Tree 與 Random Forest 的差異

| 比較 | 本次 Decision Tree | 通常的 Random Forest |
|---|---|---|
| 樹的數量 | 一棵 | 多棵 |
| 訓練資料 | 直接使用指定 Train | 每棵樹從 Train 有放回抽樣 |
| 候選 Features | 評估全部欄位 | 每個節點隨機選部分欄位 |
| 預測 | 沿規則走到葉節點 | 整合多棵樹的預測 |

Bootstrap 不等於資料洩漏；抽樣來源必須限制在 Training Set，不能混入 Test。

## Q4｜三種演算法如何處理新資料？

**題目：**新設備資料 Temperature＝42、Vibration＝0.08，KNN、K-Means、Decision Tree 分別如何推論？

**回答狀況：**三者核心流程正確；K-Means 與 Decision Tree 的用語再補精確。

| 演算法 | 推論機制 |
|---|---|
| KNN | 與訓練樣本計算距離，取最近 K 筆標籤投票，得到類別。 |
| K-Means | 與已訓練的 K 個中心計算距離，指派到最近中心所代表的群組。 |
| Decision Tree | 根據 Feature 與 Threshold 逐節點判斷，直到葉節點取得預測類別。 |

K-Means 比較的是**中心**，新資料指派時通常不必重新分群；Decision Tree 分類樹的葉節點輸出是**類別**，不是 K-Means 的群組編號。

## 最後確認｜Bootstrap 的抽樣來源

**題目：**100 筆 Training、20 筆獨立 Test。普通 Decision Tree 是否需要有放回抽樣？10 棵樹的 Random Forest 應從哪裡抽樣？

**回答與評估：**兩題正確，PASS。

1. 普通 Decision Tree 不需要先 Bootstrap，直接使用 Train 搜尋 Feature／Threshold，以加權 Gini 決定切分。
2. Random Forest 的 Bootstrap 來源是 100 筆 Training，不包含 20 筆 Test；Test 保持獨立，用於最終評估。

## 驗收狀態

| 項目 | A1 最終評估 |
|---|---|
| Gini Impurity | PASS，已修正平方計算 |
| Best Split Selection | PASS |
| Recursive Tree Prediction | PASS |
| Max Depth／Overfitting | PASS，已補強 |
| Generalization | PASS，基礎理解 |
| Decision Tree vs. Random Forest | PASS |
| NumPy From-Scratch 實作 | 依 Work 報告完成 |
| 原始碼獨立審查 | 尚未進行 |

Derivative、Chain Rule 與 Matrix Gradient 維持待複習，不增加本輪負擔。Random Forest 尚未在本輪實作。
