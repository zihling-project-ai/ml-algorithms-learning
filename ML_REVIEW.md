# A1｜Machine Learning 演算法

完整推導仍保留在各 Lab 與 [數學推導筆記](MATH_DERIVATION_NOTES.md)。本文件是總複習，不代表所有公式已能獨立推導，也不代表 Phase 01 整合驗收已通過。

## 一、演算法原理總表

「監督／無監督」描述是否使用目標標籤；Ensemble、Bagging、Boosting、Probabilistic 則描述建模方式，並非互斥的同一層分類。

| 演算法 | 學習類型／方法 | 核心原理 | 主要用途 |
|---|---|---|---|
| KNN | Supervised | 找最接近新資料的 K 個訓練樣本，根據鄰居預測 | Classification／Regression |
| K-Means | Unsupervised | 反覆分配樣本並更新群中心，使群內平方距離縮小 | Clustering |
| Linear Regression | Supervised | 學習 Features 與連續目標值之間的線性關係 | Regression |
| Logistic Regression | Supervised | 將線性分數透過 Sigmoid 轉成類別 1 的預測機率 | Binary Classification |
| Decision Tree | Supervised | 利用 Feature Threshold 逐步切分資料 | Classification／Regression |
| Random Forest | Supervised；Ensemble／Bagging | 多棵隨機化 Decision Trees 整合預測 | Classification／Regression |
| AdaBoost | Supervised；Ensemble／Boosting | 本次二元分類版本逐輪提高被錯誤分類樣本的相對訓練權重 | 本次實作為 Classification |
| Gradient Boosting | Supervised；Ensemble／Boosting | 逐輪加入 Learner，沿降低 Loss 的方向修正預測 | Classification／Regression |
| Naive Bayes | Supervised；Probabilistic | 使用 Bayes Theorem，假設 Features 在給定 Class 後條件獨立 | Classification |
| SVM | Supervised | 分類時尋找兼顧 Margin 與違規懲罰的邊界 | Classification；回歸另有 SVR |
| PCA | Unsupervised | 尋找投影變異最大的正交主方向 | Dimensionality Reduction |

KNN 利用已標註的鄰近樣本進行監督式預測；K-Means 根據距離做無監督分群；PCA 尋找新的座標方向，不直接分類。AdaBoost 另有回歸等變體，本次權重更新與投票說明限定已實作的二元分類版本。

## 二、核心數學公式總表

| 演算法／項目 | 代表性公式 | 公式代表的實際運算 |
|---|---|---|
| KNN：歐氏距離 | $d(x,z)=\sqrt{\sum_j(x_j-z_j)^2}$ | 每欄相減、平方、加總，再開根號，得到兩筆資料的距離 |
| K-Means：目標 | $J=\sum_{i=1}^{n}\lVert x_i-\mu_{c_i}\rVert^2$ | 每筆與所屬群中心的各欄差平方加總，再將全部樣本加總 |
| Linear Regression：預測 | $\hat y=w^\top x+b$ | 各 Feature 乘上對應 Weight，加總後加 Bias |
| Linear Regression：MSE | $\mathrm{MSE}=\frac1n\sum_i(y_i-\hat y_i)^2$ | 每筆真實值減預測值，平方後平均 |
| Logistic Regression：預測 | $z=w^\top x+b,\quad p=\frac1{1+e^{-z}}$ | 先算線性分數，再經 Sigmoid 轉到 0～1 |
| Logistic Regression：BCE | $L=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$ | y=1 使用 −log(p)，y=0 使用 −log(1−p)，再平均；自信但錯誤的機率預測懲罰較大 |
| Decision Tree：Gini | $G=1-\sum_k p_k^2$ | 各類別在 Node 的比例平方後加總，再用 1 減掉 |
| Random Forest：分類整合 | $\hat y=\operatorname{mode}(h_1(x),\ldots,h_T(x))$ | 各樹分別預測，再選最多票的類別 |
| AdaBoost：二元分類整合 | $F(x)=\operatorname{sign}\left(\sum_t\alpha_t h_t(x)\right)$ | 每棵樹的 −1／+1 預測乘投票權重，全部加總後看正負 |
| Gradient Boosting：更新 | $F_t(x)=F_{t-1}(x)+\eta h_t(x)$ | 舊整體預測加上「學習率×本輪修正模型」 |
| Naive Bayes：類別比較 | $P(c\mid x)\propto P(c)\prod_j p(x_j\mid c)$ | 各 Feature 的條件 Likelihood 相乘，再乘類別 Prior |
| Linear SVM：本次平均 Hinge 目標 | $\min_{w,b}\left[\frac12\lVert w\rVert^2+\frac Cn\sum_i\max(0,1-y_i(w^\top x_i+b))\right]$ | 權重平方代價，加上 C×平均間隔違反懲罰，再尋找使目標較低的參數 |
| PCA：特徵方向關係 | $\Sigma v=\lambda v$ | 共變異數矩陣乘方向後，等於原方向乘某倍率；對單位特徵向量，該倍率是投影變異數 |

### 公式的版本與實作範圍

- KNN 分類以鄰居投票，回歸可用鄰居目標平均；上表距離只是鄰居選擇的一部分。
- Decision Tree 還需要比較左右子集的加權 Gini；回歸樹通常改比較平方誤差，而非 Gini。
- Random Forest 公式是分類 Hard Voting；回歸通常平均。每個 Node 隨機限制候選 Features，而非整棵樹永遠只用一欄。
- AdaBoost 公式採二元標籤 −1／+1；alpha 根據當輪加權錯誤率計算，不能把不同輪的錯誤率視為同一套樣本權重下的直接排名。零分投票需訂同分規則。
- Gradient Boosting 一般擬合 Negative Gradient。本次平方誤差回歸用殘差 y−目前預測作為修正目標；分類不能直接照抄同一個殘差規則。
- Gaussian Naive Bayes 的 Likelihood 是密度，不是某個精確數值的機率；Prior×Likelihood 是未正規化分數，若要 Posterior 機率須再除以各類別分數總和。
- SVM 原整理採 Hinge 加總版：$\frac12\lVert w\rVert^2+C_{\mathrm{sum}}\sum_i\max(0,1-y_is_i)$。本次程式採平均版，因此改以平均版列在表中。其他項相同時，$C_{\mathrm{sum}}=C_{\mathrm{mean}}/n$，不能直接比較兩版相同 C 數字。
- PCA 上式表示方向與特徵值關係；完整主線還包括中心化、排序、選取方向及投影。保留方向數量與變異門檻是任務選擇。

## 三、數學符號

| 符號 | 名稱 | 意義／實際運算 |
|---|---|---|
| $x$ | Input／Feature Vector | 一筆資料的 Features，例如 [溫度,振動] |
| $X$ | Data Matrix | 多筆資料，每列一筆，每欄一個 Feature |
| $x_i$ | Sample | 在以 i 編樣本的公式中，表示第 i 筆資料 |
| $x_j$ | Feature Value | 在單筆向量 x 的公式中，表示第 j 欄；下標用途須依上下文判讀 |
| $x_{ij}$ | Sample／Feature Value | 第 i 筆、第 j 欄的數值 |
| $y$ | Label／Target | 真實答案，可能是類別或連續目標 |
| $\hat y$ | Prediction | 模型預測答案；帽子符號表示估計／預測 |
| $w$ | Weights | 各 Feature 的係數向量，不等同樹模型的樣本權重 |
| $b$ | Bias | 線性分數的偏移參數 |
| $n$ | Sample Count | 樣本數；1/n 常表示取平均 |
| $\sum$ | Summation | 將指定範圍數值加總；不是共變異數矩陣 Σ |
| $\prod$ | Product | 將指定範圍的數值相乘 |
| $\mu$ | Mean／Center | 平均或群中心；K-Means 的 μ 可以是一整個 Feature 向量 |
| $\sigma^2$ | Variance | 相對平均的偏離平方再平均，單位是原單位的平方 |
| $\sigma$ | Standard Deviation | 變異數的平方根，與原特徵使用相同單位 |
| $\Sigma$ | Covariance Matrix | 對角線是各欄變異數，其餘格是兩欄共變異數 |
| $\lambda$ | Eigenvalue | PCA 中單位特徵方向的投影變異量 |
| $v$ | Eigenvector | PCA 的主方向；方向用於投影時固定長度為 1 |
| $\eta$ | Learning Rate | 更新／修正的幅度；也常寫作其他符號 |
| $C$ | SVM Penalty | 間隔違反懲罰的相對比重，不是直接指定 Margin 距離 |
| $P(A\mid B)$ | Conditional Probability | 已知 B 發生時，A 發生的機率；直線表示「給定」 |
| $\lVert a\rVert$ | Norm | 此處通常為向量的歐氏長度；平方範數是各分量平方加總 |
| $w^\top x$ | Transpose／Dot Product | ᵀ 是轉置；此處將對應分量相乘後加總 |
| $\propto$ | Proportional To | 成正比；尚需共同的正規化常數才能相等 |
| $\alpha_t$ | Learner Weight | 本次 AdaBoost 第 t 棵樹的投票權重 |
| $h_t(x)$ | Base Learner Output | 第 t 個模型對 x 的輸出 |
| $F_t(x)$ | Ensemble Output | 第 t 輪整體模型對 x 的輸出 |
| $\log$／$\exp$ | Natural Log／Exponential | 此處 log 是自然對數，exp(a)=e 的 a 次方，兩者互為反運算 |

同一符號在不同上下文可能有不同意義。例如 λ 在 PCA 是 Eigenvalue，在其他模型可能代表正則化強度；alpha 可能是 AdaBoost 投票權重，也可能在另一段程式代表學習率。先看定義，不只憑字母判斷。

## 四、五組最容易混淆的演算法

| 比較 | 第一種方法 | 第二種方法 | 最重要的差別 |
|---|---|---|---|
| KNN vs. K-Means | KNN 看已標註鄰居 | K-Means 看群中心 | 監督式預測 vs. 無監督分群 |
| Linear vs. Logistic Regression | 預測連續數值 | 預測二元類別機率 | 直接線性輸出 vs. Sigmoid，訓練 Loss 也不同 |
| Decision Tree vs. Random Forest | 一棵樹 | 多棵隨機化樹 | 單模型 vs. Ensemble |
| AdaBoost vs. Gradient Boosting | 調整樣本權重 | 擬合 Negative Gradient | 改變樣本重要性 vs. 累加對現有預測的修正 |
| PCA vs. SVM | 尋找資料變異方向 | 本次尋找分類邊界 | 無監督降維表示 vs. 監督式分類 |

## 五、Ensemble Learning 補充整理

```text
Ensemble Learning：整合多個模型
├─ Bagging：不同抽樣資料，各模型分別訓練，再整合
│  └─ Random Forest：再加上節點層級隨機候選 Features
└─ Boosting：模型依序建立，後續訓練依賴目前模型
   ├─ AdaBoost：調整樣本重要性，最後加權投票
   └─ Gradient Boosting：學降低 Loss 的修正方向，累加預測
      ├─ 本次 NumPy 簡化回歸版本
      ├─ XGBoost
      ├─ LightGBM
      └─ CatBoost
```

這是本課程涉及方法的整理，不是全部 Ensemble 方法的完整分類。

| 方法 | 模型如何建立 | 如何整合 | 主要特色 |
|---|---|---|---|
| Bagging | 各模型可分別訓練，通常使用 Bootstrap | Voting／Averaging | 著重降低對訓練樣本變動的敏感度 |
| Random Forest | Bootstrap＋Node-Level Random Features | Voting／Averaging | 增加多樣性，降低 Trees 相關性 |
| AdaBoost | 依序調整樣本權重，後續更重視先前答錯的樣本 | Weighted Voting | 本次為二元分類，聚焦難分樣本 |
| Gradient Boosting | 依序擬合降低當前 Loss 的修正方向 | 累加更新 | 修正依指定 Loss 而定，本次平方誤差使用殘差 |
| XGBoost | 正則化的 Gradient Tree Boosting | 加法模型 | 梯度、曲率、正則化及工程最佳化 |
| LightGBM | 高效率 Gradient Tree Boosting | 加法模型 | Histogram、常見 Leaf-wise Growth |
| CatBoost | Gradient Tree Boosting | 加法模型 | 類別特徵處理、Ordered 方法、預設常見對稱樹 |

更多 Trees 不保證每一次 Validation 分數都上升。比較穩定性需要重複訓練與固定評估輸入，不能只憑某次一筆樣本的差距宣稱過擬合或降低 Variance。

三種 Boosting 工具特色有重疊，上表不是互斥能力清單。本課程實際執行了 XGBoost 廣告回歸案例；LightGBM、CatBoost 目前以架構與選型理解為主，未實作訓練比較。

## 六、工程實務：什麼資料應考慮什麼模型？

以下沿用課程的候選方向整理，不表示僅憑任務名稱即可決定最適模型。

| 工程問題 | 候選模型 | 原因 |
|---|---|---|
| 預測設備溫度、銷售量 | Linear Regression、Gradient Boosting | 連續數值預測 |
| 判斷設備正常／異常 | Logistic Regression、Random Forest、Boosting | 監督式分類，可比較不同決策方式 |
| 尋找感測器行為群組 | K-Means | 無明確 Labels 時探索距離群組 |
| 依已知相似案例分類 | KNN | 直接利用樣本距離與已知答案 |
| 高維 Feature 壓縮 | PCA | 減少座標數並保留主要變異 |
| 小型高維分類資料 | Linear SVM、Logistic Regression | 可作具正則化的線性分類基準 |
| 使用條件獨立近似的機率分類 | Naive Bayes | 快速、簡潔的機率式基準；假設近似成立時值得比較 |

公平比較的重點是相同資料版本、切分、可取得資訊、評估指標，以及合理可比較的搜尋與資源預算。不同模型的 max_depth 數值相同，不代表模型複雜度或訓練成本相同。

## 七、整合驗收的六個核心觀念

1. **Training vs. Prediction**：模型如何從資料學習，新資料如何進入已學好的模型。KNN 主要保存訓練樣本，不能把所有 Training 都理解成梯度更新。
2. **Supervised vs. Unsupervised**：Label 在各方法中的角色。PCA、K-Means 不需目標標籤，但若納入預測流程，仍須遵守資料隔離。
3. **Loss／Objective Function**：模型在最小化或最大化什麼。SVM 不直接最佳化 Accuracy；PCA 最大化保留方向的投影變異。
4. **Overfitting／Generalization**：Training 表現好不保證未見資料同樣良好；單次 Validation 波動也不足以證明過擬合。
5. **Train／Validation／Test**：Train 學參數，Validation 支援選模／設定，Test 留到選擇完成後評估。平均、標準差、PCA 方向等前處理也只從 Train 學習。
6. **Model Selection**：根據資料、指標、業務需求、資源與工程限制選擇；不以同一數值的超參數當作公平比較的充分條件。

PCA 本次以 Train 變異保留門檻選維度，Validation 用於觀察還原誤差，不是所有實驗都必須以最低 Validation MSE 當唯一選擇規則。

## 八、各 Lab 複習入口

| 資料夾 | 主題 |
|---|---|
| [01_numpy_fundamentals](01_numpy_fundamentals/) | NumPy 基礎 |
| [02_knn_from_scratch](02_knn_from_scratch/) | KNN |
| [03_kmeans_from_scratch](03_kmeans_from_scratch/) | K-Means |
| [04_linear_regression_from_scratch](04_linear_regression_from_scratch/) | Linear Regression |
| [05_logistic_regression_from_scratch](05_logistic_regression_from_scratch/) | Logistic Regression |
| [06_decision_tree_from_scratch](06_decision_tree_from_scratch/) | Decision Tree |
| [07_random_forest_from_scratch](07_random_forest_from_scratch/) | Random Forest |
| [08_ensemble_learning](08_ensemble_learning/) | Bagging、AdaBoost、Gradient Boosting 與工具比較 |
| [09_gaussian_naive_bayes_from_scratch](09_gaussian_naive_bayes_from_scratch/) | Gaussian Naive Bayes |
| [10_svm_form_scratch](10_svm_form_scratch/) | Linear SVM，沿用現有資料夾拼字 |
| [11_pca_from_scratch](11_pca_from_scratch/) | PCA |
