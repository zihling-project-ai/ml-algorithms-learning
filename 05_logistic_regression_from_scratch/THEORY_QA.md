# Logistic Regression 理論驗收與問答

基礎數學統一見[完整數學推導與學習紀錄](../MATH_DERIVATION_NOTES.md)。最新補充驗收為 **Derivative / Chain Rule Foundations PASS**，限平方函數、兩段函數串接與參數更新；不代表 Sigmoid＋BCE 或矩陣梯度已獨立掌握。下列原驗收回答保留當時狀態。

A1-P01-L04-B｜依 A1 主對話提供的驗收回饋整理。

**驗收狀態：Q1～Q3 PASS；Q4 Gradient / Chain Rule 待補強。** 已完成程式實作，不等於已掌握獨立數學推導，也不等於整章理論已通過驗收。先補強本章，不提前開始 Decision Tree。

## 驗收題目與回答紀錄

### Q1 Linear Regression 與 Logistic Regression

**題目**：為什麼 Logistic Regression 需要 Sigmoid？它與 Linear Regression 在預測與任務上有何差異？

**原始回答**：

> 因為要將數值限縮於 0–1 之間來去計算機率。
>
> 一個可以預測連續性數字，另一個主要是在分類上決策。

**回饋：PASS。** 兩者都先計算線性組合。Linear Regression 直接輸出連續數值；Logistic Regression 經 Sigmoid 得到類別 1 的機率估計，再依門檻決定分類。Sigmoid 是固定轉換，不依本批資料的最大最小值縮放。

### Q2 Sigmoid 與 Decision Threshold

**題目**：模型機率 p=0.5，門檻設為 0.6 與 0.4 時，分別預測哪一類？p=0.5 的題目條件依主對話附件回饋補全。

**原始回答**：

> ① Class 0，因為小於 0.6 都是 0。
>
> ② 大於 0.4 都算預測為正，變成 Class 1。

**回饋：PASS。** 本程式使用 `p >= threshold`：達到門檻為 1，未達為 0。因此同一個 p=0.5，在門檻 0.6 下為 0，門檻 0.4 下為 1。門檻決定標籤，不改變模型已輸出的機率；BCE 使用機率，不使用門檻後的標籤。

### Q3 Binary Cross-Entropy

**題目**：真實 y=1，Model A 預測 p=0.9，Model B 預測 p=0.1，哪個 BCE 較小？為什麼？

**原始回答**：

> 答案是 Model A，p=0.9，因為代表有 0.9 機率預測為 1，Model B 僅 p=0.1。

**回饋：PASS。** 更精確的說法是「模型給真實類別 1 的機率較高」，不是已驗證的實際正確率。此時 BCE=-ln(p)，A 約 0.105，B 約 2.303，因此 A 的 Loss 較小。

### Q4 Gradient 與 Chain Rule

**題目一**：要知道 Weight 改變時 Loss 如何變，應計算哪個 Derivative？

**原始回答**：

> L=BCE(p,y) / W

**修正**：方向上已在關注 Loss 與 Weight，但不是將 Loss 數值除以 Weight。正確記號為：

$$
\frac{\partial L}{\partial w}
$$

讀作「Loss 對 Weight 的偏導數」：固定其他獨立變數，觀察 Weight 在目前位置改變一點時，Loss 的局部變化率。分子是觀察的結果，分母是主動改變的參數；這不是一般分數的除法。

**題目二**：為什麼只計算 Sigmoid 的導數，不能直接得到 Weight 梯度？

**原始回答**：

> 是因為他的數值已經映射到 0–1 之間，而且 Sigmoid 算出來的東西有可能是 0 嗎？會梯度消失。

**修正**：Sigmoid 導數只回答「分數 z 改變，機率 p 怎麼變」，沒有包含「Weight 如何改變 z」及「p 如何改變 Loss」。需要 Chain Rule 的原因是函數相依，不是輸出範圍或梯度消失。飽和與梯度消失另見文末補充。

**題目三**：用自己的話解釋 Chain Rule 在訓練流程中的作用，不需完整推導。

**原始回答**：

> 只記得，計算 Loss 就可以知道 Weight 該往哪邊修正，一層一層往回計算後，最終可改變 y 輸出答案。

**回饋：待補強。** 已理解根據 Loss 調整參數的整體方向，但 Loss 數值本身不直接提供方向，需要計算梯度。更新的是模型參數與預測，不是修改真實答案 y。

重新理解：**Weight → Logit → Probability → Loss**。Chain Rule 把各段局部變化率相乘，得到最終 Loss 對 Weight 的影響；Gradient 是計算結果，Gradient Descent 才是使用它更新參數的動作。

## 基礎與模型專屬推導的分工

平方展開、極限、兩段 Chain Rule 為什麼相乘、梯度更新方向，以[完整數學推導與學習紀錄](../MATH_DERIVATION_NOTES.md)為主要版本。下方保留本模型的 Linear Combination、Sigmoid、BCE、三段 Chain Rule 與矩陣梯度，因為它們不是本次基礎筆記的替代內容。

## 公式符號與數字演算

以下保留附件中的公式、各段代入與數值步驟；先從單筆單一 Feature，再延伸到多筆矩陣。單筆 Loss 與整批平均 Loss 的記號依段落區分。

### 二、公式 1｜Linear Combination

$$
\boxed{z=w^Tx+b}
$$


先不要管上面的 $T$，從最簡單的一個 Feature 開始。

$$
z=wx+b
$$


| 符號 | 英文 | 意義 |
| --- | --- | --- |
| $x$ | Feature | 輸入數值 |
| $w$ | Weight | 模型學習的權重 |
| $b$ | Bias | 額外的偏移參數 |
| $z$ | Logit | 尚未轉換成機率的數值 |
| $\hat y$ | Prediction | 最終預測值或類別 |
假設有一頭牛：

$$
x=2,\quad w=1,\quad b=-1
$$


計算：

$$
z=1\times2-1=1
$$


得到：

$$
\boxed{z=1}
$$


但 1 目前只是 Logit，還不是分類機率。
如果有兩個 Features 呢？
例如：

$$
x=
\begin{bmatrix}
2\\
3
\end{bmatrix}
,\quad
w=
\begin{bmatrix}
1\\
0.5
\end{bmatrix}
$$


則：

$$
w^Tx+b=(1\times2)+(0.5\times3)-1=2.5
$$


這裡 $w^T$ 的 T 是 Transpose（轉置），讓 Weight 向量與 Feature 向量可以進行 Dot Product（內積）。
所以：

$$
w^Tx=w_1x_1+w_2x_2+\cdots+w_mx_m
$$


它不是 $w$ 的次方，也不是時間 T。

### 三、公式 2｜Sigmoid

$$
\boxed{p=\frac{1}{1+e^{-z}}}
$$


| 符號 | 意義 |
| --- | --- |
| $p$ | 預測為 Class 1 的機率估計 |
| $e$ | 自然常數，約 2.71828 |
| $z$ | 前一步計算的 Logit |
| $e^{-z}$ | 自然指數運算 |
延續剛才單一 Feature 的例子：

$$
z=1
$$


代入：

$$
p=\frac{1}{1+e^{-1}}
$$


因為：

$$
e^{-1}\approx0.3679
$$


所以：

$$
p=\frac{1}{1.3679}\approx0.7311
$$


因此：

$$
\boxed{p\approx0.7311}
$$


代表模型估計 Class 1 的機率約為 73.11%。
假設 Threshold 是 0.5：

$$
0.7311>0.5
$$


則預測：

$$
\hat y=1
$$


這裡要注意：$p$ 是連續的機率估計，$\hat y$ 是依 Threshold 得到的分類結果。 訓練 BCE 時使用的是 $p$，而不是 Threshold 後的 0 或 1。

### 四、公式 3｜Binary Cross-Entropy（BCE）
這個公式看起來比較複雜：

$$
\boxed{L=-[y\log p+(1-y)\log(1-p)]}
$$


但只要把 y 分成兩種情況，就非常容易理解。
| 符號 | 意義 |
| --- | --- |
| $L$ | Loss |
| $y$ | 真實 Label，0 或 1 |
| $p$ | 模型預測為 Class 1 的機率 |
| $\log$ | Natural Logarithm（自然對數），這裡以 $\ln$ 為準 |
| $1-p$ | 預測為 Class 0 的機率 |
| $-$ | 讓合理機率範圍內的 BCE 為非負值 |
情況 A：真實答案 y=1

$$
L=-[1\log p+(1-1)\log(1-p)]
$$


第二項變成 0：

$$
\boxed{L=-\log p}
$$


若剛才預測：

$$
p=0.7311
$$


則：

$$
L=-\ln(0.7311)\approx0.3133
$$


情況 B：真實答案 y=0
此時：

$$
L=-[0\log p+(1-0)\log(1-p)]
$$


第一項變成 0：

$$
\boxed{L=-\log(1-p)}
$$


如果模型仍預測：

$$
p=0.7311
$$


則：

$$
L=-\ln(0.2689)\approx1.3133
$$


比較：
| 真實 y | 預測 p | BCE |
| --- | --- | --- |
| 1 | 0.7311 | 0.3133 |
| 0 | 0.7311 | 1.3133 |
同一個模型輸出，因為真實 Label 不同，Loss 就會不同。
BCE 的目的，就是對給予真實類別較低機率的預測施加較大的懲罰。
回到 Q3 的例子：
當 y=1：
- p=0.9，BCE ≈ 0.105。
- p=0.1，BCE ≈ 2.303。
因此 Model A 比 Model B 好，原回答的判斷正確。

### 五、公式 4｜Gradient 到底怎麼算出來？
這是今天最重要的部分。
前面我們得到：

$$
z=wx+b
$$



$$
p=\sigma(z)
$$



$$
L=BCE(p,y)
$$


所以資料流是：
Weight $w$
模型需要學習的參數
Logit $z=wx+b$

Probability $p=\sigma(z)$

Loss $L=BCE(p,y)$


我們現在要問：
如果 Weight 增加一點點，最後的 Loss 會增加還是減少？
這需要計算：

$$
\frac{\partial L}{\partial w}
$$


先把這個符號讀作：
「Loss 對 Weight 的 Partial Derivative。」
其中：
- 分子 $L$：我們想觀察變化的結果。
- 分母 $w$：我們主動改變的參數。
- $\partial$：Partial Derivative 的符號，表示其他獨立變數固定時的局部變化率。
但 $w$ 與 $L$ 中間隔著 $z$ 和 $p$，所以需要 Chain Rule。
#### 5.1 Chain Rule：把三段影響相乘

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial p}
\cdot
\frac{\partial p}{\partial z}
\cdot
\frac{\partial z}{\partial w}
}
$$


可以逐段理解：
| Derivative | 在問什麼 |
| --- | --- |
| $\frac{\partial L}{\partial p}$ | p 改變，Loss 如何變？ |
| $\frac{\partial p}{\partial z}$ | z 改變，p 如何變？ |
| $\frac{\partial z}{\partial w}$ | w 改變，z 如何變？ |
三個影響相乘，就能得到 Weight 對 Loss 的總體局部影響。
這是 Q4 需要補強的連結。
#### 5.2 把真實數字代進 Chain Rule
沿用前面的單筆資料：

$$
x=2,\ w=1,\ b=-1,\ y=1
$$


已知：

$$
z=1,\quad p=0.7311
$$


第一段：Loss 對 p 的 Derivative。
因為 y=1：

$$
L=-\ln p
$$


所以：

$$
\frac{\partial L}{\partial p}=-\frac{1}{p}
$$


代入：

$$
-\frac{1}{0.7311}\approx-1.3679
$$


第二段：Sigmoid 的 Derivative。
Sigmoid 有一個重要結果：

$$
\frac{\partial p}{\partial z}=p(1-p)
$$


代入：

$$
0.7311(1-0.7311)\approx0.1966
$$


第三段：Logit 對 Weight 的 Derivative。
因為：

$$
z=wx+b
$$


當 x 和 b 固定時：

$$
\frac{\partial z}{\partial w}=x=2
$$


現在將三段相乘：

$$
\frac{\partial L}{\partial w}
\approx-1.3679\times0.1966\times2
$$


得到：

$$
\boxed{\frac{\partial L}{\partial w}\approx-0.5379}
$$


這個負號非常重要。
代表在目前的參數位置附近，增加 Weight 的方向會使 Loss 降低。
所以 Gradient Descent：

$$
w_{\text{new}}=w-\alpha\frac{\partial L}{\partial w}
$$


假設：

$$
\alpha=0.1
$$


則：

$$
w_{\text{new}}=1-0.1(-0.5379)
$$



$$
\boxed{w_{\text{new}}\approx1.05379}
$$


Weight 增加了。
這與我們希望 y=1 時提高 p、降低 BCE 的方向一致。

### 六、那麼為什麼最終 Gradient 公式那麼簡短？
待理解的平均梯度公式是：

$$
\frac{\partial L}{\partial w}=\frac{1}{n}X^T(p-y)
$$


先看只有一筆資料的情況。
Logistic Regression 的 Chain Rule 最後可以化簡成：

$$
\boxed{\frac{\partial L}{\partial w}=(p-y)x}
$$


剛才：

$$
p=0.7311,\quad y=1,\quad x=2
$$


直接代入：

$$
(0.7311-1)\times2=-0.5378
$$


和我們用三段 Chain Rule 算出的結果一致（差異僅來自四捨五入）。
如果有多筆資料，就把每筆的 Gradient 加總後取平均。
這才得到：

$$
\boxed{\nabla_w L=\frac{1}{n}X^T(p-y)}
$$


各符號代表：
| 符號 | 意義 |
| --- | --- |
| $n$ | Sample 數量 |
| $X$ | Feature Matrix，形狀 (n, m) |
| $X^T$ | X 的 Transpose，形狀 (m, n) |
| $p$ | 每筆樣本的預測機率，形狀 (n,) |
| $y$ | 真實 Labels，形狀 (n,) |
| $p-y$ | 每筆樣本的機率誤差 |
| $\nabla_w L$ | Loss 對所有 Weights 的 Gradient，形狀 (m,) |
例如有兩筆資料、單一 Feature：

$$
X=\begin{bmatrix}2\\1\end{bmatrix},
\quad
p=\begin{bmatrix}0.7\\0.4\end{bmatrix},
\quad
y=\begin{bmatrix}1\\0\end{bmatrix}
$$


先算：

$$
p-y=\begin{bmatrix}-0.3\\0.4\end{bmatrix}
$$


再算：

$$
X^T(p-y)=2(-0.3)+1(0.4)=-0.2
$$


最後平均：

$$
\nabla_w L=\frac{-0.2}{2}=\boxed{-0.1}
$$


這就是 NumPy 可以用 Matrix Multiplication 一次計算整批 Gradient 的原因。
不需要對每筆樣本寫出三段 Chain Rule，因為數學已經推導出可向量化的公式。

### 七、關於你提到的 Vanishing Gradient
Q4 原回答提到：
Sigmoid 算出來的東西有可能是 0 嗎？會 Gradient Vanishing？

你聯想到的現象確實存在，但要稍微區分。
對有限的 $z$，Sigmoid 的輸出嚴格介於 0 與 1 之間，而不是精確等於 0 或 1。
當 $z$ 非常大或非常小時：

$$
\sigma'(z)=p(1-p)
$$


會接近 0，這就是 Sigmoid Saturation（飽和）。
在多層神經網路中，若許多很小的 Derivatives 持續相乘，可能造成 Vanishing Gradient。
但這不是為什麼 Logistic Regression 需要 Chain Rule。
Chain Rule 是因為函數之間有依賴關係。
而且在 Logistic Regression 搭配 BCE 時，對 Logit 的 Gradient 可以化簡成 $p-y$，不能單純認為 Sigmoid 的 Derivative 小就一定無法學習。

## 複習狀態與實作對照

| 項目 | 狀態 |
|---|---|
| Linear vs. Logistic | PASS |
| Sigmoid 與門檻判斷 | PASS |
| BCE 機率意義 | PASS |
| Gradient Descent 數字更新 | 已有基礎 |
| Loss / Weight 與偏導數的區別 | 本次修正，待複習 |
| Chain Rule 三段連結與獨立推導 | 待補強 |
| Transpose 與 Matrix Gradient | 待複習 |

本次程式使用單一 Feature，對應矩陣平均梯度的 NumPy 寫法為：

```python
errors = p - y_true
grad_weight = np.mean(errors * x)
grad_bias = np.mean(errors)
```

多 Feature 的形式是 `X.T @ (p-y) / n`，本章只作公式延伸，未完成多 Feature 實驗。解析推導適用未正則化、未裁切的 BCE；程式 clipping 是避免 log(0) 的數值保護，裁切區的嚴格導數不能直接等同於未裁切公式。

實驗與圖表見 README；本筆記保存正式題目、原回答、回饋與數學重新理解。下一次從單筆 x=2、w=1、b=-1、y=1 重走 z → p → Loss → 三段變化率，再看矩陣，不以背誦最終公式取代理解。
