# Linear Regression 理論驗收與問答

基礎數學統一見[完整數學推導與學習紀錄](../MATH_DERIVATION_NOTES.md)。最新補充驗收為 **Derivative / Chain Rule Foundations PASS**，限平方函數、兩段函數串接與參數更新；不代表 Sigmoid＋BCE 或矩陣梯度已獨立掌握。下列原驗收回答保留當時狀態。

Gradient 與 Chain Rule 仍需持續複習；原始碼尚未獨立審查。

## Q1 Weight 與 Bias 代表什麼

模型為 `ŷ = w*x + b`。當 w=3、b=2、x=4，預測值為 14。

- Weight：單一 Feature 模型的直線斜率；x 增加 1，預測改變 w。
- Bias：截距，x=0 時的預測值。

原回答的預測與參數意義正確。

## Q2 梯度下降為什麼減去梯度

### Regression Slope 與 Loss Gradient

兩者都涉及斜率，但改變的變數與觀察結果不同。

**Regression Slope：輸入改變，預測如何改變。**

$$
\hat y=wx+b, \qquad \frac{d\hat y}{dx}=w
$$

例如 `ŷ=2x+1`：

$$
L=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

$$
\frac{\partial L}{\partial w}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)x_i,
\qquad
\frac{\partial L}{\partial b}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)
$$

| x | Prediction |
|---|---:|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

x 每增加 1，預測增加 2；這裡的 2 是回歸直線斜率。

**Loss Gradient：參數改變，Loss 如何改變。**

固定資料與其他參數，觀察 Weight 增加一點時，MSE 的局部變化率：

$$
\frac{\partial L}{\partial w}
$$

若它等於 +4，表示目前附近增加 Weight，Loss 會局部增加。這不代表回歸直線的斜率是 +4。

| 比較 | Regression Slope | Loss Gradient |
|---|---|---|
| 數學表示 | $d\hat y/dx$ | $\partial L/\partial w$ |
| 改變的變數 | Input x | Parameter w |
| 觀察的結果 | Prediction | Loss |
| 用途 | 描述輸入與預測的關係 | 決定參數更新方向 |

### 梯度更新的基礎推導

梯度正負、減號的意義及完整數值驗證，以[完整數學推導與學習紀錄](../MATH_DERIVATION_NOTES.md)的 Part 3、Part 7 為主。前次驗收的 `w=2、梯度=+4、學習率=0.1 → 新 w=1.6` 保留作題目紀錄；MSE 專屬推導接續如下。

### Chain Rule 如何連結 Weight 與 Loss

目前需要持續補強的關係是：**Weight → Prediction → Loss**。

先看單筆資料：

$$
\hat y=wx+b, \qquad E=(\hat y-y)^2
$$

Weight 先改變預測，再透過預測改變平方誤差，因此將兩段變化率相乘：

$$
\frac{\partial E}{\partial w}
=
\frac{\partial E}{\partial\hat y}
\cdot
\frac{\partial\hat y}{\partial w}
$$

分開計算：

$$
\frac{\partial E}{\partial\hat y}=2(\hat y-y),
\qquad
\frac{\partial\hat y}{\partial w}=x
$$

合併後：

$$
\frac{\partial E}{\partial w}=2(\hat y-y)x
$$

Bias 對預測的變化率為 1，所以 Bias 梯度不用額外乘 x。

若有 n 筆資料，Loss 改為 MSE，就將各筆的梯度加總取平均：

$$
L=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

$$
\frac{\partial L}{\partial w}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)x_i,
\qquad
\frac{\partial L}{\partial b}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)
$$

對照 NumPy：

```python
errors = y_pred - y_true
grad_weight = 2 * np.mean(errors * x)
grad_bias = 2 * np.mean(errors)
```

平方導數的代數展開與極限步驟見[完整數學推導與學習紀錄](../MATH_DERIVATION_NOTES.md)的 Part 1–2。

有限差分則用 `(新 Loss − 舊 Loss) / 試探步距` 近似相同的局部變化率，可以對照解析梯度，但含步距近似誤差。

這也是後續 Backpropagation 的基礎。複習先確認每段變化率「改變什麼、觀察什麼」，再練習完整推導，不以背公式取代理解。

## Q3 Learning Rate 如何影響訓練與選擇

| Learning Rate | Training MSE | Validation MSE |
|---|---:|---:|
| 0.01 | 0.217504 | 0.227815 |
| 0.1 | 0.217244 | 0.228299 |
| 0.5 | 0.217244 | 0.228299 |

本次依 Validation 選 0.01，但它與 0.1 的驗證 MSE 只差約 0.000484，且 Validation 僅有 40 筆。這只能支持本次固定設定下的選擇，不能證明 0.01 普遍較好。

- 太小：更新慢，固定輪數內可能還未充分收斂。
- 適當：Loss 能穩定下降。
- 太大：可能震盪、不收斂或發散。

本次 0.1、0.5 都較快到達相近的低 Training MSE，沒有觀察到發散。判斷要結合 Loss 曲線與 Validation，不能只看學習率數值大小。

## 實驗與評估摘要

使用 NumPy 從零完成 Forward Pass、MSE、解析梯度與 Gradient Descent，未使用 Scikit-learn。

- 資料：200 筆，`y = 3*x + 2 + noise`；noise 平均 0、標準差 0.5。
- Train / Validation / Test：120 / 40 / 40；各候選設定共用切分，每組更新 200 次。
- 選定學習率：0.01。
- 最終模型：`ŷ = 3.059527*x + 2.070859`。
- Test MSE：0.386479；選定參數後才評估，不依 Test 回頭調參。

Test 高於 Validation，可能受有限樣本、雜訊或個別大誤差影響，不能僅憑此差距判定過擬合或資料分布改變。低 Training Loss 也不保證未見資料同樣良好。

## 五個核心觀念

1. **Forward Pass**：用 $\hat y=wx+b$ 產生預測。
2. **Loss Function**：MSE 衡量預測與真實答案的平均平方誤差。
3. **Gradient**：描述 Loss 對 Weight、Bias 的局部變化率。
4. **Gradient Descent**：往梯度反方向更新，以 Learning Rate 控制步幅。
5. **Generalization**：Training Loss 低，不代表未見資料的誤差也低。

### 數學關係速查

**預測：**

$$
\hat y=wx+b
$$

**Loss：**

$$
L=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

**參數梯度：**

$$
\frac{\partial L}{\partial w}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)x_i
$$

$$
\frac{\partial L}{\partial b}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)
$$

**使用同一組目前參數算出梯度後，再更新：**

$$
w\leftarrow w-\alpha\frac{\partial L}{\partial w}
$$

$$
b\leftarrow b-\alpha\frac{\partial L}{\partial b}
$$

## 間隔複習重點

已通過：Forward Pass、Weight / Bias、MSE 基本概念、參數更新數值計算、學習率與 Validation / Test 分工。

前次需補強的基礎項目已有後續教學紀錄，見完整數學筆記；仍須間隔回憶，並持續練習多參數偏導數與模型專屬推導。

後續 Logistic Regression 沿用 **Forward Pass → Loss → Gradient → Parameter Update** 的閱讀順序，不需反覆重做整個 Linear Regression 專案。

備註：
開啟 THEORY_QA.md。
按 ⌘ K，放開，再按 V。
右側會開啟預覽，$$ ... $$ 就會顯示成公式。
