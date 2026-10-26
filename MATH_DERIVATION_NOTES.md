# Derivative、Chain Rule 與 Gradient Descent 數學推導筆記

## 閱讀位置與文件分工

這是 Linear / Logistic Regression 共用的基礎數學主筆記，完整保留本次代數推導、數字驗證、原始回答與修正。重複基礎說明以本檔為準，模型專屬推導仍放在各 Lab 問答中。

- [Linear Regression 驗收問答與 MSE 梯度](04_linear_regression_from_scratch/THEORY_QA.md)
- [Logistic Regression 驗收問答與 Sigmoid／BCE 梯度](05_logistic_regression_from_scratch/THEORY_QA.md)
- 各 Lab README：實作流程、實驗數值、圖表與執行方法。

本輪 PASS 僅指基礎教學驗收；保留「完整 Logistic BCE 推導待複習」與間隔回憶的限制，不提升為整個 Logistic Regression 理論 PASS。


**課程：A1-P01-L04｜Linear / Logistic Regression From Scratch**

**學習目的：** 從基礎代數出發，理解 Derivative 如何表示變化率、Chain Rule 為什麼需要相乘，以及 Gradient Descent 如何利用 Gradient 更新 Weight。

**本輪驗收：Derivative / Chain Rule Foundations — PASS**

Logistic Regression 的完整 BCE Gradient 推導仍待後續複習。

---

### 一、最初遇到的問題

學習 Logistic Regression 時，遇到三個不理解的公式。

#### 1. Chain Rule

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial p}
\frac{\partial p}{\partial z}
\frac{\partial z}{\partial w}
$$

#### 2. Gradient

$$
\nabla_wL=\frac{1}{n}X^T(p-y)
$$

#### 3. Gradient Descent

$$
w_{\text{new}}=w-\alpha\nabla_wL
$$

當時主要不理解：

- Derivative 的數學符號代表什麼。
- 為什麼 Chain Rule 需要將變化率相乘。
- Gradient 的正負值代表什麼。
- 為什麼 Gradient Descent 使用減法。
- 為什麼可以假設 $L=w^2$。
- 為什麼不同函數要使用不同的展開方式。

因此暫時不處理 Sigmoid、BCE 與 Matrix Gradient，先使用簡單函數建立數學基礎。

---

## Part 1｜從代數理解 Derivative

### 二、為什麼先假設 Loss = Weight²？

為了理解 Derivative，先建立一個簡化的教學函數：

$$
\boxed{L=w^2}
$$

其中：

- $L$：Loss。
- $w$：Weight。

這不是說所有 ML 模型的 Loss 都等於 Weight 的平方。

真正的 Linear Regression 可能使用：

$$
L=(\hat y-y)^2
$$

其中：

$$
\hat y=wx+b
$$

因此：

$$
L=(wx+b-y)^2
$$

這個公式比較複雜。

所以先使用 $L=w^2$，觀察 Weight 改變時，Loss 如何變化。

| Weight | Loss |
|---|---:|
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |

可以看到 Weight 每次增加 1，但 Loss 增加的幅度不同。

這表示平方函數的變化率不是固定的。

---

### 三、完全平方公式如何推導？

假設原本 Weight 是 $w$。

現在讓 Weight 增加一個很小的數值 $h$。

新的 Weight：

$$
w_{\text{new}}=w+h
$$

新的 Loss：

$$
L_{\text{new}}=(w+h)^2
$$

平方代表自己乘自己：

$$
(w+h)^2=(w+h)(w+h)
$$

使用 Distributive Property（乘法分配律）：

$$
=w\times w+w\times h+h\times w+h\times h
$$

整理：

$$
=w^2+wh+hw+h^2
$$

因為：

$$
wh=hw
$$

所以：

$$
wh+hw=2wh
$$

最後得到：

$$
\boxed{(w+h)^2=w^2+2wh+h^2}
$$

這就是 Perfect Square Identity（完全平方公式）。

---

### 四、為什麼 Loss 的變化量是 2wh + h²？

首先，$\Delta$（Delta）表示變化量。

因此：

$$
\Delta L=L_{\text{new}}-L_{\text{old}}
$$

原本：

$$
L_{\text{old}}=w^2
$$

新的：

$$
L_{\text{new}}=(w+h)^2
$$

所以：

$$
\Delta L=(w+h)^2-w^2
$$

將完全平方公式代入：

$$
\Delta L=w^2+2wh+h^2-w^2
$$

因為：

$$
w^2-w^2=0
$$

因此：

$$
\boxed{\Delta L=2wh+h^2}
$$

**這個公式不是另外背誦的規則，而是由 New Loss 減去 Old Loss，再透過代數化簡得到。**

#### 原始回答

$$
(w+h)^2=w^2+2wh+h^2
$$

$$
(w+h)^2-w^2
=w^2+2wh+h^2-w^2
=2wh+h^2
$$

**驗收：PASS**

---

### 五、使用實際數字驗證變化量

假設：

$$
w=2,\quad h=0.1
$$

#### 方法 A：直接計算

Old Loss：

$$
L_{\text{old}}=2^2=4
$$

New Loss：

$$
L_{\text{new}}=(2+0.1)^2=4.41
$$

因此：

$$
\Delta L=4.41-4=0.41
$$

#### 方法 B：使用展開公式

$$
\Delta L=2wh+h^2
$$

代入：

$$
=2(2)(0.1)+(0.1)^2
$$

$$
=0.4+0.01
$$

$$
\boxed{\Delta L=0.41}
$$

兩種方法得到相同結果，確認代數推導正確。

---

## Part 2｜從變化量推導 Derivative

### 六、什麼是 Average Rate of Change？

已知：

$$
\Delta L=2wh+h^2
$$

而 Weight 的變化量為：

$$
\Delta w=h
$$

因此 Average Rate of Change 為：

$$
\frac{\Delta L}{\Delta w}
=\frac{2wh+h^2}{h}
$$

使用代數化簡：

$$
=\frac{2wh}{h}+\frac{h^2}{h}
$$

當 $h\ne0$ 時：

$$
\boxed{\frac{\Delta L}{h}=2w+h}
$$

這表示在 Weight 增加 $h$ 的過程中，Loss 的平均變化率。

---

### 七、Derivative 是如何得到的？

Derivative（導數）描述的是某個位置的 Instantaneous Rate of Change（瞬間變化率）。

讓 $h$ 越來越接近 0：

$$
\frac{\Delta L}{h}=2w+h
$$

當：

$$
h\rightarrow0
$$

得到：

$$
\boxed{\frac{dL}{dw}=2w}
$$

更完整的數學表示為：

$$
\frac{dL}{dw}
=
\lim_{h\to0}
\frac{L(w+h)-L(w)}{h}
$$

注意：不是直接將分母設成 0，而是觀察 $h$ 逐漸接近 0 時，整體比值趨近哪個數值。

#### 實際練習

假設：

$$
w=3,\quad h=0.1
$$

則：

$$
\frac{\Delta L}{h}=2(3)+0.1=6.1
$$

我的回答：

> 2 × 3 + 0.1 = 6.1，若越來越接近 0，則約等於 6。

正確。

當 $h$ 趨近 0：

$$
\boxed{\frac{dL}{dw}=6}
$$

**驗收：PASS**

#### 目前理解

Derivative 不是單純計算兩個數值的差，而是在描述：

當輸入變數發生極小的變化時，輸出會以什麼速率改變。

---

## Part 3｜Gradient 與 Gradient Descent

### 八、Gradient 是什麼？

對單一 Weight 的函數：

$$
L=w^2
$$

Derivative：

$$
\frac{dL}{dw}=2w
$$

當：

$$
w=3
$$

則：

$$
\boxed{\frac{dL}{dw}=6}
$$

在只有一個參數的情況下，這個 Derivative 就是 Gradient。

它告訴我們目前 Loss 對 Weight 的局部變化率。

| Gradient | 意義 |
|---|---|
| Positive | 增加 Weight 時，Loss 局部上升 |
| Negative | 增加 Weight 時，Loss 局部下降 |
| Zero | 目前局部變化率為零 |

若模型有多個 Weights，Gradient 就是一個包含各參數 Partial Derivatives 的 Vector。

---

### 九、為什麼 Gradient Descent 要使用減法？

Gradient Descent 的目標是降低 Loss。

更新公式：

$$
\boxed{w_{\text{new}}=w-\alpha\times Gradient}
$$

其中：

- $w$：目前 Weight。
- $\alpha$：Learning Rate。
- $Gradient$：Loss 對 Weight 的局部變化率。

Gradient 表示 Loss 局部增加最快的方向。

因此 Gradient Descent 使用負號，沿相反方向更新參數。

這不代表 Weight 一定會變小。

#### Gradient 為正

例如：

$$
Gradient=+6
$$

則 Weight 會減少。

#### Gradient 為負

例如：

$$
Gradient=-6
$$

因為：

$$
w_{\text{new}}=w-\alpha(-6)
$$

Weight 反而增加。

**所以減號是在反轉 Gradient 的方向，而不是規定 Weight 必須下降。**

---

### 十、第一次 Gradient Descent 計算

假設：

$$
w=3
$$

$$
Gradient=6
$$

$$
\alpha=0.1
$$

更新：

$$
w_{\text{new}}=3-0.1(6)
$$

$$
=3-0.6
$$

$$
\boxed{w_{\text{new}}=2.4}
$$

#### 第一次的原始回答

> 3 − 0.6 = 0.4，減少。

當時判斷 Weight 應該減少是正確的，但減法計算錯誤。

正確結果是：

$$
3-0.6=2.4
$$

#### 驗證 Loss

原本：

$$
L_{\text{old}}=3^2=9
$$

更新後：

$$
L_{\text{new}}=2.4^2=5.76
$$

結果：

$$
\boxed{9\rightarrow5.76}
$$

這次 Gradient Descent 成功讓 Loss 下降。

但 Gradient Descent 並不保證每次更新都能降低 Loss。如果 Learning Rate 太大，可能造成 Overshooting、Oscillation 或 Divergence。

---

## Part 4｜從 Linear Function 進入 Chain Rule

### 十一、加入中間變數 z

原本直接使用：

$$
L=w^2
$$

現在改成兩段函數：

$$
z=2w
$$

$$
L=z^2
$$

也就是：

**Weight → z → Loss**

這代表 Weight 先影響 z，再由 z 影響 Loss。

#### 第一次練習時的錯誤

題目：

$$
z=2w
$$

$$
w=3
$$

Weight 增加：

$$
h=0.01
$$

我原本寫：

> z = 2(3 + 0.01)²，然後使用 $2wh+h^2$ 計算變化。

錯誤原因：

把前面 $L=w^2$ 的 Squared Function，誤用到新的 Linear Function：

$$
z=2w
$$

兩種函數的計算方式不同。

#### 正確推導

原本：

$$
z_{\text{old}}=2(3)=6
$$

新的 Weight：

$$
w_{\text{new}}=3.01
$$

新的 z：

$$
z_{\text{new}}=2(3.01)=6.02
$$

所以：

$$
\Delta z=6.02-6
$$

$$
\boxed{\Delta z=0.02}
$$

#### 修正回答

> 新 z − 舊 z = 2(3 + 0.01) − 2 × 3 = 2 × 0.01 = 0.02。

**驗收：PASS**

---

### 十二、第二次 Linear Function 練習

題目：

$$
z=3w
$$

目前：

$$
w=2
$$

Weight 增加：

$$
h=0.1
$$

原本：

$$
z_{\text{old}}=3(2)=6
$$

更新後：

$$
z_{\text{new}}=3(2.1)=6.3
$$

所以：

$$
\Delta z=6.3-6=0.3
$$

#### 原始回答

> 新 z − 舊 z = 3(2 + 0.1) − 3 × 2 = 3 × 0.1 = 0.3。

**驗收：PASS**

接著計算變化率：

$$
\frac{\Delta z}{\Delta w}=\frac{0.3}{0.1}=3
$$

因為：

$$
z=3w
$$

是 Linear Function，其變化率固定為 3。

所以：

$$
\boxed{\frac{dz}{dw}=3}
$$

這是 Chain Rule 的第一段。

---

## Part 5｜計算 Chain Rule 的第二段

### 十三、z 如何影響 Loss？

現在假設：

$$
L=z^2
$$

目前：

$$
z=6
$$

因此：

$$
L_{\text{old}}=6^2=36
$$

如果 z 增加：

$$
\Delta z=0.1
$$

則：

$$
z_{\text{new}}=6.1
$$

New Loss：

$$
L_{\text{new}}=6.1^2=37.21
$$

Loss 的變化量：

$$
\Delta L=37.21-36
$$

$$
\boxed{\Delta L=1.21}
$$

#### 原始回答

> ① 新的 Loss = (6 + 0.1)²
>
> ② Loss 比原本增加多少 = (6 + 0.1)² − 36。

列式正確。

後續計算確認為 37.21 與 1.21。

**驗收：PASS**

---

### 十四、從第二段求出 Derivative

已知：

$$
\Delta z=0.1
$$

$$
\Delta L=1.21
$$

Average Rate of Change：

$$
\frac{\Delta L}{\Delta z}
=\frac{1.21}{0.1}
$$

$$
=12.1
$$

如果讓 $\Delta z$ 越來越小：

| $\Delta z$ | Average Rate of Change |
|---|---:|
| 0.1 | 12.1 |
| 0.01 | 12.01 |
| 0.001 | 12.001 |
| 趨近 0 | 12 |

因為平方函數：

$$
L=z^2
$$

Derivative：

$$
\frac{dL}{dz}=2z
$$

當：

$$
z=6
$$

所以：

$$
\boxed{\frac{dL}{dz}=12}
$$

這是 Chain Rule 的第二段。

---

## Part 6｜第一次完整理解 Chain Rule

### 十五、把兩段 Derivative 連起來

目前有兩個函數：

$$
z=3w
$$

$$
L=z^2
$$

當：

$$
w=2
$$

則：

$$
z=6,\quad L=36
$$

我們已經算出：

**第一段：Weight → z**

$$
\frac{dz}{dw}=3
$$

**第二段：z → Loss**

$$
\frac{dL}{dz}=12
$$

現在想知道：

Weight 發生極小變化時，Loss 的總體局部變化率是多少？

Chain Rule：

$$
\frac{dL}{dw}
=
\frac{dL}{dz}
\times
\frac{dz}{dw}
$$

代入：

$$
=12\times3
$$

得到：

$$
\boxed{\frac{dL}{dw}=36}
$$

#### 為什麼要相乘？

因為 Weight 的微小變化先被第一段函數放大 3 倍，再被第二段函數以當前局部變化率 12 倍影響 Loss。

因此總體局部變化率為：

$$
3\times12=36
$$

這就是 Chain Rule 的基本原理。

---

### 十六、使用微小變化量驗證 Chain Rule

現在已知：

$$
\frac{dL}{dw}=36
$$

假設 Weight 增加：

$$
\Delta w=0.001
$$

由 Derivative 的局部近似：

$$
\Delta L\approx\frac{dL}{dw}\Delta w
$$

因此：

$$
\Delta L\approx36(0.001)
$$

$$
\boxed{\Delta L\approx0.036}
$$

#### 原始回答

> dL = 36 × 0.001

正確。

#### 再用原始函數驗證

由：

$$
z=3w
$$

$$
L=z^2
$$

可以合併得到：

$$
L=9w^2
$$

原本：

$$
w=2
$$

$$
L_{\text{old}}=9(2)^2=36
$$

Weight 增加 0.001：

$$
w_{\text{new}}=2.001
$$

新的 Loss：

$$
L_{\text{new}}=9(2.001)^2
$$

$$
=36.036009
$$

實際變化量：

$$
\Delta L=0.036009
$$

與 Derivative 估計：

$$
0.036
$$

非常接近。

這代表 Chain Rule 得到的局部變化率，確實能用來估計極小參數變動所造成的 Loss 變化。

**驗收：PASS**

---

## Part 7｜使用 Chain Rule 的結果更新 Weight

### 十七、Gradient Descent 最終計算

已知：

$$
w=2
$$

$$
Gradient=36
$$

$$
\alpha=0.01
$$

Gradient Descent：

$$
w_{\text{new}}=w-\alpha\times Gradient
$$

代入：

$$
w_{\text{new}}=2-(0.01\times36)
$$

$$
=2-0.36
$$

$$
\boxed{w_{\text{new}}=1.64}
$$

#### 原始回答

> 2 − (0.01 × 36) = 2 − 0.36 = 1.64。

**驗收：PASS**

#### 驗證 Loss 是否下降

原本：

$$
z_{\text{old}}=3(2)=6
$$

$$
L_{\text{old}}=6^2=36
$$

更新後：

$$
z_{\text{new}}=3(1.64)=4.92
$$

$$
L_{\text{new}}=4.92^2=24.2064
$$

比較：

| 項目 | 更新前 | 更新後 |
|---|---:|---:|
| Weight | 2 | 1.64 |
| z | 6 | 4.92 |
| Loss | 36 | 24.2064 |

因此：

$$
\boxed{Loss:36\rightarrow24.2064}
$$

這次 Gradient Descent 成功降低 Loss。

---

## Part 8｜完整數學推導總整理

### 十八、從代數到 Gradient Descent

#### Step 1：定義簡化函數

$$
L=w^2
$$

#### Step 2：增加微小變化量

$$
w_{\text{new}}=w+h
$$

#### Step 3：計算 Loss 變化量

$$
\Delta L=(w+h)^2-w^2
$$

$$
\Delta L=2wh+h^2
$$

#### Step 4：計算 Average Rate of Change

$$
\frac{\Delta L}{h}=2w+h
$$

#### Step 5：讓 h 趨近 0

$$
\frac{dL}{dw}=2w
$$

#### Step 6：加入中間函數

$$
z=3w
$$

$$
L=z^2
$$

#### Step 7：計算各段 Derivative

$$
\frac{dz}{dw}=3
$$

$$
\frac{dL}{dz}=2z
$$

#### Step 8：Chain Rule

$$
\frac{dL}{dw}
=
\frac{dL}{dz}
\frac{dz}{dw}
$$

當 $w=2$、$z=6$：

$$
\frac{dL}{dw}=12\times3=36
$$

#### Step 9：Gradient Descent

$$
w_{\text{new}}=w-\alpha\frac{dL}{dw}
$$

當 $\alpha=0.01$：

$$
w_{\text{new}}=2-0.01(36)=1.64
$$

#### Step 10：驗證結果

$$
L_{\text{old}}=36
$$

$$
L_{\text{new}}=24.2064
$$

Loss 下降，完成一次 Parameter Update。

---

## Part 9｜最重要的數學符號

| 符號 | 英文名稱 | 意義 |
|---|---|---|
| $w$ | Weight | 模型參數 |
| $z$ | Intermediate Variable / Logit | 中間運算結果；在 Logistic Regression 中可稱 Logit |
| $L$ | Loss | 預測錯誤的衡量值 |
| $h$ | Increment | 輸入增加的一小段數值 |
| $\Delta$ | Delta | 有限的變化量 |
| $\frac{\Delta L}{\Delta w}$ | Average Rate of Change | 有限區間的平均變化率 |
| $\frac{dL}{dw}$ | Derivative | 某個位置的瞬間變化率 |
| $\frac{\partial L}{\partial w}$ | Partial Derivative | 多變數函數對指定變數的變化率 |
| $\nabla L$ | Gradient | 所有參數 Partial Derivatives 組成的 Vector |
| $\alpha$ | Learning Rate | Parameter Update 的步長係數 |
| $\leftarrow$ | Assignment / Update | 將計算結果更新至左邊變數 |
| $\approx$ | Approximately Equal | 近似相等 |
| $\lim$ | Limit | 觀察變數趨近指定值時的結果 |

---

## Part 10｜本次錯誤與修正紀錄

| 曾經卡住的問題 | 原因 | 修正後的理解 |
|---|---|---|
| 為什麼可以假設 $L=w^2$ | 不知道這是教學簡化函數 | 它只是用來學習 Derivative，不是固定的 ML Loss |
| 為什麼出現 $2wh+h^2$ | 不熟悉乘法分配律 | 從 $(w+h)(w+h)$ 展開 |
| 為什麼要減去 $w^2$ | 不理解 Delta | $\Delta L$ 是 New Loss 減去 Old Loss |
| 為什麼 Derivative 是 $2w$ | 不理解變化率與 Limit | 從 $(2wh+h^2)/h$ 推導 |
| 為什麼 Gradient Descent 要減 | 誤以為 Weight 必須一直減少 | 負號代表沿 Gradient 的反方向更新 |
| 把 $z=2w$ 誤算成平方函數 | 混淆不同函數形式 | Linear 與 Quadratic Function 的變化量不同 |
| 為什麼 Chain Rule 要相乘 | 不理解中間函數的依賴 | 每段局部變化率相乘，得到總體局部變化率 |
| Weight 更新的簡單算術錯誤 | 減法計算失誤 | 以更新後的數值與 Loss 再驗證 |

---

## Part 11｜最終驗收狀態

| 學習概念 | 狀態 |
|---|---|
| Perfect Square Identity | PASS |
| Algebraic Expansion | PASS |
| Delta / Change Calculation | PASS |
| Average Rate of Change | PASS |
| Derivative 基本概念 | PASS |
| Linear Function Derivative | PASS |
| Quadratic Function Derivative | PASS |
| Chain Rule（兩段函數） | PASS |
| Gradient Direction | PASS |
| Gradient Descent Update | PASS |
| Loss Reduction Verification | PASS |
| Partial Derivative（多參數） | 待補強 |
| Matrix Gradient | 待補強 |
| Sigmoid + BCE Chain Rule | 待補強 |
| Backpropagation | Phase 03 正式學習 |

**本輪結論：Derivative / Chain Rule Foundations — PASS。**

但目前是當下教學驗收，仍需要間隔後的無提示回憶，才能確認長期掌握程度。

---

## Part 12｜後續與 Logistic Regression 的關係

本次已理解：

**Weight → z → Loss**

對應：

$$
z=3w
$$

$$
L=z^2
$$

後續 Logistic Regression 則是：

**Weight → Logit → Probability → Loss**

$$
z=w^Tx+b
$$

$$
p=\sigma(z)
$$

$$
L=BCE(p,y)
$$

因此 Logistic Regression 的 Chain Rule 為：

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial p}
\frac{\partial p}{\partial z}
\frac{\partial z}{\partial w}
$$

**它不是另一套完全不同的數學，而是在已理解的 Chain Rule 中增加一個中間函數。**

下一輪只需要補充 Sigmoid Derivative 與 BCE Derivative，再理解它們如何化簡成：

$$
\boxed{\nabla_wL=\frac{1}{n}X^T(p-y)}
$$

這是後續 Phase 02 Neural Network 與 Phase 03 Backpropagation 的重要基礎。

---

### 最短複習記憶卡

**Derivative：** 函數在某個位置的瞬間變化率。

**Chain Rule：** 當一個變數透過多個函數間接影響結果時，將每一段局部變化率相乘。

**Gradient：** Loss 對模型參數的局部變化率；多參數時形成 Vector。

**Gradient Descent：** 沿 Gradient 的反方向更新參數，嘗試降低 Loss。

**本次最重要的數字：**

$$
\frac{dz}{dw}=3
$$

$$
\frac{dL}{dz}=12
$$

$$
\frac{dL}{dw}=12\times3=36
$$

$$
w_{\text{new}}=2-0.01(36)=1.64
$$

$$
Loss:36\rightarrow24.2064
$$

**完整主線：**

Algebra → Delta → Rate of Change → Derivative → Chain Rule → Gradient → Gradient Descent → Loss Verification。
