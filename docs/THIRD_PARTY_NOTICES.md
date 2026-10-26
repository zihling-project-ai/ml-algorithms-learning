# Third-Party Dependencies and Content Review

審查日期：2026-10-26。本文件是套件與內容來源紀錄，不授予本專案新的授權，也不是完整法律權利認證。

## 直接依賴

| 套件 | 本輪版本 | 本機 distribution 的授權資訊 | 用途 |
|---|---|---|---|
| NumPy | 2.5.4 | BSD-3-Clause；distribution 另列 0BSD、MIT、Zlib、CC0-1.0 組件 | 陣列、線性代數、亂數 |
| Matplotlib | 3.11.2 | Matplotlib License Agreement／相容 PSF 的授權條款 | 圖表 |
| XGBoost（選用） | 3.4.1 | Apache-2.0 | 唯一的第三方模型訓練示範 |

NumPy 與 Matplotlib 實作演算法所需的數值與繪圖工具；並未將函式庫內部原始碼複製進 Repository。XGBoost 模型不宣稱 From Scratch。

## 轉移依賴

Matplotlib 的 contourpy、cycler、fonttools、kiwisolver、packaging、pillow、pyparsing、python-dateutil、six，以及本次解析得到的 tzdata；XGBoost 的 SciPy，均由套件管理器安裝。完整實測版本見 requirements-lock.txt。

本機 NumPy／SciPy distribution 包含其組件的授權清單。Repository 不隨附這些套件、編譯二進位、字型檔或完整虛擬環境。若日後要發佈打包應用／容器或 vendored 程式碼，需針對實際隨附內容重新處理 notices。

## 教學內容與圖檔

- 數據由本專案程式生成，未發現外部 CSV、資料庫匯出、真實設備／客戶紀錄或第三方訓練資料。
- 正式實驗 PNG 由 Matplotlib 產生；少量靜態教學圖保存於 images/，不全部由主線腳本重新產生。
- 已檢查 PNG 的文字 metadata，沒有發現本機絕對路徑、電子郵件或憑證模式。
- 圖表像素內容另做代表性人工檢視；自動 metadata 掃描不能證明所有圖片完全沒有敏感內容。
- 筆記記錄個人學習過程與 AI 協作修正；未發現宣告複製其他程式庫原始碼的區段。此項檢查不等同對每段內容作完整來源鑑定。
- 現有 Boosting 筆記附官方文件連結，僅保存連結與教學整理，沒有打包外部文件全文。

## 專案自身授權：待擁有人決定

尚未新增根目錄 LICENSE。公開前需確認擬採授權，以及現有程式、教學圖與筆記是否均有權以該授權公開。第三方套件可安裝使用，不代表本專案自動套用同一授權。

本輪沒有發布任何檔案至 GitHub，也沒有替擁有人接受新的授權條款或設定公開可見性。
