# Veri Kalitesi Raporu — Churn_Modelling

**Veri seti:** `data/raw/Churn_Modelling.csv`
**Boyut:** 10.000 satır × 14 sütun
**Hedef değişken:** `Exited` (1 = churn)
**EDA notebook:** [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb)
**Maskelenmiş snapshot:** `data/processed/churn_masked.parquet`

---

## 1. Şema

| Sütun | Tip | Rol | Not |
|---|---|---|---|
| RowNumber | int64 | drop | Satır indeksi, modele girmez |
| CustomerId | int64 | drop | Banka içi kimlik, modele girmez |
| Surname | object | **PII** | SHA-256 ile hash'lendi |
| CreditScore | int64 | feature | 350–850 |
| Geography | object | feature | France / Germany / Spain |
| Gender | object | feature | Male / Female |
| Age | int64 | feature | 18–92 |
| Tenure | int64 | feature | 0–10 yıl |
| Balance | float64 | feature | 0–250.898 |
| NumOfProducts | int64 | feature | 1–4 |
| HasCrCard | int64 | feature (binary) | 0/1 |
| IsActiveMember | int64 | feature (binary) | 0/1 |
| EstimatedSalary | float64 | feature | 11–199.992 |
| Exited | int64 | **target** | 0/1 |

## 2. Veri Kalitesi Bulguları

| Kontrol | Sonuç | Etiket |
|---|---|---|
| Eksik değer | Hiçbir sütunda yok (0%) | 🟢 |
| Tam satır duplicate | 0 | 🟢 |
| `CustomerId` duplicate | 0 | 🟢 |
| Tip uyumsuzluğu | Yok | 🟢 |
| PII sütunu (`Surname`) | Hash'lendi | 🟢 |
| Düşük varyans / sabit sütun | Yok | 🟢 |

## 3. Hedef Dağılımı (Sınıf Dengesizliği)

| Exited | Oran |
|---|---|
| 0 (kalan) | %79.63 |
| 1 (churn) | %20.37 |

> **Etiket: 🟡** — Sınıf dengesizliği ~%20'de. Modelleme aşamasında `class_weight`, stratified split veya SMOTE/undersampling değerlendirilmeli.

## 4. Sayısal Değişkenler — Outlier (IQR 1.5×)

| Sütun | Outlier sayısı | Oran | Etiket |
|---|---|---|---|
| CreditScore | 15 | %0.15 | 🟢 |
| Age | 359 | %3.59 | 🟡 (sağa çarpık, yaşlı müşteri kuyruğu) |
| Tenure | 0 | %0.00 | 🟢 |
| Balance | 0 | %0.00 | 🟢 (ama %36.17 müşterinin Balance = 0) |
| NumOfProducts | 326 | %3.26 | 🟡 (3–4 ürünlü uç vakalar; churn ile güçlü ilişkili) |
| EstimatedSalary | 0 | %0.00 | 🟢 (uniform dağılım) |

> `Balance == 0` segmenti %36 — ayrı bir "no-balance" feature türetilmeli.
> `EstimatedSalary` neredeyse uniform → muhtemelen sentetik; tahmin gücü düşük olacak.

## 5. Kategorik Değişkenler — Kardinalite

| Sütun | Eşsiz değer | Etiket |
|---|---|---|
| Geography | 3 (France 5014, Germany 2509, Spain 2477) | 🟢 |
| Gender | 2 (Male 5457, Female 4543) | 🟢 |
| NumOfProducts | 4 (1: 5084, 2: 4590, 3: 266, 4: 60) | 🟡 (3 ve 4 sınıfı seyrek) |
| Surname (hash) | 2932 | bilgi amaçlı, modele alınmaz |

## 6. Hedef ile İlişki

### Pearson korelasyon (Exited)
| Değişken | r |
|---|---|
| Age | **+0.285** |
| Balance | +0.119 |
| EstimatedSalary | +0.012 |
| HasCrCard | -0.007 |
| Tenure | -0.014 |
| CreditScore | -0.027 |
| NumOfProducts | -0.048 |
| IsActiveMember | **-0.156** |

### Segment bazlı churn oranı
| Segment | Churn oranı | Not |
|---|---|---|
| Geography = Germany | **%32.4** | France %16.2, Spain %16.7 — 🔴 ana risk segmenti |
| Gender = Female | %25.1 | Male %16.5 — 🟡 |
| IsActiveMember = 0 | %26.9 | Aktif olmayan üye → 🟡 |
| NumOfProducts = 3 | **%82.7** | 🔴 |
| NumOfProducts = 4 | **%100.0** | 🔴 (60 müşteri, hepsi churn — özel davranış) |
| NumOfProducts = 2 | %7.6 | en sadık segment |

## 7. Risk / Uyarılar

| # | Bulgu | Etiket |
|---|---|---|
| R1 | `NumOfProducts = 4` segmentinde %100 churn → veri sızıntısı şüphesi; modelde hedef-leakage olmadığı doğrulanmalı | 🔴 |
| R2 | Almanya segmenti diğer ülkelerin ~2 katı churn oranına sahip | 🔴 |
| R3 | Sınıf dengesizliği %20 → metrik olarak yalnız accuracy kullanılmamalı (ROC-AUC, PR-AUC, F1 öncelikli) | 🟡 |
| R4 | `Balance` sütununun %36'sı sıfır → bimodal dağılım; "has_balance" türev feature önerilir | 🟡 |
| R5 | `EstimatedSalary` uniform → düşük tahmin gücü, doğrulanmalı | 🟡 |
| R6 | `Age` sağa çarpık, 60+ kuyruğunda outlier → ölçekleme/bin'leme düşünülmeli | 🟡 |
| R7 | `Surname` PII → hash'lendi, ham hâli `data/processed/` dışına yazılmıyor | 🟢 |
| R8 | Eksik değer / duplicate yok, şema temiz | 🟢 |

## 8. Sonraki Adım Önerileri (modelleme dışı)

- `data/processed/churn_masked.parquet` üzerinden feature engineering notebook'u (`02_features.ipynb`)
- Önerilen türev özellikler: `has_balance`, `balance_to_salary`, `products_per_tenure`, `geo_x_gender`
- Train/val/test ayrımı **stratified** ve `random_state=42` ile yapılmalı
- `NumOfProducts in {3,4}` davranışı ürün ekibiyle doğrulanmalı (kampanya/hesap kapatma sürecinin yan etkisi olabilir)

---

**Grafikler:** `reports/figures/` altında
- `target_distribution.png`
- `numeric_distributions.png`
- `categorical_counts.png`
- `churn_rate_by_segment.png`
- `correlation_heatmap.png`