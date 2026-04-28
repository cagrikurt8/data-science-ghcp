# Feature Engineering Raporu

**Pipeline modülü:** [src/features/pipeline.py](../src/features/pipeline.py)
**Serileştirilmiş preprocessor:** `models/preprocessor.joblib`
**Çıktı parquet'ler:** `data/processed/{train,val,test}.parquet`
**Referans:** [reports/data_quality.md](data_quality.md) (Data Analyzer raporu)

## Split

| Set | Satır | Hedef oranı (Exited=1) |
|---|---|---|
| train | 7.000 | %20.37 |
| val   | 1.500 | %20.40 |
| test  | 1.500 | %20.33 |

- Stratified split, `random_state=42`, oranlar 70/15/15
- `fit` yalnızca **train** üzerinde çağrıldı → val/test sızıntısı yok
- Tüm split'lerde hedef oranı korundu (Data Analyzer R3'e uygun)

## Drop Edilen Kolonlar

| Kolon | Sebep |
|---|---|
| RowNumber | Satır indeksi, tahmin gücü yok |
| CustomerId | Banka içi kimlik (PII benzeri) |
| Surname | PII (Data Analyzer R7 — EDA'da hash'lendi, modelleme dışı) |

## Feature Listesi (17 adet, transform çıktısı)

### Sayısal — median impute + StandardScaler

| Feature | Tür | Dönüşüm | Gerekçe |
|---|---|---|---|
| CreditScore | int → float | impute(median) → standard scale | Ham; Data Analyzer §1 |
| Age | int → float | impute(median) → standard scale | En güçlü korelasyon (r=+0.285), Data Analyzer §6 |
| Tenure | int → float | impute(median) → standard scale | Ham |
| Balance | float | impute(median) → standard scale | Bimodal; Data Analyzer R4 |
| NumOfProducts | int → float | impute(median) → standard scale | 3–4 segmentinde yüksek churn (R1) |
| EstimatedSalary | float | impute(median) → standard scale | Uniform; Data Analyzer R5 — düşük tahmin gücü uyarısıyla bırakıldı |
| HasCrCard | binary | passthrough sayısal olarak ölçeklenir | Ham; korelasyon ihmal edilebilir ama maliyetsiz |
| IsActiveMember | binary | passthrough sayısal olarak ölçeklenir | Hedefle r=-0.156, Data Analyzer §6 |

### Türetilmiş Sayısal — Data Analyzer önerilerinden

| Feature | Formül | Gerekçe |
|---|---|---|
| has_balance | `(Balance > 0).astype(int)` | Data Analyzer R4 — %36 müşteri Balance=0, bimodal yapıyı sinyalize eder |
| balance_to_salary | `Balance / (EstimatedSalary + 1)` | Servet/gelir oranı; uniform `EstimatedSalary` (R5) tek başına zayıf, ratio bilgi taşıyabilir |
| products_per_tenure | `NumOfProducts / (Tenure + 1)` | Yıl başına ürün edinimi → hız sinyali (R1 ile etkileşim) |
| is_senior | `(Age >= 60).astype(int)` | Data Analyzer R6 — sağa çarpık yaş kuyruğu; doğrusal olmayan yaş etkisini yakalar |

> **Sızıntı kontrolü:** Tüm türev feature'lar yalnızca girdi kolonlarından üretilir; hedefe (`Exited`) doğrudan/dolaylı bağımlı değildir. Ayrıca türetme `fit` öncesi (train üzerinde) ve `transform` sırasında deterministik hesaplanır → val/test'e parametre kaçışı yok.

### Kategorik — most_frequent impute + OneHotEncoder

| Ham kolon | Çıkan feature'lar | Not |
|---|---|---|
| Geography | `Geography_France`, `Geography_Germany`, `Geography_Spain` | 3 sınıf; Almanya yüksek-risk segmenti (Data Analyzer R2) |
| Gender | `Gender_Female`, `Gender_Male` | 2 sınıf |

`handle_unknown="ignore"` → inference'da görülmemiş değer gelirse sıfır vektör.

## Pipeline Mimarisi

```
Pipeline
├── derived: DerivedFeatures (custom transformer; 4 türev kolon ekler)
└── preprocess: ColumnTransformer
    ├── num: SimpleImputer(median) → StandardScaler
    │   (8 ham + 4 türev = 12 sayısal)
    └── cat: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown=ignore)
        (Geography, Gender → 5 one-hot)
```

Toplam çıktı: **17 feature**.

## Uygulanmayan / Bilinçli Tercihler

- **Target encoding** kullanılmadı: Geography ve Gender düşük kardinaliteli (≤3); OneHot daha güvenli, sızıntıya açık değil.
- **`NumOfProducts` bin'leme yapılmadı**: Sayısal olarak tutuldu; ağaç tabanlı modeller 3–4 segmentini doğal yakalar. Doğrusal modelde gerekirse `02_features.ipynb`'te eklenebilir.
- **Outlier kırpma yapılmadı**: StandardScaler yeterli; ağaç tabanlı modeller etkilenmez.
- **SMOTE / class_weight** burada uygulanmadı; bu modelleme aşamasının kararı (Data Analyzer R3).

## Kullanım

```python
from src.features.pipeline import run
result = run()  # parquet ve joblib üretir

# Inference
import joblib, pandas as pd
pre = joblib.load("models/preprocessor.joblib")
X = pre.transform(new_df)  # new_df ham kolonları içermeli (RowNumber/CustomerId/Surname/Exited hariç)
```

CLI:

```powershell
uv run python -c "from src.features.pipeline import run; run()"
```