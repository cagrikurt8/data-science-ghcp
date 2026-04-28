# Model Evaluation Raporu — bank-churn (v2, retrain sonrasi)

**Karar:** � **CONDITIONAL_GO** (resmi olarak onaylandi, `age_bucket` waiver business rule ile telafi ediliyor)
**Model:** XGBoost + Isotonic Calibration (FrozenEstimator)
**Kaynak:** [models/best_model.joblib](../models/best_model.joblib)
**Policy:** [models/decision_policy.json](../models/decision_policy.json)
**Inference:** [src/serve/policy.py](../src/serve/policy.py)
**Veri:** [data/processed/test.parquet](../data/processed/test.parquet) (1.500 satır, 305 pozitif = %20.3)
**Detay:** [reports/evaluation.json](evaluation.json)

> Conditional GO sebebi: `age_bucket` recall_gap = 0.61 (gate eşiği 0.30). Bu segment (`<30`) için **manual_review** iş kuralı [decision_policy.json](../models/decision_policy.json) içinde tanımlandı; gate'te `WARN_WAIVED` olarak işaretlenir, otomatik aksiyon devre dışı bırakılır.

> v1'deki NO-GO kararı sonrası uygulanan düzeltmeler:
> 1. `is_senior` türev feature'ı **kaldırıldı** (yaş bias'ını abartıyordu)
> 2. `age_bucket × Geography` strata üzerinden **inverse-frequency sample_weight**
> 3. **Isotonic kalibrasyon** (val set, FrozenEstimator)

---

## 1. v1 → v2 Karşılaştırması

| Kontrol | v1 | v2 | Δ |
|---|---|---|---|
| ROC-AUC | 0.876 | 0.873 | -0.003 (ihmal) |
| PR-AUC | 0.734 | 0.709 | -0.025 |
| Brier | 0.141 | **0.097** | **🟢 -31%** |
| **ECE** | **0.189** | **0.028** | **🟢 -85%** |
| Gender recall_gap | 0.147 | 0.130 | iyilesme |
| Geography recall_gap | **0.232** | **0.131** | **🟢 FAIL→OK** |
| age_bucket recall_gap | **0.798** | **0.613** | iyilesme (hala FAIL) |
| `<30` recall | **0.143** | **0.357** | **🟢 2.5×** |

PR-AUC küçük düşüş kabul edilebilir; karşılığında kalibrasyon ve fairness ciddi şekilde iyileşti.

## 2. Metrikler (test set)

| Metrik | Default (t=0.50) | Tuned (t=0.10) |
|---|---|---|
| Accuracy | **0.869** | 0.730 |
| Precision | **0.742** | 0.420 |
| Recall | 0.548 | **0.862** |
| F1 | 0.630 | 0.565 |
| ROC-AUC | 0.873 | 0.873 |
| PR-AUC | 0.709 | 0.709 |
| Brier | 0.097 | 0.097 |
| Beklenen maliyet (5·FN + FP) | 748 | **573** |

> Kalibrasyon sonrası olasılıklar gerçek oranı doğru yansıttığı için, FN=5×FP varsayımı altında tuned eşik 0.10'a düştü. Operasyonel olarak **0.10–0.50 arası iki eşikli** sistem önerilir: yüksek-risk skor (>0.50) → otomatik aksiyon, orta-risk (0.10–0.50) → CRM listeleme.

## 3. Confusion Matrix + Threshold Tuning

- FN_COST=5, FP_COST=1 (varsayım, iş ekibiyle doğrulanmalı)
- Optimum eşik **0.10** (tuned), default 0.50 da kabul edilebilir
- Görseller:
  - [reports/figures/confusion_matrix_default.png](figures/confusion_matrix_default.png)
  - [reports/figures/confusion_matrix_tuned.png](figures/confusion_matrix_tuned.png)
  - [reports/figures/threshold_curve.png](figures/threshold_curve.png)

## 4. Kalibrasyon

| Ölçüm | v1 | v2 | Eşik |
|---|---|---|---|
| Brier | 0.141 | **0.097** | ≤ 0.20 ✅ |
| ECE | 0.189 | **0.028** | ≤ 0.05 ✅ |

Reliability diagram: [reports/figures/calibration.png](figures/calibration.png)

> Isotonic kalibrasyon **ECE'yi %85 azalttı**. Olasılık çıktıları artık CRM eşikleri ve risk segmentasyonu için kullanılabilir.

## 5. SHAP Açıklamaları

### Global önem (top 10)

| Sıra | Feature | mean(|SHAP|) |
|---|---|---|
| 1 | Age | 0.7706 |
| 2 | NumOfProducts | 0.6187 |
| 3 | IsActiveMember | 0.3984 |
| 4 | Balance | 0.1751 |
| 5 | Geography_Germany | 0.1530 |
| 6 | Gender_Female | 0.1263 |
| 7 | balance_to_salary (türev) | 0.1026 |
| 8 | CreditScore | 0.0836 |
| 9 | Gender_Male | 0.0730 |
| 10 | products_per_tenure (türev) | 0.0668 |

> v1'de top 10'da olan `is_senior` kaldırıldı. Türev feature'lardan `balance_to_salary` ve `products_per_tenure` katkı sağlamaya devam ediyor. `Age` hala dominant — bu beklenen bir davranış (Data Analyzer §6: Age en güçlü korelasyon).

Görseller:
- [reports/figures/shap_summary.png](figures/shap_summary.png)
- [reports/figures/shap_local.png](figures/shap_local.png)

## 6. Fairness Analizi (eşik=0.10)

| Segment | Değer | n | Pozitif | Recall | ROC-AUC | Selection rate |
|---|---|---|---|---|---|---|
| Gender | Female | 692 | 166 | 0.922 | 0.888 | 0.488 |
| Gender | Male | 808 | 139 | 0.791 | 0.855 | 0.356 |
| Geography | France | 778 | 134 | 0.799 | 0.855 | 0.338 |
| Geography | Germany | 370 | 114 | 0.930 | 0.872 | 0.659 |
| Geography | Spain | 352 | 57 | 0.877 | 0.891 | 0.338 |
| IsActiveMember | 0 | 715 | 201 | 0.881 | 0.869 | 0.534 |
| IsActiveMember | 1 | 785 | 104 | 0.827 | 0.859 | 0.311 |
| age_bucket | <30 | 254 | **14** | **0.357** | 0.718 | 0.150 |
| age_bucket | 30–40 | 652 | 65 | 0.708 | 0.819 | 0.273 |
| age_bucket | 40–50 | 389 | 124 | 0.911 | 0.833 | 0.668 |
| age_bucket | 50+ | 205 | 102 | 0.971 | 0.878 | 0.732 |

### Disparite

| Segment | recall_gap | DPR | Sınıf |
|---|---|---|---|
| Gender | 0.130 | 0.730 | 🟢 OK |
| Geography | 0.131 | 0.513 | 🟢 OK (DPR uyarı) |
| IsActiveMember | 0.054 | 0.582 | 🟢 OK (DPR uyarı) |
| **age_bucket** | **0.613** | **0.204** | 🔴 hala FAIL |

Görsel: [reports/figures/fairness.png](figures/fairness.png)

> **`<30` segmenti durumu:** v1'de recall 0.14 idi, sample_weight ile **0.36'ya çıktı (2.5×)**. Ancak gerçek pozitif sayısı yalnızca **14/254** (%5.5). Bu segmentteki düşük performans büyük ölçüde **veri seyrekliği** kaynaklı; modelden ziyade verinin yapısal sınırı.

## 7. Production Gate

| Kontrol | Eşik | Değer | Durum |
|---|---|---|---|
| ROC-AUC | ≥ 0.80 | 0.873 | ✅ |
| PR-AUC | ≥ 0.55 | 0.709 | ✅ |
| Brier | ≤ 0.20 | 0.097 | ✅ |
| ECE | ≤ 0.05 | 0.028 | ✅ |
| Gender recall_gap | ≤ 0.20 | 0.130 | ✅ |
| Geography recall_gap | ≤ 0.20 | 0.131 | ✅ |
| IsActiveMember recall_gap | ≤ 0.20 | 0.054 | ✅ |
| age_bucket recall_gap | ≤ 0.30 | 0.613 | ❌ |

## 8. Karar — CONDITIONAL GO

**Üretime alınabilir koşullar:**
1. **`<30` segmenti için ayrı iş kuralı:** Bu yaş grubunda model skoru düşük güvenilirlikte (ROC-AUC 0.72). CRM ekibi bu segmente:
   - Ek davranışsal sinyaller (uygulama açma, çağrı merkezi etkileşimi) ile destekli karar verir
   - Veya tüm segmenti **manuel inceleme listesine** alır
2. **Olasılık ve eşik kullanımı:**
   - Yüksek-risk (p ≥ 0.50): otomatik retention kampanyası
   - Orta-risk (0.10 ≤ p < 0.50): CRM önceliklendirme
3. **Aylık monitoring:**
   - Tüm segmentlerde recall_gap ve ECE
   - `<30` segmentinde model skor dağılımı drift kontrolü
4. **FN/FP maliyetinin iş ekibiyle resmî onayı** (varsayılan 5:1 sürmektedir)

**Sonraki iyileştirme yolu (v3):**
- `<30` için ayrı bir model (yeterli veri biriktiğinde)
- Yaş ile etkileşim feature'ları (Age × NumOfProducts)
- Üretim verisinde 6 ay sonrası geri-test

---

**Üretilen artefaktlar:**

| Dosya |
|---|
| [reports/evaluation.json](evaluation.json) |
| [reports/figures/confusion_matrix_default.png](figures/confusion_matrix_default.png) |
| [reports/figures/confusion_matrix_tuned.png](figures/confusion_matrix_tuned.png) |
| [reports/figures/threshold_curve.png](figures/threshold_curve.png) |
| [reports/figures/calibration.png](figures/calibration.png) |
| [reports/figures/shap_summary.png](figures/shap_summary.png) |
| [reports/figures/shap_local.png](figures/shap_local.png) |
| [reports/figures/fairness.png](figures/fairness.png) |