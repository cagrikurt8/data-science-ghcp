# Model Evaluation Raporu — bank-churn

**Karar:** 🔴 **NO-GO (mevcut hâliyle üretime uygun değil)**
**Model:** XGBoost (Optuna HPO sonucu best_model)
**Kaynak:** [models/best_model.joblib](../models/best_model.joblib)
**Veri:** [data/processed/test.parquet](../data/processed/test.parquet) (1.500 satır, 305 pozitif = %20.3)
**Üretim notları:** [reports/evaluation.json](evaluation.json) — JSON tüm sayısal değerler

> ModelEvaluator agent talimatı `job` ve `age_bucket` segmentlerini istedi. Veri seti `job` içermez; bunun yerine `Geography`, `Gender`, `IsActiveMember` segmentleri eklendi. `age_bucket` `Age`'den türetildi: `<30`, `30–40`, `40–50`, `50+`.

---

## 1. Metrikler (test set)

| Metrik | Default (t=0.50) | Tuned (t=0.52) |
|---|---|---|
| Accuracy | 0.8027 | **0.8213** |
| Precision | 0.5097 | **0.5427** |
| Recall | **0.7738** | 0.7705 |
| F1 | 0.6146 | **0.6369** |
| ROC-AUC | 0.8757 | 0.8757 |
| PR-AUC | 0.7342 | 0.7342 |
| Brier | 0.1413 | 0.1413 |
| Beklenen maliyet (5·FN + FP) | 572 | **548** |

> Sıralama bazlı metrikler (ROC-AUC, PR-AUC, Brier) eşik bağımsızdır → değişmez. Threshold tuning F1 ve maliyet üzerinde marjinal kazanç sağladı.

## 2. Confusion Matrix + Threshold Tuning

- **Maliyet varsayımı:** FN_COST = 5 (churn kaçırmak), FP_COST = 1. Bankacılık iş bağlamında ayrılan müşteriyi kaçırmak yanlış alarmdan ~5× pahalıdır (varsayım — iş ekibiyle doğrulanmalı).
- **Optimum eşik:** **0.52** (90 noktalı grid)
- Görseller:
  - [reports/figures/confusion_matrix_default.png](figures/confusion_matrix_default.png)
  - [reports/figures/confusion_matrix_tuned.png](figures/confusion_matrix_tuned.png)
  - [reports/figures/threshold_curve.png](figures/threshold_curve.png)

> Tuned eşikte 70 FN ve 198 FP gözlemlendi; toplam maliyet 548. F1 0.62 → 0.64'e çıktı.

## 3. Kalibrasyon

| Ölçüm | Değer | Yorum |
|---|---|---|
| Brier | 0.141 | İyi (< 0.20 üretim eşiği) |
| **ECE (10 quantile bin)** | **0.189** | 🟡 yüksek — model olasılıkları gerçek oranı **olduğundan büyük gösteriyor** |

Reliability diagram: [reports/figures/calibration.png](figures/calibration.png)

> XGBoost `scale_pos_weight` kullanıldığı için pozitif sınıf olasılıkları sistematik olarak şişiyor. Üretimden önce **Platt scaling** veya **isotonic regression** ile post-hoc kalibrasyon zorunlu.

## 4. SHAP Açıklamaları

### Global önem (top 10, 500 satır örneklem)

| Sıra | Feature | mean(\|SHAP\|) |
|---|---|---|
| 1 | Age | 0.7434 |
| 2 | NumOfProducts | 0.6303 |
| 3 | IsActiveMember | 0.3154 |
| 4 | Geography_Germany | 0.1883 |
| 5 | Gender_Female | 0.1681 |
| 6 | Balance | 0.1089 |
| 7 | balance_to_salary (türev) | 0.0583 |
| 8 | CreditScore | 0.0447 |
| 9 | Geography_France | 0.0446 |
| 10 | EstimatedSalary | 0.0321 |

> Türev feature'lar arasında `balance_to_salary` top 10'a girdi. `is_senior`, `has_balance`, `products_per_tenure` daha düşük katkı sağladı; ileride `Age`'in zaten güçlü olduğu düşünülürse `is_senior` çıkarılabilir.

Görseller:
- [reports/figures/shap_summary.png](figures/shap_summary.png) — global bar chart
- [reports/figures/shap_local.png](figures/shap_local.png) — 3 yerel açıklama

### Yerel açıklama (3 örnek)

| Etiket | Test idx | p(churn) | Yorum |
|---|---|---|---|
| Yüksek risk | 709 | 0.984 | Almanya, yaşlı, aktif değil profili |
| Düşük risk | 122 | 0.043 | Genç, aktif, NumOfProducts=2 |
| Sınır | 1082 | 0.500 | Ayırt edici sinyal yok — ek izleme önerilir |

## 5. Fairness Analizi

Eşik 0.52 sabit tutularak segment metrikleri (recall = churn yakalama oranı):

| Segment | Değer | n | Recall | ROC-AUC | Selection rate |
|---|---|---|---|---|---|
| Gender | Female | 692 | 0.837 | 0.890 | 0.358 |
| Gender | Male | 808 | 0.691 | 0.857 | 0.229 |
| Geography | France | 778 | 0.672 | 0.856 | 0.216 |
| Geography | Germany | 370 | 0.904 | 0.881 | 0.516 |
| Geography | Spain | 352 | 0.737 | 0.896 | 0.210 |
| IsActiveMember | 0 | 715 | 0.786 | 0.875 | 0.379 |
| IsActiveMember | 1 | 785 | 0.740 | 0.860 | 0.206 |
| age_bucket | <30 | 254 | 0.143 | 0.698 | 0.039 |
| age_bucket | 30–40 | 652 | 0.446 | 0.813 | 0.110 |
| age_bucket | 40–50 | 389 | 0.871 | 0.839 | 0.545 |
| age_bucket | 50+ | 205 | 0.941 | 0.887 | 0.678 |

### Disparite özeti

| Segment | recall_gap | ROC-AUC_gap | demographic_parity_ratio | Sınıf |
|---|---|---|---|---|
| Gender | 0.147 | 0.033 | 0.639 | 🟢 OK |
| Geography | 0.232 | 0.040 | 0.407 | 🟡 WARN (gap > 0.20, DPR < 0.6) |
| IsActiveMember | 0.046 | 0.015 | 0.544 | 🟡 WARN (DPR < 0.6) |
| **age_bucket** | **0.798** | **0.188** | **0.058** | 🔴 **FAIL** (gap > 0.30) |

Görsel: [reports/figures/fairness.png](figures/fairness.png)

> **Kritik bulgu:** `<30` yaş grubunda recall yalnızca 0.14, ROC-AUC 0.70. Model genç müşteri churn'ünü neredeyse yakalayamıyor. Bu grupta 254 müşteriden yalnızca 14 pozitif var → veri seyrekliği ana neden ama üretim için kabul edilemez bir performans.

## 6. Production Gate Sonuçları

| Kontrol | Durum |
|---|---|
| ROC-AUC ≥ 0.80 | ✅ 0.876 |
| PR-AUC ≥ 0.55 | ✅ 0.734 |
| Brier ≤ 0.20 | ✅ 0.141 |
| ECE ≤ 0.05 | 🟡 0.189 |
| Gender recall_gap ≤ 0.20 | ✅ 0.147 |
| Geography recall_gap ≤ 0.20 | 🟡 0.232 |
| IsActiveMember recall_gap ≤ 0.20 | ✅ 0.046 |
| **age_bucket recall_gap ≤ 0.30** | 🔴 **0.798** |

## 7. Karar — NO-GO

**Sebepler:**
1. **age_bucket fairness FAIL:** `<30` yaş grubunda recall 0.14 — model bu segmentte neredeyse rastgele. Üretime alınması hâlinde genç churn riski sistematik olarak kaçırılır.
2. **Kalibrasyon zayıf (ECE 0.19):** Olasılık çıktısı kampanya/CRM eşikleri için güvenilir değil.
3. **Geography disparitesi 🟡:** Almanya yüksek recall (0.90), Fransa düşük (0.67); maliyet/fayda raporlamasında bu bilgi şeffaf paylaşılmalı.

**Önerilen düzeltmeler (yeniden eğitim gerekir):**
- `age_bucket` ve `Geography` katmanlı **stratified k-fold** ile yeniden eğitim
- Genç segmentte ya **ayrı model** ya da **örneklem ağırlıklandırma** (sample_weight) — şu an 254 satırda 14 pozitif (%5.5)
- Post-hoc **isotonic kalibrasyon** (val set üzerinde) → ECE < 0.05 hedefi
- `is_senior` türev feature'ı yeniden değerlendir; muhtemelen yaş etkisini abartıyor (50+ recall 0.94, <30 recall 0.14)
- FN/FP maliyetini **iş ekibiyle resmî olarak doğrula**; varsayılan 5:1 oranı revize edilebilir

**Yeniden değerlendirmeden sonra şu eşiklerin tamamı sağlanırsa GO önerilebilir:**
- Tüm segmentlerde recall_gap ≤ 0.20
- ECE ≤ 0.05
- Demographic parity ratio ≥ 0.6 (Geography ve age_bucket için)

---

**Üretilen artefaktlar:**

| Dosya |
|---|
| [reports/evaluation.json](evaluation.json) — tüm sayısal sonuçlar |
| [reports/figures/confusion_matrix_default.png](figures/confusion_matrix_default.png) |
| [reports/figures/confusion_matrix_tuned.png](figures/confusion_matrix_tuned.png) |
| [reports/figures/threshold_curve.png](figures/threshold_curve.png) |
| [reports/figures/calibration.png](figures/calibration.png) |
| [reports/figures/shap_summary.png](figures/shap_summary.png) |
| [reports/figures/shap_local.png](figures/shap_local.png) |
| [reports/figures/fairness.png](figures/fairness.png) |