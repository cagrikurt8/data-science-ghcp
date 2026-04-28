# Deployment Plan — bank-churn (CONDITIONAL_GO, v2-2026-04-28)

**Hedef:** XGBoost + isotonic calibration modelini Azure ML Managed Online Endpoint olarak yayina almak.
**Karar:** [reports/evaluation.md](evaluation.md) — `CONDITIONAL_GO` (gate'te `<30` segmenti business rule ile waived)
**Policy:** [models/decision_policy.json](../models/decision_policy.json) (v2-2026-04-28)
**Inference helper:** [src/serve/policy.py](../src/serve/policy.py)

> Bu doküman gercek kimlik bilgisi icermez; tum hassas degerler `${ENV_VAR}` placeholder.

---

## 1. Bilesenler

| Bilesen | Konum |
|---|---|
| FastAPI uygulamasi (`/predict`, `/health`) | [src/serve/app.py](../src/serve/app.py) |
| Karar policy modulu | [src/serve/policy.py](../src/serve/policy.py) |
| Container imaji | [Dockerfile](../Dockerfile) (python:3.11-slim, non-root) |
| AzureML scoring entry | [scoring/score.py](../scoring/score.py) |
| AzureML endpoint manifest | [deploy/azureml/endpoint.yml](../deploy/azureml/endpoint.yml) |
| AzureML deployment manifest | [deploy/azureml/deployment.yml](../deploy/azureml/deployment.yml) |
| Conda environment | [deploy/conda.yml](../deploy/conda.yml) |
| Model artefakti | [models/best_model.joblib](../models/best_model.joblib) |
| Preprocessor | [models/preprocessor.joblib](../models/preprocessor.joblib) |
| Decision policy | [models/decision_policy.json](../models/decision_policy.json) |

## 2. Endpoint Sozlesmesi

`POST /predict` (FastAPI) ve AzureML `score()` ayni JSON sozlesmesini paylasir:

```json
{
  "customers": [
    {
      "CustomerId": 15634602,
      "CreditScore": 619, "Geography": "France", "Gender": "Female",
      "Age": 42, "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1,
      "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88
    }
  ]
}
```

Yanit:
```json
{
  "model_version": "v2-2026-04-28",
  "decision": "CONDITIONAL_GO",
  "predictions": [
    {"customer_id": 15634602, "score": 0.37, "risk_band": "medium",
     "action": "crm_priority_list", "reason": "..."}
  ]
}
```

Pydantic validasyonu (Geography enum, age 18-120, vb.) sayesinde kotu istek 422 doner.

## 3. Kaynak Boyutlandirma (Standard_DS3_v2: 4 vCPU, 14 GiB)

| Olcum | Deger | Kaynak |
|---|---|---|
| Model + preprocessor disk | ~150 KB toplam | `ls -lh models/` |
| Tek istekte bellek (1 musteri) | < 50 MB (XGBoost overhead) | local profiling |
| 100 musterilik batch p99 latency (local) | < 25 ms | smoke test |
| Hedef p95 (AML) | < 150 ms | SLA |
| Hedef RPS (per instance, batch=1) | ~150 | conservative; XGBoost predict ~3 ms |

> Standard_DS3_v2 baslangic icin yeterli. CPU bound; GPU gereksiz. RAM headroom bol.

## 4. Olcekleme

`deployment.yml` icinde `target_utilization` autoscale:
- min instances: 2 (HA)
- max instances: 6
- hedef CPU: %70
- polling: 60 s

Trafik tahmini: gunluk ~5M scoring (CRM batch + realtime). Realtime: ~58 RPS ortalama.

## 5. SLA / SLO

| Metric | Hedef |
|---|---|
| Availability | 99.5% (managed online endpoint native) |
| Latency p95 | < 150 ms |
| Latency p99 | < 300 ms |
| Error rate (5xx) | < 0.5% |
| Cold start | < 30 s (init probe period) |

Probelar: liveness 30 s, readiness 10 s. `/health` artefakt yuklendi mi kontrolu.

## 6. Guvenlik (OWASP)

| Risk | Onlem |
|---|---|
| A01 Broken Access Control | AML `auth_mode: key`; key rotation 90 gunde. Tercih: Entra ID + AAD. |
| A02 Crypto Failures | TLS endpoint default; key vault'ta saklanir |
| A03 Injection | Pydantic strict input; pandas DataFrame whitelist kolonlar |
| A04 Insecure Design | Karar policy ayri JSON; degisiklik PR ile audit edilir |
| A05 Security Misconfig | Non-root user, `egress_public_network_access: disabled` |
| A06 Vulnerable Components | `uv.lock` pinned; aylik `uv lock --upgrade` + tarama (Trivy) |
| A07 Auth Failures | Endpoint key + mTLS opsiyonu |
| A08 Software/Data Integrity | Model imza: AML registered model immutable; SHA tag log |
| A09 Logging | App Insights + AML log streaming; PII loglanmaz (Surname istek alanı yok) |
| A10 SSRF | Container egress kapali |

> PII notu: `Surname` API sozlesmesinde **kabul edilmiyor**; `CustomerId` ise opsiyonel ve hash'lenmemis dis sistem id'si olarak gecer (banka ic kimliği).

## 7. Drift Monitoring

### Data drift
- AML `data_collector` ile request/response toplanir (rolling_rate=hour).
- Haftalik PSI (Population Stability Index) hesaplanir: `Age`, `Balance`, `NumOfProducts`, `Geography`, `Gender`.
- Esik: PSI > 0.20 alarm; PSI > 0.25 retrain trigger.
- Job: Azure ML pipeline `monitor_data_drift.yml` (her Pazartesi 06:00 UTC).

### Model drift / performance drift
- Gercek etiket gelene kadar (60-90 gun gecikme) **proxy izleme**:
  - selection_rate per segment (dpr) gunluk
  - skor dagilimi histogrami (KS test vs son 30 gun)
- Etiket eldesi sonrasi: PR-AUC, recall, ECE rolling 30 gun.
- Esik: PR-AUC dusus > %5 kalibrasyon; > %10 retrain.

### Fairness drift
- Gunluk: `<30` segmentinde `manual_review` orani; > %30 olur ise iş ekibine alarm (kural fazla tetikleniyor).
- Aylik: `evaluation.json` segmentleri yeniden uretilir, `recall_gap` izlenir.

## 8. Rollback Plani

| Olay | Aksiyon | Sure |
|---|---|---|
| Latency p95 > 500 ms 10 dk | Auto-scale failure -> `green` deployment'a traffic switch | 5 dk |
| Error rate > %5 5 dk | `az ml online-endpoint update --traffic blue=0 green=100` | 2 dk |
| Modelde regresyon (PR-AUC dusus > %10) | Onceki versiyon (`v1-...` registered model) deploy | 15 dk |
| Policy hatasi (manual_review patlamasi) | `decision_policy.json` rollback (storage versionlu) | 5 dk |

Blue/green stratejisi: yeni deployment `green` ile %10 trafikle baslar, 24 saat soak; sonra %50, sonra %100.

## 9. Deployment Akisi (Az CLI)

```pwsh
# Env vars (KeyVault / pipeline secrets)
$env:ENV_NAME = "${ENV_NAME}"          # dev | staging | prod
$env:MODEL_VERSION = "${MODEL_VERSION}"
$env:ENV_VERSION = "${ENV_VERSION}"
$env:REGISTRY_NAME = "${REGISTRY_NAME}"
$env:TEAM_OWNER_EMAIL = "${TEAM_OWNER_EMAIL}"
$env:WORKSPACE = "${AML_WORKSPACE}"
$env:RESOURCE_GROUP = "${AML_RESOURCE_GROUP}"

# 1) Modeli kaydet
az ml model create --name bank-churn-model --version $env:MODEL_VERSION `
    --path models --type custom_model `
    --workspace-name $env:WORKSPACE --resource-group $env:RESOURCE_GROUP

# 2) Endpoint
az ml online-endpoint create -f deploy/azureml/endpoint.yml `
    --workspace-name $env:WORKSPACE --resource-group $env:RESOURCE_GROUP

# 3) Deployment (blue)
az ml online-deployment create -f deploy/azureml/deployment.yml --all-traffic `
    --workspace-name $env:WORKSPACE --resource-group $env:RESOURCE_GROUP
```

Local Docker icin:
```pwsh
docker build -t bank-churn-serve:v2 .
docker run --rm -p 8080:8080 bank-churn-serve:v2
curl http://localhost:8080/health
```

## 10. Kabul Kriterleri (UAT)

- [ ] `/health` 200; `policy_version=v2-2026-04-28`
- [ ] 1.000 ornekli batch < 2 sn p95
- [ ] `<30` musteri yuksek skor + Geography=Germany testinde `manual_review` doner
- [ ] Pydantic ile gecersiz Geography 422 doner
- [ ] App Insights'a request logu + custom dimension (`risk_band`, `policy_version`) gider
- [ ] Aylik retraining pipeline (`reports/metrics.json` regen) PR-AUC esiklerini sagliyor

## 11. Re-approval

[decision_policy.json](../models/decision_policy.json) icinde `reapproval_required_in_days: 180`. Bu sureden once retrain + reevaluation calistirilarak `evaluation.json` yeniden uretilmeli.