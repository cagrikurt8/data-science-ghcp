# GitHub Copilot ile Uçtan Uca Veri Bilimi Workshop'u

> **Süre:** 3 saat (180 dk)
> **Hedef Kitle:** Veri bilimciler, ML mühendisleri, analitik ekipleri
> **Ön Koşul:** VS Code + GitHub Copilot (Chat & Agent Mode), Python 3.11+, temel ML bilgisi
> **Amaç:** GitHub Copilot'un **Custom Agents**, **Custom Instructions**, **Prompt Files** ve **MCP** özelliklerini kullanarak — Data Analyzer, Feature Engineer, Model Trainer, Model Evaluator, Deployment Advisor gibi **custom agent**'lar ile uçtan uca bir veri bilimi yaşam döngüsünü kurgulamak.

---

## 🎯 Workshop Sonunda Katılımcılar Şunları Yapabilecek

- GitHub Copilot'u **veri bilimi iş akışına** özelleştirme (workspace-level instructions, prompt files, custom agents).
- **Rol bazlı custom agent** tanımlama (`.github/agents/*.agent.md`).
- Bir veri setini yükleyip **Data Analyzer agent** ile keşifsel analiz (EDA) yapma.
- **Feature Engineer agent** ile özellik mühendisliği pipeline'ı üretme.
- **Model Trainer agent** ile model eğitimi ve hiperparametre denemeleri.
- **Model Evaluator agent** ile metrik, fairness ve hata analizi raporu çıkarma.
- **Deployment Advisor agent** ile modelin Azure ML / container olarak yayına alınma planı oluşturma.
- MCP server'ları (ör. Azure MCP, filesystem, sqlite) Copilot'a bağlayıp canlı veriyle konuşma.

---

## 📅 Ajanda (180 dk)

| Blok | Süre | Konu |
|------|------|------|
| 0 | 10 dk | Giriş + Ortam kontrolü |
| 1 | 25 dk | GitHub Copilot'u Veri Bilimi için Özelleştirme |
| 2 | 25 dk | Custom Agent Mimarisi (Custom Agents + Prompt Files) |
| 3 | 25 dk | **Lab 1 — Data Analyzer Agent** (EDA) |
| 4 | 10 dk | ☕ Mola |
| 5 | 25 dk | **Lab 2 — Feature Engineer Agent** |
| 6 | 25 dk | **Lab 3 — Model Trainer Agent** |
| 7 | 20 dk | **Lab 4 — Model Evaluator Agent** |
| 8 | 10 dk | **Lab 5 — Deployment Advisor Agent** |
| 9 | 5 dk  | Kapanış + Q&A |

---

## 🧰 Ortam Hazırlığı (Workshop öncesi)

```powershell
# 1) Repo klonla
git clone <repo-url> data-science-gh
cd data-science-gh

# 2) uv kurulumu (yoksa)
#   Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS / Linux
#   curl -LsSf https://astral.sh/uv/install.sh | sh

# 3) Python sürümünü sabitle ve sanal ortam oluştur
uv python pin 3.11
uv venv
.\.venv\Scripts\Activate.ps1

# 4) Bağımlılıkları ekle (pyproject.toml + uv.lock üretir)
uv add pandas numpy scikit-learn matplotlib seaborn jupyter `
       xgboost lightgbm shap mlflow optuna fastapi uvicorn joblib
uv add --dev ruff pytest

# 5) Ortamı senkronize et (lock'tan kurulum)
uv sync

# 6) VS Code eklentileri
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension charliermarsh.ruff
```

> 💡 Komutları aktivasyon yapmadan çalıştırmak için `uv run <komut>` (ör. `uv run jupyter lab`, `uv run pytest`) kullanabilirsiniz.

**Kullanılacak veri seti:** UCI *Bank Marketing* veya *Credit Default* (bankacılık temalı, ikili sınıflandırma). `data/raw/bank.csv` altına konur.

---

## 🧩 Referans Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                    VS Code + Copilot Chat                   │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │DataAnalyzer │→ │FeatureEng.  │→ │ModelTrainer │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         ↓                                 ↓                 │
│  ┌─────────────┐                   ┌─────────────┐          │
│  │ModelEvaluator│ ←────────────────│DeploymentAdv│          │
│  └─────────────┘                   └─────────────┘          │
│                                                             │
│  Ortak katman: .github/copilot-instructions.md              │
│                .github/prompts/*.prompt.md                  │
│                .github/agents/*.agent.md                    │
│                MCP: filesystem, sqlite, azure               │
└─────────────────────────────────────────────────────────────┘
          ↓                ↓                ↓
     data/raw/       notebooks/        models/
     data/processed/                   reports/
```

---

## 🔧 Blok 1 — Copilot'u Veri Bilimi için Özelleştirme (25 dk)

### 1.1 Workspace Custom Instructions

`.github/copilot-instructions.md` dosyası — her sohbete otomatik eklenir.

```markdown
# Proje Bağlamı
Bu repo, bankacılık müşteri davranışı üzerine ikili sınıflandırma modelleri geliştirir.
- Dil: Python 3.11, Pandas, scikit-learn, XGBoost, MLflow
- Paket yöneticisi: **uv** — bağımlılık ekleme `uv add <paket>`, çalıştırma `uv run <komut>`
- Bağımlılıklar `pyproject.toml`'da; `uv.lock` commit'lenir; `pip install` / `requirements.txt` KULLANMA
- Kod stili: PEP8, `ruff` ile lint, tip ipuçları zorunlu
- Veri: PII içerir → örnek/maskelenmiş veri dışında log'a yazma
- Notebooklar: `notebooks/` altında, numaralı (`01_eda.ipynb`, `02_features.ipynb` ...)
- Deneyler MLflow ile loglanır, metrikler `reports/metrics.json`'a yazılır

# Davranış Kuralları
- Kod üretirken reproducibility için `random_state=42` kullan
- Veri sızıntısına karşı train/val/test ayrımını her zaman ilk adımda yap
- Grafikler matplotlib/seaborn, başlık + eksen etiketi zorunlu
- Açıklamalarda Türkçe, kod içinde İngilizce isimlendirme
- Terminal komutlarını `uv run ...` ile öner (ör. `uv run pytest`, `uv run jupyter lab`)
```

### 1.2 Path-spesifik Instructions

`.github/instructions/notebooks.instructions.md`:

```markdown
---
applyTo: "notebooks/**/*.ipynb"
---
- Her notebook başında amaç, girdi, çıktı bölümü yaz
- Her hücre tek sorumluluk taşısın
- Son hücrede üretilen artefaktı `data/processed/` veya `models/` altına kaydet
```

### 1.3 Prompt Files (tekrar eden görevler için)

`.github/prompts/eda-summary.prompt.md`:

```markdown
---
mode: agent
description: Verilen DataFrame için standart EDA özet raporu üretir
---
Verilen pandas DataFrame için şunları yap:
1. `df.info()`, `df.describe(include="all")`
2. Eksik değer yüzdeleri (sütun bazlı, > %5 olanları işaretle)
3. Sayısal kolonlar için histogram + boxplot
4. Kategorik kolonlar için countplot (top 10)
5. Hedef değişkene göre korelasyon/anova özeti
6. Bulguları `reports/eda.md` dosyasına markdown tablo olarak yaz
```

> 💡 **Demo:** `/eda-summary` yazınca prompt file tetiklenir.

---

## 🤖 Blok 2 — Custom Agent Mimarisi (25 dk)

GitHub Copilot'ta **Custom Agent** = rol odaklı, kısıtlı araç setli, özel system prompt'lu bir "agent persona".

Dosya: `.github/agents/<name>.agent.md`

```markdown
---
description: "Kısa açıklama — chat picker'da görünür"
tools: ["codebase", "editFiles", "runCommands", "runNotebooks", "search"]
model: "GPT-5"
---
# System Prompt buraya yazılır
Sen ...... bir agentsın. Yalnızca ..... yap.
```

Workshop'ta 5 custom agent tanımlayacağız:

| Agent | Rol | Önemli Araçlar |
|-------|-----|----------------|
| `DataAnalyzer` | EDA, veri kalitesi, profil raporu | `runNotebooks`, `editFiles`, `search` |
| `FeatureEngineer` | Feature pipeline, encoding, scaling | `editFiles`, `runCommands` |
| `ModelTrainer` | Model seçimi, tuning, MLflow log | `runCommands`, `runNotebooks` |
| `ModelEvaluator` | Metrik, confusion, SHAP, fairness | `runNotebooks`, `editFiles` |
| `DeploymentAdvisor` | Paketleme, Azure ML / container plan | `editFiles`, `runCommands`, MCP `azure` |

---

## 🧪 Lab 1 — Data Analyzer Agent (25 dk)

**Dosya:** `.github/agents/DataAnalyzer.agent.md`

```markdown
---
description: "Keşifsel veri analizi ve veri kalitesi uzmanı"
tools: ["codebase", "editFiles", "runNotebooks", "search", "runCommands"]
model: "Claude Opus 4.7"
---
Sen bir **Data Analyzer** agent'ısın. Görevin:

1. Verilen veri setini yükle, şema ve tipleri doğrula
2. Veri kalitesi raporu üret: eksik, duplicate, outlier, cardinality
3. Hedef değişkene (varsa) göre dağılım ve sınıf dengesizliği analizi
4. EDA notebook'unu `notebooks/01_eda.ipynb` olarak oluştur/güncelle
5. Bulguları `reports/data_quality.md`'ye yaz; risk/uyarıları **Kırmızı/Sarı/Yeşil** etiketle

Kısıtlar:
- Asla modelleme YAPMA — sadece analiz
- PII sütunlarını (`ad`, `tckn`, `iban`, `telefon`) otomatik maskele
- Her grafiği `reports/figures/` altına PNG olarak kaydet
```

### Lab 1 — Akış

1. Agent picker'dan **DataAnalyzer**'ı seç.
2. İstek: *"`data/raw/bank.csv`'yi analiz et, hedef kolon `y`."*
3. Agent notebook'u oluşturur, hücreleri çalıştırır, raporu yazar.
4. Katılımcılar çıktı üzerinden **sınıf dengesizliği, eksik değer, outlier** tartışır.

**Tartışma Soruları**
- Agent hangi varsayımları otomatik yaptı?
- Prompt'a hangi kısıtı ekleseydik daha iyi sonuç alırdık?

---

## 🧪 Lab 2 — Feature Engineer Agent (25 dk)

**Dosya:** `.github/agents/FeatureEngineer.agent.md`

```markdown
---
description: "Özellik mühendisliği ve preprocessing pipeline uzmanı"
tools: ["codebase", "editFiles", "runCommands"]
model: "Claude Opus 4.7"
---
Sen bir **Feature Engineer** agent'ısın. Data Analyzer çıktısını baz alarak:

1. `src/features/pipeline.py` dosyasında `sklearn.pipeline.Pipeline` + `ColumnTransformer` üret
2. Sayısal: median impute + StandardScaler; Kategorik: mode impute + OneHot/Target encoding
3. Train/val/test split (stratified, `random_state=42`), sızıntı olmayacak şekilde `fit` sadece train'de
4. `data/processed/{train,val,test}.parquet` kaydet
5. Pipeline'ı `models/preprocessor.joblib` olarak serileştir
6. Feature listesini `reports/features.md`'ye yaz (tür, dönüşüm, gerekçe)

Kısıtlar:
- Hedef sızıntısına yol açabilecek feature üretme
- Yeni bir feature eklerken Data Analyzer raporuna referans ver
```

### Lab 2 — Akış
1. `@FeatureEngineer "01_eda.ipynb çıktılarına göre preprocessing pipeline'ı üret"`
2. Katılımcılar kodu inceler, agent'a feature eklettirir (ör. `age_bucket`, `balance_log`).
3. Pipeline'ın **idempotent** çalıştığı test edilir.

---

## 🧪 Lab 3 — Model Trainer Agent (25 dk)

**Dosya:** `.github/agents/ModelTrainer.agent.md`

```markdown
---
description: "Model eğitimi, hiperparametre arama, MLflow loglama"
tools: ["codebase", "editFiles", "runCommands", "runNotebooks"]
model: "Claude Opus 4.7"
---
Sen bir **Model Trainer** agent'ısın.

1. Baseline: LogisticRegression + hazır preprocessor
2. Güçlü model: XGBoost veya LightGBM
3. MLflow experiment adı: `bank-churn`; her run'da param+metrik+model logla
4. HPO: Optuna ile 20 trial, hedef = PR-AUC (dengesiz sınıf)
5. En iyi modeli `models/best_model.joblib` olarak kaydet
6. `notebooks/03_training.ipynb` özet notebook'u güncelle

Kısıtlar:
- Test set'e eğitim sürecinde ASLA dokunma
- Early stopping için val set kullan
- Her çalıştırmada `random_state=42`
```

### Lab 3 — Akış
1. `@ModelTrainer "baseline + xgboost eğit, mlflow'a logla"`
2. `mlflow ui` ile karşılaştırma.
3. Agent'a: *"PR-AUC'u %3 artırmak için ne önerirsin?"* → prompt chaining.

---

## 🧪 Lab 4 — Model Evaluator Agent (20 dk)

**Dosya:** `.github/agents/ModelEvaluator.agent.md`

```markdown
---
description: "Metrik, hata analizi, açıklanabilirlik, fairness"
tools: ["codebase", "editFiles", "runNotebooks"]
model: "Claude Opus 4.7"
---
Sen bir **Model Evaluator** agent'ısın. `models/best_model.joblib` ve `data/processed/test.parquet` üzerinde:

1. Metrikler: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier
2. Confusion matrix + threshold tuning (cost-sensitive)
3. Kalibrasyon eğrisi
4. SHAP global + yerel açıklama (top 10 feature)
5. Fairness: `job`, `age_bucket` segmentlerinde metrik farkı
6. Bulguları `reports/evaluation.md` ve `reports/figures/`'a yaz
7. Üretime uygun mu? **Go / No-Go** önerisi ver ve gerekçelendir

Kısıtlar:
- Sadece test set'i kullan
- Modeli yeniden eğitme
```

### Lab 4 — Akış
1. `@ModelEvaluator "best_model'i değerlendir"`
2. Katılımcılar **Go/No-Go** kararını tartışır.
3. Agent'a threshold'u iş maliyetine göre ayarlatırın.

---

## 🧪 Lab 5 — Deployment Advisor Agent (10 dk)

**Dosya:** `.github/agents/DeploymentAdvisor.agent.md`

```markdown
---
description: "Model paketleme ve Azure ML / Container dağıtım uzmanı"
tools: ["codebase", "editFiles", "runCommands"]
model: "Claude Opus 4.7"
---
Sen bir **Deployment Advisor** agent'ısın.

1. `src/serve/app.py` — FastAPI inference endpoint (`/predict`, `/health`)
2. `Dockerfile` (python:3.11-slim, non-root user)
3. `scoring/score.py` — Azure ML online endpoint entry script
4. `deploy/azureml/endpoint.yml` + `deployment.yml` taslağı
5. `reports/deployment_plan.md`: kaynak boyutu, SLA, monitoring, rollback

Kısıtlar:
- Gerçek credential YAZMA; `${ENV_VAR}` placeholder kullan
- Data drift + model drift monitoring önerisini dahil et
```

### Lab 5 — Akış
1. `@DeploymentAdvisor "best_model için Azure ML online endpoint planı hazırla"`
2. Dockerfile + endpoint YAML üretilir.
3. Opsiyonel: Azure MCP ile gerçek kaynak önerileri.

---

## 🔌 Opsiyonel — MCP Entegrasyonları

`.vscode/mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}/data"]
    },
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "${workspaceFolder}/data/bank.db"]
    },
    "azure": {
      "command": "npx",
      "args": ["-y", "@azure/mcp@latest", "server", "start"]
    }
  }
}
```

Kullanım: Data Analyzer → `sqlite` ile canlı sorgu; Deployment Advisor → `azure` ile subscription/ML workspace önerisi.

---

## 📂 Önerilen Repo Yapısı

```
data-science-gh/
├── .github/
│   ├── copilot-instructions.md
│   ├── instructions/
│   │   └── notebooks.instructions.md
│   ├── prompts/
│   │   ├── eda-summary.prompt.md
│   │   ├── hpo-suggest.prompt.md
│   │   └── model-card.prompt.md
│   └── agents/
│       ├── DataAnalyzer.agent.md
│       ├── FeatureEngineer.agent.md
│       ├── ModelTrainer.agent.md
│       ├── ModelEvaluator.agent.md
│       └── DeploymentAdvisor.agent.md
├── .vscode/
│   └── mcp.json
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── features/pipeline.py
│   ├── train/train.py
│   └── serve/app.py
├── models/
├── reports/
│   ├── figures/
│   ├── data_quality.md
│   ├── eda.md
│   ├── features.md
│   ├── evaluation.md
│   └── deployment_plan.md
├── deploy/azureml/
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

---

## ✅ Başarı Kriterleri (Katılımcı Checklist)

- [ ] `.github/agents/` altında en az 3 çalışan custom agent
- [ ] `reports/` altında data_quality + evaluation markdown raporları
- [ ] MLflow'da en az 3 karşılaştırılabilir run
- [ ] `models/best_model.joblib` + `models/preprocessor.joblib`
- [ ] `deployment_plan.md` — Go/No-Go gerekçeli
- [ ] `.github/copilot-instructions.md` projenin konvansiyonlarını yansıtıyor

---

## 📚 Kaynaklar

- [Customize Copilot Chat (custom agents, prompt files, instructions)](https://code.visualstudio.com/docs/copilot/copilot-customization)
- [GitHub Copilot Agent Mode](https://code.visualstudio.com/docs/copilot/chat/chat-agent-mode)
- [Model Context Protocol (MCP) in VS Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
