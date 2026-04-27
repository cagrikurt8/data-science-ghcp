# Proje Bağlamı
Bu repo, bankacılık müşteri davranışı üzerine ikili sınıflandırma modelleri geliştirir.
- Dil: Python 3.12, Pandas, scikit-learn, XGBoost, MLflow
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