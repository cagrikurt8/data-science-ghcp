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