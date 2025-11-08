# BA Thesis: Mitigating Bias in Toxic Comment Classification

**Author:** Philipp Stocker  
**Start date:** 2025-10-01  
**Created scaffold:** 2025-10-18

This repository contains code and materials for a bachelor's thesis on detecting and mitigating bias against gender and LGBTQ identities in toxic comment classification using the Jigsaw *Unintended Bias in Toxicity Classification* dataset.

## Structure

```
BA_Thesis/
├── data/
│   ├── raw/          # Unaltered source data (read-only)
│   └── processed/    # Cleaned/derived datasets for modeling
├── notebooks/        # Exploratory and analysis notebooks
├── src/              # Reusable Python modules
├── reports/          # Interim reports if existing
├── figures/          # Generated plots & diagrams
├── requirements.txt  # Python dependencies (pip) NOTE: not yet implemented
└── .gitignore        # Ignore patterns
```

## Quickstart  (NOTE: for later)

### pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

