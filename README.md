# BIONEXUS-QA — Domain-Adaptive Pretraining (DAPT) + SFT Pipeline

> Production-oriented repo for biomedical instruction tuning.  
> We first run **DAPT** (unsupervised, causal LM) on in-domain PubMedQA-style text, then perform **SFT** on curated instruction data.

---

##  What’s here

- **Data pipeline** to split and prepare PubMedQA family datasets.
- **Config-driven PyTorch DDP** trainer for **LLaMA-3-8B** DAPT (no CLI args).
- Clean project hygiene: `.gitignore`, tracked configs, ignored big artifacts.
- GCP single-node multi-GPU launch instructions.

---

##  Repo layout

