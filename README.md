**Author:** Luyolo Sokatsha (student number 217148875)
**Module:** MSP470S Mathematical Sciences Project 4, Advanced Diploma in Mathematical Sciences
**Institution:** Cape Peninsula University of Technology (CPUT)
**Date:** November 2026

## What this project is
A comparison of recommender models for ranking employers for each candidate on a talent-matching platform. A heterogeneous GraphSAGE graph neural network (GNN) is compared with a popularity ranking, BPR matrix factorisation and two logistic regression models, using a time-based test, Recall@K and NDCG@K, an ablation, a cold-start test, permutation importance, and a sensitivity analysis across six synthetic worlds.

**All data are synthetic.** No real people, companies or confidential records are used anywhere in this project.

## How to run it (Google Colab)
Open the notebooks in the `notebooks/` folder in Google Colab and run them in order. No GPU is needed. Each notebook starts with a setup cell that can mount your Google Drive, so everything is kept in `MyDrive/MSP470S_Project` between sessions.

| Step | Notebook | What it does | Time |
|---|---|---|---|
| 1 | `MSP470S_01_Synthetic_Data_Generation.ipynb` | Generates the synthetic candidates, employers and interactions (with deliberate data-quality problems) | ~10 s |
| 2 | `MSP470S_02_Data_Cleaning_and_EDA.ipynb` | Situation report, documented cleaning, exploration, time-based and cold-start split | ~30 s |
| 3 | `MSP470S_03_Models_Evaluation_and_Results.ipynb` | Popularity, BPR-MF, logistic regression (B, B+), GNN, ablations, bootstrap, importance, model card | ~8 min |
| 4 | `MSP470S_04_Sensitivity_Analysis.ipynb` | Six-world sensitivity analysis (optional; `QUICK = True` runs one world) | ~25 min |

## Folder structure
```
MSP470S_Project/
├── README.md
├── requirements.txt
├── notebooks/        the four Colab notebooks (run in order)
├── src/              the same code as plain Python files
│   ├── generate_data.py     synthetic data generator
│   ├── gen_core.py          parameterised generator used by the sensitivity analysis
│   └── msp470s_models.py    models, tuning, evaluation, figures, model card
├── data/             raw and cleaned CSV files, data dictionary, cleaning log
├── results/          main results: tables, figures, trained GNN weights, model card
└── results_sensitivity/   results of the six-world sensitivity analysis
```

## Running without notebooks
```
pip install -r requirements.txt
cd src && python generate_data.py                # writes the CSVs to the current folder
# (run Notebook 02 or copy data/*_clean.csv next to the script, then)
MSP_DATA=../data MSP_OUT=../results python msp470s_models.py
```

## Reproducibility
Data generation seed 470; model seeds 0–4 (0–2 in the sensitivity analysis). Small numerical differences are possible on different hardware or library versions. Versions used: Python 3.12, PyTorch 2.15, PyTorch Geometric 2.8, scikit-learn 1.9.

## Headline findings
On the main data the GNN beat matrix factorisation and profile-only logistic regression but was level with a plain popularity ranking and behind logistic regression with employer popularity added (B+). The ablation showed the graph itself added nothing there. In the sensitivity analysis the graph helped when candidates had more history or the collaborative signal was stronger, and the GNN was the most robust model overall (largest shortfall behind the best model 0.035 NDCG@10), but matrix factorisation matched it when the signal was strongest. For candidates with no history, B+ was best in every world. See the project report for details and limitations.

## Limitations
Synthetic data designed by the author, so results describe how the models behave under that generator, not on real hiring data. The MovieLens sanity check proposed in the Business Case was not carried out. No fairness audit was run. The model is a ranking prototype, not an automated hiring decision tool.
