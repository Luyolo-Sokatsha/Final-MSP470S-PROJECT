**Author:** Luyolo Sokatsha (student number 217148875)
**Module:** MSP470S Mathematical Sciences Project 4, Advanced Diploma in Mathematical Sciences
**Institution:** Cape Peninsula University of Technology (CPUT)
**Date:** November 2026

## What this project is
A comparison of recommender models for ranking employers for each candidate on a talent-matching platform. A heterogeneous GraphSAGE graph neural network (GNN) is compared with a popularity ranking, BPR matrix factorisation and two logistic regression models, using a time-based test, Recall@K and NDCG@K, an ablation, a cold-start test, permutation importance, and a sensitivity analysis across six synthetic worlds.

**All data are synthetic.** No real people, companies or confidential records are used anywhere in this project.

## Run it in Google Colab
No GPU is needed. Run the notebooks in order. Each one starts with a setup cell that can mount your Google Drive, so files are kept in `MyDrive/MSP470S_Project` between sessions.

| Step | Notebook (click to open in Colab) | What it does | Time |
|---|---|---|---|
| 1 | [MSP470S_01_Synthetic_Data_Generation](https://colab.research.google.com/github/Luyolo-Sokatsha/Final-MSP470S-PROJECT/blob/main/MSP470S_01_Synthetic_Data_Generation.ipynb) | Generates the synthetic candidates, employers and interactions (with deliberate data-quality problems) | ~10 s |
| 2 | [MSP470S_02_Data_Cleaning_and_EDA](https://colab.research.google.com/github/Luyolo-Sokatsha/Final-MSP470S-PROJECT/blob/main/MSP470S_02_Data_Cleaning_and_EDA.ipynb) | Situation report, documented cleaning, exploration, time-based and cold-start split | ~30 s |
| 3 | [MSP470S_03_Models_Evaluation_and_Results](https://colab.research.google.com/github/Luyolo-Sokatsha/Final-MSP470S-PROJECT/blob/main/MSP470S_03_Models_Evaluation_and_Results.ipynb) | Popularity, BPR-MF, logistic regression (B, B+), GNN, ablations, bootstrap, importance, model card | ~8 min |
| 4 | [MSP470S_04_Sensitivity_Analysis](https://colab.research.google.com/github/Luyolo-Sokatsha/Final-MSP470S-PROJECT/blob/main/MSP470S_04_Sensitivity_Analysis.ipynb) | Six-world sensitivity analysis (optional; set `QUICK = True` to run one world) | ~25 min |

The notebooks in this repository were run in Google Colab and keep their saved outputs.

## What is in this repository
```
.
├── README.md
├── requirements.txt
├── MSP470S_01_Synthetic_Data_Generation.ipynb
├── MSP470S_02_Data_Cleaning_and_EDA.ipynb
├── MSP470S_03_Models_Evaluation_and_Results.ipynb
├── MSP470S_04_Sensitivity_Analysis.ipynb
├── generate_data.py          synthetic data generator
├── gen_core.py               parameterised generator used by the sensitivity analysis
├── msp470s_models.py         models, tuning, evaluation, figures, model card
├── data.zip                  raw and cleaned CSV files, data dictionary, cleaning log
├── results.zip               main results: tables, figures, trained GNN weights, model card
└── results_sensitivity.zip   results of the six-world sensitivity analysis
```

## Running without notebooks
```
pip install -r requirements.txt
python generate_data.py                 # writes the CSVs to the current folder
# run Notebook 02 (cleaning), then:
MSP_DATA=./data MSP_OUT=./results python msp470s_models.py
```

## Reproducibility
Data generation seed 470; model seeds 0–4 (0–2 in the sensitivity analysis). Data generation is deterministic. Results for popularity, logistic regression and matrix factorisation are identical across software environments tested; GNN results can differ slightly with library versions (a second environment gave NDCG@10 of 0.157 for the GNN against 0.163 in Colab).

## Headline findings (main data, candidates with history, test period)
| Model | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.216 | 0.153 |
| A: BPR matrix factorisation | 0.197 | 0.120 |
| B: logistic regression (profile only) | 0.132 | 0.081 |
| B+: logistic regression + employer popularity | 0.243 | 0.176 |
| C: GNN | 0.227 | 0.163 |
| C: GNN without graph | 0.225 | 0.160 |

The GNN beat matrix factorisation, profile-only logistic regression and, by a small margin, popularity, but not logistic regression with employer popularity (B+). Removing the graph lowered NDCG@10 by only 0.004. In the sensitivity analysis the graph helped when candidates had more history or the collaborative signal was stronger: the GNN was best in three of six worlds and never more than 0.039 NDCG@10 behind the best model, but matrix factorisation beat it when the signal was strongest. For candidates with no history, B+ was best on the main data. See the project report for details.

## Limitations
Synthetic data designed by the author, so results describe how the models behave under that generator, not on real hiring data. The MovieLens sanity check proposed in the Business Case was not carried out. No fairness audit was run. The model is a ranking prototype, not an automated hiring decision tool.
