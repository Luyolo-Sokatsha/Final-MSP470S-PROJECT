"""Parameterised version of generate_data.py (clean data, no injected errors) for the sensitivity analysis.
latent_weight : strength of the hidden collaborative-preference signal (default data used 0.35)
mean_inter    : average number of interactions per candidate (default data used 12)"""
import numpy as np, pandas as pd

SKILLS = ["python","sql","excel","accounting","marketing","sales","project_mgmt","data_analysis","customer_service","logistics",
          "nursing","teaching","engineering","design","negotiation","writing","compliance","finance_modelling","hr_admin","operations"]
SECTORS = ["Finance","Technology","Healthcare","Retail","Manufacturing","Education","Logistics","Energy"]
REGIONS = ["Western Cape","Gauteng","KwaZulu-Natal","Eastern Cape"]
EDU = ["Matric","Diploma","Degree","Honours","Masters"]; SIZES = ["Small","Medium","Large"]

def make_data(latent_weight=0.35, mean_inter=12, seed=470, n_cand=2000, n_emp=300, latent=8):
    rng = np.random.default_rng(seed)
    cs = rng.integers(0, len(SECTORS), n_cand); ck = np.zeros((n_cand, len(SKILLS)), int)
    for u in range(n_cand):
        ck[u, rng.choice(len(SKILLS), size=rng.integers(3, 8), replace=False)] = 1; ck[u, (cs[u] * 2) % len(SKILLS)] = 1
    exp = np.clip(rng.gamma(2.2, 3.0, n_cand), 0, 35).round(1)
    edu = np.clip((exp / 8 + rng.normal(1.2, 1.0, n_cand)).round().astype(int), 0, 4)
    cand = pd.DataFrame({"candidate_id": np.arange(1, n_cand + 1), "years_experience": exp, "education_level": [EDU[i] for i in edu],
        "sector_preference": [SECTORS[i] for i in cs], "region": rng.choice(REGIONS, n_cand, p=[.35, .35, .2, .1]),
        "expected_salary_zar": (180000 + exp * 22000 + edu * 40000 + rng.normal(0, 40000, n_cand)).round(-3),
        "skills": ["|".join(s for s, f in zip(SKILLS, r) if f) for r in ck]})
    es = rng.integers(0, len(SECTORS), n_emp); ek = np.zeros((n_emp, len(SKILLS)), int)
    for i in range(n_emp):
        ek[i, rng.choice(len(SKILLS), size=rng.integers(3, 7), replace=False)] = 1; ek[i, (es[i] * 2) % len(SKILLS)] = 1
    sz = rng.choice(3, n_emp, p=[.45, .35, .2])
    emp = pd.DataFrame({"employer_id": np.arange(1, n_emp + 1), "sector": [SECTORS[i] for i in es],
        "region": rng.choice(REGIONS, n_emp, p=[.3, .45, .15, .1]), "size_band": [SIZES[i] for i in sz],
        "offered_salary_zar": (250000 + sz * 60000 + rng.normal(0, 70000, n_emp)).round(-3),
        "required_skills": ["|".join(s for s, f in zip(SKILLS, r) if f) for r in ek]})
    P = rng.normal(0, 1, (n_cand, latent)); Q = rng.normal(0, 1, (n_emp, latent))
    pop = rng.pareto(2.0, n_emp); pop = (pop - pop.mean()) / pop.std() * 0.5
    overlap = (ck @ ek.T) / np.maximum(ek.sum(1), 1)
    sm = (cs[:, None] == es[None, :]).astype(float)
    rm = (cand["region"].to_numpy(dtype=object)[:, None] == emp["region"].to_numpy(dtype=object)[None, :]).astype(float)
    score = latent_weight * (P @ Q.T) + 2.0 * overlap + 1.0 * sm + 0.6 * rm + pop[None, :]
    start, end = pd.Timestamp("2024-11-01"), pd.Timestamp("2026-10-31"); span = (end - start).days; rows = []
    for u in range(n_cand):
        n = int(np.clip(rng.poisson(mean_inter), 4, 60)); p = np.exp(score[u] - score[u].max()); p /= p.sum()
        for i in rng.choice(n_emp, size=n, replace=False, p=p):
            day = int(rng.integers(0, span)); hired = rng.random() < 1 / (1 + np.exp(-(score[u, i] - 5.2)))
            sl = hired or rng.random() < 0.35
            rows.append((u + 1, i + 1, (start + pd.Timedelta(days=day)).date(), "hired" if hired else ("shortlisted" if sl else "applied")))
    inter = pd.DataFrame(rows, columns=["candidate_id", "employer_id", "event_date", "event_type"]).sort_values("event_date").reset_index(drop=True)
    inter.insert(0, "interaction_id", np.arange(1, len(inter) + 1))
    return cand, emp, inter
