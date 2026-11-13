"""
Synthetic two-sided matching data for the MSP470S project (candidates <-> employers).
Fully synthetic: no real people, companies or confidential records.
Run:  python generate_data.py   ->  candidates.csv, employers.csv, interactions.csv, data_dictionary.csv
"""
import numpy as np, pandas as pd

SEED = 470
rng = np.random.default_rng(SEED)
N_CAND, N_EMP, LATENT = 2000, 300, 8
SKILLS = ["python","sql","excel","accounting","marketing","sales","project_mgmt","data_analysis",
          "customer_service","logistics","nursing","teaching","engineering","design","negotiation",
          "writing","compliance","finance_modelling","hr_admin","operations"]
SECTORS = ["Finance","Technology","Healthcare","Retail","Manufacturing","Education","Logistics","Energy"]
REGIONS = ["Western Cape","Gauteng","KwaZulu-Natal","Eastern Cape"]
EDU = ["Matric","Diploma","Degree","Honours","Masters"]
SIZES = ["Small","Medium","Large"]

# ---- candidates
cand_sector = rng.integers(0, len(SECTORS), N_CAND)
cand_skill = np.zeros((N_CAND, len(SKILLS)), dtype=int)
for u in range(N_CAND):                                   # skills cluster around the preferred sector
    base = rng.choice(len(SKILLS), size=rng.integers(3, 8), replace=False)
    cand_skill[u, base] = 1
    cand_skill[u, (cand_sector[u] * 2) % len(SKILLS)] = 1
exp = np.clip(rng.gamma(2.2, 3.0, N_CAND), 0, 35).round(1)
edu_idx = np.clip((exp / 8 + rng.normal(1.2, 1.0, N_CAND)).round().astype(int), 0, 4)
cand = pd.DataFrame({
    "candidate_id": np.arange(1, N_CAND + 1),
    "years_experience": exp,
    "education_level": [EDU[i] for i in edu_idx],
    "sector_preference": [SECTORS[i] for i in cand_sector],
    "region": rng.choice(REGIONS, N_CAND, p=[.35, .35, .2, .1]),
    "expected_salary_zar": (180000 + exp * 22000 + edu_idx * 40000 + rng.normal(0, 40000, N_CAND)).round(-3),
    "skills": ["|".join(s for s, f in zip(SKILLS, row) if f) for row in cand_skill],
})

# ---- employers
emp_sector = rng.integers(0, len(SECTORS), N_EMP)
emp_skill = np.zeros((N_EMP, len(SKILLS)), dtype=int)
for i in range(N_EMP):
    emp_skill[i, rng.choice(len(SKILLS), size=rng.integers(3, 7), replace=False)] = 1
    emp_skill[i, (emp_sector[i] * 2) % len(SKILLS)] = 1
size_idx = rng.choice(3, N_EMP, p=[.45, .35, .2])
emp = pd.DataFrame({
    "employer_id": np.arange(1, N_EMP + 1),
    "sector": [SECTORS[i] for i in emp_sector],
    "region": rng.choice(REGIONS, N_EMP, p=[.3, .45, .15, .1]),
    "size_band": [SIZES[i] for i in size_idx],
    "offered_salary_zar": (250000 + size_idx * 60000 + rng.normal(0, 70000, N_EMP)).round(-3),
    "required_skills": ["|".join(s for s, f in zip(SKILLS, row) if f) for row in emp_skill],
})

# ---- interactions from a hidden preference score (so the true structure is known)
P = rng.normal(0, 1, (N_CAND, LATENT)); Q = rng.normal(0, 1, (N_EMP, LATENT))
pop = rng.pareto(2.0, N_EMP); pop = (pop - pop.mean()) / pop.std() * 0.5      # popularity bias
overlap = (cand_skill @ emp_skill.T) / np.maximum(emp_skill.sum(1), 1)
sector_match = (cand_sector[:, None] == emp_sector[None, :]).astype(float)
region_match = (cand["region"].to_numpy(dtype=object)[:, None] == emp["region"].to_numpy(dtype=object)[None, :]).astype(float)
score = 0.35 * (P @ Q.T) + 2.0 * overlap + 1.0 * sector_match + 0.6 * region_match + pop[None, :]
start, end = pd.Timestamp("2024-11-01"), pd.Timestamp("2026-10-31")
span = (end - start).days
rows = []
for u in range(N_CAND):
    n = int(np.clip(rng.poisson(12), 4, 30))
    p = np.exp(score[u] - score[u].max()); p /= p.sum()
    for i in rng.choice(N_EMP, size=n, replace=False, p=p):
        day = int(rng.integers(0, span))
        hired = rng.random() < 1 / (1 + np.exp(-(score[u, i] - 5.2)))
        shortlisted = hired or rng.random() < 0.35
        rows.append((u + 1, i + 1, (start + pd.Timedelta(days=day)).date(),
                     "hired" if hired else ("shortlisted" if shortlisted else "applied")))
inter = pd.DataFrame(rows, columns=["candidate_id", "employer_id", "event_date", "event_type"])
inter = inter.sort_values("event_date").reset_index(drop=True)
inter.insert(0, "interaction_id", np.arange(1, len(inter) + 1))

# ---- realistic data-quality problems injected on purpose (for the cleaning step)
def pick(df, frac): return rng.choice(df.index, int(len(df) * frac), replace=False)
cand.loc[pick(cand, .04), "expected_salary_zar"] = np.nan
cand.loc[pick(cand, .02), "education_level"] = np.nan
cand.loc[pick(cand, .01), "years_experience"] = rng.choice([-3.0, 61.0, 75.0, 99.0], int(N_CAND * .01))
cand.loc[pick(cand, .01), "expected_salary_zar"] = 0
ix = pick(cand, .03); cand.loc[ix, "region"] = rng.choice(["western cape", "W Cape", "Gauteng ", "KZN"], len(ix))
emp.loc[pick(emp, .05), "offered_salary_zar"] = np.nan
inter.loc[pick(inter, .004), "event_date"] = pd.Timestamp("2027-03-15").date()       # future dates
inter = pd.concat([inter, inter.sample(40, random_state=SEED)], ignore_index=True)     # duplicate rows
inter.loc[pick(inter, .002), "employer_id"] = 9999                                     # unknown employer

cand.to_csv("candidates.csv", index=False); emp.to_csv("employers.csv", index=False); inter.to_csv("interactions.csv", index=False)
dd = pd.DataFrame([
 ("candidates.csv","candidate_id","Unique candidate key"),("candidates.csv","years_experience","Years of work experience (0-35 expected)"),
 ("candidates.csv","education_level","Matric / Diploma / Degree / Honours / Masters"),("candidates.csv","sector_preference","Sector the candidate prefers"),
 ("candidates.csv","region","Province-style region"),("candidates.csv","expected_salary_zar","Expected annual salary, ZAR"),
 ("candidates.csv","skills","Pipe-separated skill list (20 possible skills)"),
 ("employers.csv","employer_id","Unique employer key"),("employers.csv","sector","Employer sector"),("employers.csv","region","Employer region"),
 ("employers.csv","size_band","Small / Medium / Large"),("employers.csv","offered_salary_zar","Typical offered annual salary, ZAR"),
 ("employers.csv","required_skills","Pipe-separated required skills"),
 ("interactions.csv","interaction_id","Row id"),("interactions.csv","candidate_id","FK to candidates"),("interactions.csv","employer_id","FK to employers"),
 ("interactions.csv","event_date","Date of event (data window 2024-11-01 to 2026-10-31)"),
 ("interactions.csv","event_type","applied / shortlisted / hired")], columns=["file","column","description"])
dd.to_csv("data_dictionary.csv", index=False)
print(len(cand), len(emp), len(inter))
