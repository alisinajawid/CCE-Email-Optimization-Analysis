import os
import re
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = BASE / ".cache"
DATA_CANDIDATES = [BASE / "DS_project_data.csv", BASE / "Customer Feedback.csv"]

CACHE.mkdir(exist_ok = True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mp
import matplotlib.gridspec as gs
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from scipy.stats import mannwhitneyu
from imblearn.over_sampling import SMOTE
from kmodes.kmodes import KModes

DATA = next((path for path in DATA_CANDIDATES if path.exists()), None)
if DATA is None:
    expected = ", ".join(path.name for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Could not find a dataset file. Expected one of: {expected}")

OUT  = str(BASE / "output")
Path(OUT).mkdir(exist_ok = True)

PAL  = ["#2D6A4F","#E76F51","#457B9D","#E9C46A","#9B5DE5"]
SEED = 42
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

df = pd.read_csv(DATA)
df.columns = ["ts", "role", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q1_1", "q8_1", "q9", "q10", "q11a", "q11b", "q11c", "q11d", "q11e", "q12"]
df.drop(columns = ["ts", "q1_1", "q8_1", "q12"], inplace = True)

df["q6"] = df["q6"].replace({"Nah, it distracts me": "Nope, it distracts me"})
for c in df.select_dtypes("object"): df[c] = df[c].str.strip()

df["target"]   = (df["q1"] != "None of them").astype(int)
df["q3"]       = df["q3"].map({"From 3 to 5 words": 1, "From 5 to 7 words": 2, "From 7 to 10 words": 3})
df["q5"]       = df["q5"].map({"Less than 50 words": 1, "From 50 to 100 words": 2, "From 101 to 150 words": 3})
df["q7"]       = df["q7"].map({"At the beginning": 1, "In the middle": 2, "At the end": 3, "In the part with links only": 4})
df["role"]     = df["role"].map({"Student": 1, "Staff": 2,"Professor": 3})
df["q4"]       = (df["q4"] == "Yes").astype(int)
df["q6"]       = df["q6"].map({"Yeah, I like this format": 2, "It's alright": 1, "Nope, it distracts me": 0})

LIKERT = {"Strongly Disagree": 1, "Disagree": 2, "Agree": 3, "Strongly Agree": 4}
L_COLS = ["q11a", "q11b", "q11c", "q11d", "q11e"]
L_LBLS = ["Deadlines", "Formatting", "Engaging", "Opportunities", "Satisfied"]
for c in L_COLS: df[c] = df[c].map(LIKERT)

def mlb(series, pfx):
    parsed = series.fillna("").apply(
        lambda x: [i.strip() for i in (re.split(r"\s*;\s*", x) if ";" in x else x.split(","))
                   if i.strip() and "none in particular" not in i.lower() and "opportunities for international" not in i.lower()])
    m = MultiLabelBinarizer()
    return pd.DataFrame(m.fit_transform(parsed), columns = [f"{pfx}_{c[:20]}" for c in m.classes_], index = series.index)

kw_enc, feat_enc = mlb(df.pop("q2"), "kw"), mlb(df.pop("q10"), "feat")
rem_enc          = mlb(df.pop("q9"),  "rem")
df = pd.concat([pd.get_dummies(df, columns = ["q8"], prefix = "q8"), kw_enc, rem_enc, feat_enc], axis = 1)

X = df.drop(columns = ["q1", "target"]).fillna(0).astype(float)
y = df["target"]
print(f"Shape: {X.shape} | Those who read: {y.sum()} | Those who don't: {(y == 0).sum()}")

KM_COLS = ["q3", "q4", "q5", "q6", "q7", "role"] + [c for c in df.columns if c.startswith("q8_")]
df_km   = df[KM_COLS].fillna(0).astype(str)
costs   = {k: KModes(n_clusters = k, init = "Huang", n_init = 10, random_state = SEED, verbose = 0).fit(df_km.values).cost_ for k in range(2, 7)}

K = 3
km = KModes(n_clusters = K, init = "Huang", n_init = 10, random_state = SEED, verbose = 0)
df["cluster"] = km.fit_predict(df_km.values)
profile = df.groupby("cluster")[L_COLS].mean()
profile.columns = L_LBLS
print("Cluster sizes:", df["cluster"].value_counts().sort_index().to_dict())

scaler  = StandardScaler()
Xs      = scaler.fit_transform(X)
Xr, yr  = SMOTE(random_state = SEED, k_neighbors = 2).fit_resample(Xs, y)
lr      = LogisticRegression(max_iter = 1000, C = 0.5, class_weight = "balanced", random_state = SEED)
cv_acc  = cross_val_score(lr, Xr, yr, cv = StratifiedKFold(5, shuffle = True, random_state = SEED), n_jobs = 1)

lr.fit(Xr, yr)
y_prob  = lr.predict_proba(Xs)[:, 1]
auc     = roc_auc_score(y, y_prob)
coef    = pd.Series(lr.coef_[0], index = X.columns)
top_f   = pd.concat([coef.nlargest(6), coef.nsmallest(4)]).sort_values()
print(f"CV Acc: {cv_acc.mean():.3f} +- {cv_acc.std():.3f} | AUC: {auc:.3f}")

threshold = 0.7
y_pred_70 = (y_prob >= threshold).astype(int)
print(classification_report(y, y_pred_70, target_names = ["Non-reader","Reader"], zero_division = 0))

c0, c1 = df[df["cluster"] == 0], df[df["cluster"] == 1]
ab = pd.DataFrame([{
    "Dim": l,
    "C0": round(c0[c].mean(),2), "C1": round(c1[c].mean(),2),
    "p":  round(mannwhitneyu(c0[c].dropna(), c1[c].dropna(), alternative = "two-sided").pvalue, 4),
}
    for c, l in zip(L_COLS, L_LBLS)])
ab["sig"] = ab["p"].apply(lambda p: "Y" if p < 0.05 else "N")
print("\n Mann-Whitney (C0 vs C1):\n", ab.to_string(index = False))

pca  = PCA(n_components = 2, random_state = SEED)
Xpca = pca.fit_transform(Xs)

def hbar(ax, series, color, title, xlabel = "Count", shorten = {}):
    s = series.copy(); s.index = [shorten.get(i, i) for i in s.index]
    ax.barh(s.index[::-1], s.values[::-1], color = color, edgecolor = "white", height = 0.6)
    for i, v in enumerate(s.values[::-1]):
        ax.text(v+0.15, i, str(v), va = "center", fontsize = 8, fontweight = "bold")
    ax.set_xlabel(xlabel); ax.set_xlim(0, s.max()+5); ax.set_title(title, fontweight = "bold")

SHORT = {"With attached files (posters, photos, and etc.)": "Attached files",
         "In the part with links only": "Links only", "Within the email body": "Within body"}
clean = lambda n: (n.replace("_enc", "").replace("kw_", "kw: ").replace("rem_", "rem: ")
                    .replace("feat_", "feat: ").replace("q8_", "format: ")
                    .replace("q11a", "deadlines").replace("q11b", "formatting")
                    .replace("q11c", "engaging").replace("q11d", "opportunities")
                    .replace("q11e", "satisfied").replace("_", " "))

fig = plt.figure(figsize = (18, 13))
G   = gs.GridSpec(3, 3, figure = fig, hspace = 0.55, wspace = 0.42)

ax = fig.add_subplot(G[0,0])
ax.plot(list(costs), list(costs.values()), marker = "o", color = PAL[0], lw = 2)
ax.axvline(K, color = PAL[1], ls = "--", lw = 1.5, label = f"k = {K}")
ax.set(xlabel = "k", ylabel = "Cost", title = "K-Modes Cost Curve"); ax.legend(fontsize = 8, frameon = False)

ax = fig.add_subplot(G[0,1])
for i in range(K):
    m = df["cluster"].values == i
    ax.scatter(Xpca[m,0], Xpca[m,1], color = PAL[i], label = f"C{i} (n = {m.sum()})", s = 65, edgecolors = "white", lw = 0.6, zorder = 3)

ax.set(xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
       ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
       title = "Clusters (PCA)"); ax.legend(fontsize = 8, frameon = False)

ax = fig.add_subplot(G[0,2])
im = ax.imshow(profile.values, cmap = "RdYlGn", vmin = 1, vmax = 4, aspect = "auto")
ax.set(xticks = range(len(L_LBLS)), yticks = range(K),
       xticklabels = [l[:4] for l in L_LBLS], yticklabels = [f"C{i}" for i in range(K)],
       title = "Cluster Satisfaction")

[[ax.text(j, i, f"{profile.values[i,j]:.2f}", ha = "center", va = "center", fontsize = 8, fontweight = "bold")

  for j in range(len(L_LBLS))] for i in range(K)]
plt.colorbar(im, ax = ax, label = "Mean (1 – 4)")

ax = fig.add_subplot(G[1,0:2])
ax.barh([clean(f) for f in top_f.index], top_f.values,
        color = [PAL[0] if v > 0 else PAL[1] for v in top_f.values], edgecolor = "white", height=0.65)
ax.axvline(0, color = "black", lw = 0.8)
ax.legend(handles = [mp.Patch(color = PAL[0], label = "↑ reading"),
                     mp.Patch(color = PAL[1], label = "↓ reading")],
                            fontsize = 8, frameon = False)
ax.set(xlabel = "LR Coefficient", title = "Feature Coefficients (reading behaviour)")

ax = fig.add_subplot(G[1,2])
fpr, tpr, _ = roc_curve(y, y_prob)
ax.plot(fpr, tpr, color = PAL[0], lw = 2, label = f"AUC = {auc:.2f}")
ax.plot([0,1], [0,1], "k--", lw = 1, alpha = 0.4); ax.fill_between(fpr, tpr, alpha = 0.08, color = PAL[0])
ax.set(xlabel = "FPR", ylabel = "TPR", title = "ROC — Logistic Regression"); ax.legend(fontsize = 9, frameon = False)

ax = fig.add_subplot(G[2, 0 : 2])
x, w = np.arange(len(ab)), 0.35
ax.bar(x - w / 2, ab["C0"], w, label = "C0", color = PAL[0], edgecolor = "white")
ax.bar(x + w / 2, ab["C1"], w, label = "C1", color = PAL[2], edgecolor = "white")

[ax.text(i, max(r.C0,r.C1) + 0.1,"*", ha = "center", fontsize = 14, color = PAL[1], fontweight = "bold")

for i,r in enumerate(ab.itertuples()) if r.sig == "Y"]

ax.set(xticks = x, xticklabels = [l[:4] for l in ab["Dim"]], ylim = (1, 4.6),
       ylabel = "Mean (1 – 4)", title = "Mann-Whitney A/B: C0 vs C1  (* p < 0.05)")
ax.legend(fontsize = 9, frameon = False)

ax = fig.add_subplot(G[2,2])
hbar(ax, df["q1"].value_counts().reindex(
    ["All of them", "Most of them", "Sometimes", "Only specific ones", "None of them"]),
    PAL[0], "Q1 — Reading Frequency", "Respondents")

fig.suptitle("CCE Emails Optimization — Analysis Dashboard", fontsize = 14, fontweight = "bold", y = 1.005)
plt.savefig(f"{OUT}/cce_dashboard.png", dpi = 150, bbox_inches = "tight"); plt.close()

fig2, axes = plt.subplots(2, 3, figsize = (16, 9))
fig2.suptitle("Format & Content Preferences", fontsize = 13, fontweight = "bold")

for ax, (col, title) in zip (axes.flat, [
    ("q3", "Q3 — Title length"), ("q4", "Q4 — Emojis"), ("q5", "Q5 — Body length"),
    ("q6", "Q6 — Colored text"), ("q7", "Q7 — Links placement"), ("role", "Respondent roles in uni")]):

    hbar(ax, df[col].value_counts(), PAL[0], title, shorten = SHORT)

plt.tight_layout()
plt.savefig(f"{OUT}/cce_preferences.png", dpi = 150, bbox_inches = "tight"); plt.close()

fig3, (a1, a2) = plt.subplots(1, 2, figsize = (13, 5))
fig3.suptitle("Multi-Select Questions", fontsize = 12, fontweight = "bold")
hbar(a1, kw_enc.sum().sort_values().rename(index = lambda n: n.replace("kw_","")),
     PAL[0], "Q2 — Attractive Keywords", "Mentions")
hbar(a2, feat_enc.sum().sort_values().rename(index = lambda n: n.replace("feat_","")),
     PAL[2], "Q10 — Good Email Features", "Mentions")
plt.tight_layout()
plt.savefig(f"{OUT}/cce_keywords_features.png", dpi = 150, bbox_inches = "tight"); plt.close()

print("\nAll outputs are saved in 'output' folder")
