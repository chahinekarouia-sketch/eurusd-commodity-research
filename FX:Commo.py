# =============================================================================
# EUR/USD & Commodities — Correlation Analysis
# BIAT Internship Project | 2018–2026
# =============================================================================

# ── Libraries ─────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns
from scipy import stats
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.expanduser("~/Desktop/FX_Commo_Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(filename):
    plt.savefig(os.path.join(OUTPUT_DIR, filename))


# ── 1. DATA DOWNLOAD ──────────────────────────────────────────────────────────
tickers = {
    "EURUSD":       "EURUSD=X",
    "Gold":         "GC=F",
    "Silver":       "SI=F",
    "Copper":       "HG=F",
    "Platinum":     "PL=F",
    "Palladium":    "PA=F",
    "Brent":        "BZ=F",
    "WTI":          "CL=F",
    "Heating_Oil":  "HO=F",
    "Gasoline":     "RB=F",
    "Gas":          "NG=F",
    "Gas_EU":       "TTF=F",
    "Wheat":        "ZW=F",
    "Corn":         "ZC=F",
    "Soybeans":     "ZS=F",
    "Soybean_Oil":  "ZL=F",
    "Soybean_meal": "ZM=F",
    "Coffee":       "KC=F",
    "Sugar":        "SB=F",
    "Cocoa":        "CC=F",
    "Cotton":       "CT=F",
}

close_prices = {}
for name, ticker in tickers.items():
    data = yf.download(ticker, start="2016-01-01", end="2026-06-01")
    close_prices[name] = data["Close"].squeeze()


# ── 2. DAILY RETURNS ──────────────────────────────────────────────────────────
table_1 = pd.DataFrame(close_prices)
daily_returns = table_1.dropna().pct_change().dropna()

commodity_columns = [col for col in daily_returns.columns if col != "EURUSD"]

# Category groupings
metals_commo         = ["Gold", "Silver", "Copper", "Platinum", "Palladium"]
energy_commo         = ["Brent", "WTI", "Heating_Oil", "Gasoline", "Gas", "Gas_EU"]
agricultural_commo   = ["Wheat", "Corn", "Soybeans", "Soybean_Oil", "Soybean_meal",
                        "Coffee", "Sugar", "Cocoa", "Cotton"]
agricultural_reduced = ["Wheat", "Coffee", "Cocoa"]

# 60-day rolling averages for visualisation (×100 → percentage)
metals_rolling  = daily_returns[metals_commo].mean(axis=1).rolling(60).mean().dropna() * 100
energy_rolling  = daily_returns[energy_commo].mean(axis=1).rolling(60).mean().dropna() * 100
agri_rolling    = daily_returns[agricultural_reduced].rolling(60).mean().dropna() * 100
eurusd_rolling  = daily_returns["EURUSD"].rolling(60).mean().dropna() * 100


# ── 3. DAILY RETURNS CHART ────────────────────────────────────────────────────
categories = {
    "Energy": (energy_rolling, "darkorange"),
    "Metals": (metals_rolling, "salmon"),
}

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

for i, (name, (data, color)) in enumerate(categories.items()):
    axes[i].plot(data, color=color, label=name)
    axes[i].plot(eurusd_rolling, color="blue", label="EURUSD", linewidth=2)
    axes[i].set_title(f"10-year Daily Returns of {name} and EURUSD")
    axes[i].set_ylabel("Daily Returns (%)")
    axes[i].set_xlabel("Time")
    axes[i].grid(True)
    axes[i].axhline(linewidth=1.5, color="black")
    axes[i].legend(fontsize=6)

agri_colors = {"Wheat": "red", "Coffee": "orange", "Cocoa": "olive"}
for asset in agricultural_reduced:
    axes[2].plot(agri_rolling[asset], color=agri_colors[asset], label=asset)
axes[2].plot(eurusd_rolling, color="blue", label="EURUSD", linewidth=2.5)
axes[2].set_title("10-year Daily Returns of Agricultural Commodities and EURUSD")
axes[2].set_ylabel("Daily Returns (%)")
axes[2].set_xlabel("Time")
axes[2].grid(True)
axes[2].axhline(linewidth=1.5, color="black")
axes[2].legend(fontsize=6)

mplcursors.cursor(hover=True)
plt.tight_layout()
save("daily_returns_commo.png")
plt.show()


# ── 4. PEARSON CORRELATION ────────────────────────────────────────────────────
pearson_corr = daily_returns.corr()

plt.figure(figsize=(14, 8))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", annot_kws={"size": 6})
plt.title("Pearson Correlation Heatmap - Daily Returns (2016-2026)")
plt.tight_layout()
save("pearson_corr_heatmap.png")

print("\n── Pairwise Pearson correlations with EURUSD ──")
for asset in commodity_columns:
    corr, p_value = stats.pearsonr(daily_returns["EURUSD"], daily_returns[asset])
    print(f"  {asset:<15} r={corr:+.4f}  p={p_value:.4f}")


# ── 5. ROLLING CORRELATION (60-day window) ────────────────────────────────────
rolling_corr = {
    asset: daily_returns["EURUSD"].rolling(60).corr(daily_returns[asset])
    for asset in daily_returns.columns if asset != "EURUSD"
}
rolling_corr = pd.DataFrame(rolling_corr).dropna()

categories_corr = {
    "Energy":       energy_commo,
    "Metals":       metals_commo,
    "Agricultural": agricultural_commo,
}

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
for i, (name, assets) in enumerate(categories_corr.items()):
    for asset in assets:
        axes[i].plot(rolling_corr[asset], label=asset)
    axes[i].set_title(f"EURUSD vs {name} — Rolling Correlation (60-Day Window)")
    axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Correlation Coefficient")
    axes[i].grid(True)
    axes[i].axhline(linewidth=1.5, color="black")
    axes[i].legend(fontsize=6)

mplcursors.cursor(hover=True)
plt.tight_layout(pad=3)
save("rolling_corr_all.png")
plt.show()


# ── 6. STANDARDISATION (Z-scores) ─────────────────────────────────────────────
scaler = StandardScaler()
daily_returns_scaled = pd.DataFrame(
    scaler.fit_transform(daily_returns),
    columns=daily_returns.columns,
    index=daily_returns.index
)

# Exclude EURUSD from PCA to avoid circularity
daily_returns_scaled_commo = daily_returns_scaled[commodity_columns]


# ── 7. PCA — Optimal component selection (90% variance threshold) ─────────────
pca_temp = PCA()
pca_temp.fit(daily_returns_scaled_commo)
cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.9) + 1

pca = PCA(n_components=n_components)
pca_array = pca.fit_transform(daily_returns_scaled_commo)
pc_columns = [f"PC{i+1}" for i in range(n_components)]

daily_returns_scaled_pca = pd.DataFrame(
    pca_array,
    columns=pc_columns,
    index=daily_returns_scaled_commo.index
)

print(f"\n── PCA Results ──")
print(f"  Components selected: {n_components}")
print(f"  Explained variance:  {pca.explained_variance_ratio_}")
print(f"  Total variance:      {pca.explained_variance_ratio_.sum():.2%}")

# Loadings matrix
loadings = pd.DataFrame(
    pca.components_[:n_components],
    columns=commodity_columns,
    index=pc_columns
)
print("\n── Loadings ──")
print(loadings)

# Loadings heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(loadings, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"PCA Loadings Heatmap ({n_components} components)")
plt.tight_layout()
save("PCA_loadings_heatmap.png")


# ── 8. VARIANCE-WEIGHTED PCA RECONSTRUCTION ───────────────────────────────────
# Step 1 — PC weights (explained variance ratios)
var_weights = pca.explained_variance_ratio_[:n_components]

# Step 2 — Weighted loadings per commodity
raw_weights = {
    asset: sum(loadings[asset][f"PC{i+1}"] * var_weights[i] for i in range(n_components))
    for asset in commodity_columns
}

# Step 3 — Normalise betas to sum to 1
total_weight = sum(raw_weights.values())
betas = {asset: w / total_weight for asset, w in raw_weights.items()}

print("\n── Variance-Weighted PCA Betas ──")
for asset, beta in betas.items():
    print(f"  {asset:<15} {beta:+.4f}")

# Step 4 — Reconstruct EURUSD
alpha = daily_returns_scaled["EURUSD"].mean()  # ≈ 0 by construction
reconstructed_EURUSD_pca = alpha + sum(
    betas[asset] * daily_returns_scaled[asset]
    for asset in commodity_columns
)

# Statistical tests
rmse_pca = np.sqrt(np.mean((daily_returns_scaled["EURUSD"] - reconstructed_EURUSD_pca) ** 2))
corr_pca, p_pca = stats.pearsonr(
    daily_returns_scaled["EURUSD"].dropna(),
    reconstructed_EURUSD_pca.dropna()
)
print(f"\n── PCA Model Performance ──")
print(f"  RMSE:        {rmse_pca:.4f}")
print(f"  Correlation: {corr_pca:.4f}")
print(f"  P-value:     {p_pca:.6f}")
print(f"  R-squared:   {corr_pca**2:.4f}")

# Visualisation
actual_EURUSD = daily_returns_scaled["EURUSD"].rolling(60).mean()
reconstructed_EURUSD_pca_rolling = reconstructed_EURUSD_pca.rolling(60).mean()

plt.figure(figsize=(14, 6))
plt.plot(reconstructed_EURUSD_pca_rolling, label="Reconstructed EURUSD (PCA)", color="orange")
plt.plot(actual_EURUSD, label="Actual EURUSD", color="blue", alpha=0.7)
plt.title("PCA Reconstructed vs Actual EURUSD Daily Returns")
plt.xlabel("Date")
plt.ylabel("Standardised Daily Returns (Z-score)")
plt.legend()
plt.grid()
mplcursors.cursor(hover=True)
save("PCA_reconstructed_vs_actual.png")


# ── 9. PRINCIPAL COMPONENT REGRESSION (PCR) ───────────────────────────────────
X_pcr = sm.add_constant(daily_returns_scaled_pca)
y_pcr = daily_returns_scaled["EURUSD"]
pcr_model = sm.OLS(y_pcr, X_pcr).fit()
print("\n── PCR OLS Summary ──")
print(pcr_model.summary())

# Extract gammas and convert to commodity betas
gammas = pcr_model.params[1:]
print("\n── Gammas (PC coefficients) ──")
print(gammas)

betas_pcr = loadings.T @ gammas
print("\n── PCR Commodity Betas ──")
print(betas_pcr.sort_values(ascending=False))

# Reconstruct EURUSD
reconstructed_EURUSD_pcr = alpha + sum(
    betas_pcr[asset] * daily_returns_scaled[asset]
    for asset in commodity_columns
)

# Statistical tests
rmse_pcr = np.sqrt(np.mean((daily_returns_scaled["EURUSD"] - reconstructed_EURUSD_pcr) ** 2))
corr_pcr, p_pcr = stats.pearsonr(
    daily_returns_scaled["EURUSD"].dropna(),
    reconstructed_EURUSD_pcr.dropna()
)
print(f"\n── PCR Model Performance ──")
print(f"  RMSE:        {rmse_pcr:.4f}")
print(f"  Correlation: {corr_pcr:.4f}")
print(f"  P-value:     {p_pcr:.6f}")
print(f"  R-squared:   {corr_pcr**2:.4f}")

# Visualisation
reconstructed_EURUSD_pcr_rolling = reconstructed_EURUSD_pcr.rolling(60).mean()

plt.figure(figsize=(14, 6))
plt.plot(reconstructed_EURUSD_pcr_rolling, label="Reconstructed EURUSD (PCR)", color="orange")
plt.plot(actual_EURUSD, label="Actual EURUSD", color="blue", alpha=0.7)
plt.title("PCR Reconstructed vs Actual EURUSD Daily Returns")
plt.xlabel("Date")
plt.ylabel("Standardised Daily Returns (Z-score)")
plt.legend()
plt.grid()
mplcursors.cursor(hover=True)
save("PCR_reconstructed_vs_actual.png")
plt.show()


# ── 10. MODEL COMPARISON SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"{'Metric':<20} {'PCA':>10} {'PCR':>10}")
print("-" * 40)
print(f"{'P-value':<20} {p_pca:>10.4f} {p_pcr:>10.6f}")
print(f"{'R-squared':<20} {corr_pca**2:>10.4f} {corr_pcr**2:>10.4f}")
print(f"{'Correlation':<20} {corr_pca:>10.4f} {corr_pcr:>10.4f}")
print(f"{'RMSE':<20} {rmse_pca:>10.4f} {rmse_pcr:>10.4f}")
print(f"\nCharts saved to: {OUTPUT_DIR}")
