# EUR/USD & Commodity Correlation Research

Multi-factor EUR/USD return model investigating the relationship between 
FX and commodity markets using dimensionality reduction and regression 
techniques across a 10-year daily dataset (2016–2026).

## Objective
Identify which commodity markets drive EUR/USD daily returns and quantify 
their explanatory power using both unsupervised and supervised methods.

## Methodology
- **Data**: 20 commodities across metals, energy, and agriculture — 
  daily returns via yfinance (2016–2026)
- **Pearson correlation & 60-day rolling correlation**: identified 
  regime shifts in commodity/FX relationships across macro cycles
- **PCA**: decomposed commodity co-movement into principal components 
  explaining 91.86% of variance across 14 components
- **Variance-weighted PCA reconstruction**: rebuilt EUR/USD returns 
  using commodity betas derived from explained variance ratios
- **Principal Component Regression (PCR)**: benchmarked against PCA 
  model across RMSE, R², and p-value metrics

## Key Finding
Commodities have statistically significant but limited explanatory power 
on daily EUR/USD returns (PCR R²=0.014), consistent with short-horizon 
FX predictability literature. Rolling correlation analysis revealed 
stronger commodity/FX co-movement during macro stress periods.

## Stack
Python — pandas, numpy, statsmodels, sklearn, seaborn, matplotlib, yfinance
