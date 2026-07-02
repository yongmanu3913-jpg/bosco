# Reproducibility Workflow

This document describes the planned reproducible workflow for the project.

## 1. Data Preparation

- Collect administrative-dong-level demographic and socioeconomic indicators
- Build neighborhood environment indicators
- Exclude restricted or personally identifiable data from the public repository
- Prepare sample or synthetic data for demonstration

## 2. Small-Area Estimation

- Use upper-level depression case counts and lower-level auxiliary indicators
- Estimate administrative-dong-level treatment-based depression prevalence
- Validate model stability and uncertainty
- Export estimated prevalence for spatial analysis where permitted

## 3. Spatial Distribution Analysis

- Calculate descriptive statistics for estimated depression prevalence
- Visualize administrative-dong-level spatial distribution
- Conduct Moran’s I analysis
- Conduct Getis-Ord’s Gi* hotspot analysis

## 4. Regression Modeling

- Fit OLS models to examine global associations
- Fit MGWR models to examine spatially varying associations
- Compare model fit using AIC, AICc, R-squared, adjusted R-squared, and residual spatial autocorrelation

## 5. Local Association Interpretation

- Identify significant local association areas
- Compare significant and non-significant areas
- Interpret associations through demographic, socioeconomic, medical-accessibility, and neighborhood-environment characteristics

## 6. Outputs

- Tables
- Figures
- Maps
- Reproducible scripts
- Documentation

## Data Restriction

Restricted health-insurance data are not shared in this repository. Public examples will rely on sample or synthetic data.
