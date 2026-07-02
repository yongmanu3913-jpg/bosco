# bosco

## Seoul Depression & Neighborhood Environment Spatial Analysis

This repository contains research code and documentation for analyzing spatial association patterns between depression prevalence and neighborhood environmental characteristics at the administrative-dong level in Seoul, South Korea.

본 저장소는 박사학위논문 「서울시 행정동별 우울증 유병률과 근린환경 특성의 연관구조 분석」을 기반으로 한 연구 코드 및 재현성 문서화 저장소입니다.

## Project Overview

This project estimates treatment-based depression prevalence at the administrative-dong level in Seoul and analyzes its spatial association with neighborhood environmental characteristics.

The research workflow includes:

1. Estimating administrative-dong-level depression prevalence using small-area estimation
2. Identifying spatial clustering using Moran’s I and Getis-Ord’s Gi*
3. Comparing global and spatially varying associations using OLS and MGWR
4. Interpreting significant local associations through population, socioeconomic, medical-accessibility, and neighborhood-environment characteristics

## Study Area

- Spatial unit: Administrative dongs in Seoul
- Study year: 2024
- Research field: Urban planning, neighborhood environment, spatial epidemiology, and mental health

## Main Methods

- Small-Area Estimation
- Spatial Autocorrelation Analysis
- Hotspot Analysis
- Ordinary Least Squares Regression
- Multiscale Geographically Weighted Regression
- Group Comparison Analysis

## Why This Project Matters

This repository provides a reproducible research workflow for studying spatial inequalities in treatment-based depression prevalence and neighborhood environmental characteristics in Seoul.

The project is designed to support transparent documentation of spatial analysis workflows, while respecting restrictions on sensitive health-insurance and administrative datasets.

## Data Notice

Raw health-insurance data, personally identifiable information, and restricted administrative datasets are not included in this public repository.

Only reproducible code, documentation, public-data processing scripts, and anonymized or sample data will be shared when permitted by the original data providers.

## Repository Structure

```text
bosco/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # Not committed
│   ├── processed/    # Processed public or permitted data only
│   └── sample/       # Sample or synthetic data
├── docs/             # Documentation and research notes
├── notebooks/        # Exploratory analysis notebooks
├── src/              # Reproducible analysis scripts
└── outputs/
    ├── figures/
    └── tables/
```

## How to Use

1. Review the project workflow in `docs/`.
2. Check the variable definitions in `docs/data_dictionary.md`.
3. Run example scripts using sample or synthetic data.
4. Reproduce spatial analysis outputs where permitted by data restrictions.

## Installation

```bash
pip install -r requirements.txt
```

## Planned Contents

- Data preprocessing scripts
- Small-area estimation workflow
- Spatial autocorrelation and hotspot analysis
- OLS and MGWR modeling scripts
- Figure and table generation scripts
- Documentation for reproducibility

## Roadmap

- [ ] Add public-data preprocessing scripts
- [ ] Add synthetic sample dataset
- [ ] Add spatial autocorrelation example
- [ ] Add MGWR workflow documentation
- [ ] Add figure and table reproduction scripts

## Limitations

The estimated depression prevalence is based on treatment and prescription records, not direct observation at the administrative-dong level. Therefore, the results should be interpreted as regional-level associations rather than causal effects.

This repository does not provide restricted health-insurance data or personally identifiable information.

## Citation

Jang, Y. J. (2026). *An Analysis of Association Patterns between Depression Prevalence and Neighborhood Environmental Characteristics at the Administrative Dongs in Seoul*. Doctoral dissertation draft.
