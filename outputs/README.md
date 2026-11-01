# Naukri.com Job Market Analysis

## Run
```bash
pip install pandas numpy scipy matplotlib scikit-learn
python naukri_job_market_analysis.py --input "your_naukri_dataset.csv" --output-dir outputs
```

## Main outputs
- `cleaned_dataset.csv`: cleaned rows and engineered features
- `kpi_summary.csv`, `kpi_by_industry.csv`, `kpi_by_location.csv`, `kpi_by_experience.csv`, `kpi_by_role.csv`
- `monthly_job_postings.csv`, `forecast_results.csv`, `forecast_model_comparison.csv`, `forecast_train_test.csv`
- `statistical_results.csv`, `descriptive_statistics.csv`, quality and feature-dictionary reports
- `charts/` and `forecasting/`: PNG visualizations
- `powerbi_field_mapping.csv`: suggested visuals and source tables
- `project_documentation.json`: problem, scope, stakeholders, success criteria and limitations

## Suggested Power BI pages
1. Job Market Overview: KPI cards, listings by industry, role and location, experience distribution, monthly trend.
2. Salary and Skills: salary disclosure, median salary by experience/industry/role, skill mentions and salary distribution.
3. Recruitment Trends and Forecasting: monthly listings, moving average, forecast, model comparison and forecast evaluation.

## Interpretation warning
The dataset is a historical one-portal sample with uneven monthly coverage. Forecasts are exploratory and cannot be interpreted as present-day labour demand. Currency is not documented in the source.
