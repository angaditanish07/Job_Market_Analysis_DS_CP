# Power BI dashboard setup

Import the CSV files from this output folder into Power BI Desktop.

## Page 1: Job Market Overview
- KPI cards: total listings, unique companies, unique industries, salary availability rate.
- Bar charts: listings by industry, primary city, role family and experience category.
- Add slicers for industry, primary city, role family and posting year.

## Page 2: Salary and Skills
- Cards: median annualised salary midpoint and salary availability rate.
- Bar charts: median salary by experience, role family and industry.
- Bar chart: most frequently mentioned skills.
- Add a note that salary is only available for a subset and the source currency is undocumented.

## Page 3: Recruitment Trends and Forecasting
- Line chart: monthly postings and 3-month moving average.
- Line or column chart: three-month forecast.
- Table: model comparison using MAE, RMSE and MAPE.
- Add a note that missing months are not assumed to have zero listings and forecasts are exploratory.

The Python script prepares Power BI-ready CSVs and this field mapping. Build and save the interactive report in Power BI Desktop; this script does not generate a .pbix file.
