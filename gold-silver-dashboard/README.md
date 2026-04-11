# Gold & Silver Price Dashboard (1968–2026)

An interactive Power BI dashboard analyzing gold and silver price trends across 58 years of market history from the Nixon shock of 1971 to the geopolitical driven surge of 2025–2026.


## Why This Project Matters

Precious metals are one of the oldest and most reliable indicators of global economic health. When investors are scared whether from a financial crisis, a pandemic, or geopolitical conflict they flee to gold and silver. When markets are calm, the metals cool off.

This dashboard was built to answer a simple but powerful question: **what does 58 years of gold and silver prices actually tell us about the world?**

The answer turns out to be a lot:

- The 1971 Nixon shock (US abandons gold standard) triggered gold's first major rally
- The 1980 Hunt Brothers silver squeeze nearly cornered the global silver market
- The 2008 financial crisis sent gold on a 3-year climb to $1,900
- In April 2020, COVID-19 caused the gold/silver ratio to hit **111x** one of the most extreme readings in recorded history, signaling silver was massively undervalued
- The 2025–2026 geopolitical tensions (US-Iran-Israel conflict) drove gold past **$5,000** and silver past **$90** new all time highs

Understanding these patterns is directly relevant to roles in financial services, risk management, and investment analytics which is exactly why this project was built.

## Dashboard Features

### Visual 1 — Price Trend Line Chart
- Plots average monthly gold and silver prices from 1968 to 2026
- Interactive metal selector (Gold / Silver toggle)
- Date slicer to zoom into any time period
- Clearly shows major market events: 2008 crisis, 2011 peak, 2020 COVID crash, 2025–2026 rally

### Visual 2 — Average Price by Year (Bar Chart)
- Clustered bar chart comparing gold and silver average annual prices side by side
- Immediately shows how dramatically prices diverged post-2020
- Filtered to 2000–2026 to focus on the most significant market era

### Visual 3 — DAX Queries (Gold/Silver Ratio Analysis)
- Written directly in DAX query editor using `CALCULATE`, `AVERAGE`, `DIVIDE`, `SUMMARIZECOLUMNS`, `ADDCOLUMNS`, `SUMMARIZE`, `VAR`, `TOPN`
- Computes gold/silver ratio for every month reveals historically extreme readings
- Identifies best and worst performing years for each metal


## Key Insights Found

| Event | Year | What Happened |
|---|---|---|
| Nixon shock | 1971 | US leaves gold standard gold price freed from $35/oz peg |
| Hunt Brothers squeeze | 1980 | Silver hit $49/oz before COMEX intervened |
| Post-2008 rally | 2008–2011 | Gold climbed from $800 to $1,900 as trust in banks collapsed |
| COVID ratio spike | Apr 2020 | Gold/Silver ratio hit **111x** — silver historically cheap |
| Geopolitical surge | 2025–2026 | Gold surpassed $5,000 · Silver hit all-time high of $121 |

## Technical Stack

| Tool | Purpose |
|---|---|
| Python (pandas) | Data cleaning, merging 4 datasets, handling date mismatches |
| Power BI Web | Dashboard building, visualization, interactivity |
| DAX | Analytical queries — ratio calculation, yearly aggregations, ranking |
| GitHub | Version control and portfolio hosting |

## Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| Gold daily prices (1968–2021) | Kaggle | Daily |
| Silver daily prices (1968–2021) | Kaggle | Daily |
| Gold monthly prices (1833–2026) | github.com/datasets/gold-prices | Monthly |
| Silver daily prices (2016–2026) | Kaggle (pranjalverma08) | Daily |

All four datasets were merged and cleaned using Python. Daily data was converted to monthly averages to ensure gold and silver aligned on the same dates — a critical data engineering decision that enabled the ratio analysis.


## Data Engineering Decisions

**Problem:** Gold data ended in April 2021 while silver data extended to January 2026. Directly joining on date left the ratio blank for 2021–2026.

**Solution:** Converted all data to monthly averages using `pandas groupby` and `dt.to_period('M')`. This aligned both metals to the same monthly timestamps, enabling clean ratio calculations across the full 58 year period.

**Why this matters in banking:** Date mismatches between data feeds are one of the most common data quality issues in financial reporting. Knowing how to identify and resolve them is a core analyst skill.


## Skills Demonstrated

- **Data wrangling** — merging 4 datasets with different frequencies and date formats using Python pandas
- **Data quality thinking** — identifying and resolving date alignment issues before analysis
- **Power BI** — building interactive multi visual dashboards with slicers, filters, and formatted visuals
- **DAX** — writing analytical queries using CALCULATE, DIVIDE, SUMMARIZECOLUMNS, ADDCOLUMNS, VAR, TOPN
- **Financial domain knowledge** — understanding what gold/silver ratio means, why it matters, and how to contextualize price movements within real-world events
- **Storytelling with data** — framing raw price data as a narrative about global economic history


## Relevance to Financial Services Roles

This project directly maps to skills required for data analyst roles at Canadian banks (RBC, TD, CIBC) and fintech companies:

- **Portfolio analytics** tracking asset performance over time, identifying trends and anomalies
- **KPI development** building measures like gold/silver ratio that drive investment decisions
- **Dashboard development** creating self serve BI tools for non-technical stakeholders
- **Data governance** handling data quality issues, ensuring accuracy before reporting
- **Stakeholder communication** turning raw data into a story that a portfolio manager or risk officer can act on

[LinkedIn](https://www.linkedin.com/in/aryan-bansal17/) · [GitHub](https://github.com/Aryanbansal1785) · [Kaggle](https://www.kaggle.com/work)
