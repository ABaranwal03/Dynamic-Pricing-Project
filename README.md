# Dynamic Pricing for Ride-Hailing

Exploratory analysis and model benchmarking to understand what drives ride fares and how well demand, supply and ride features can predict them.

## Overview

A ride-sharing company wants to move beyond pricing on ride duration alone. This project explores a historical ride dataset, engineers demand/supply features, and compares regression models that predict ride cost.

All costs are treated as **Indian Rupees (₹)**. The original Kaggle dataset does not specify a currency.

## Dataset

- **Source:** [Dynamic Pricing Dataset](https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset) by arashnic on Kaggle
- **License:** CC0: Public Domain
- **Size:** 1,000 rides, 10 columns, no missing values

| Column | Description |
|---|---|
| `Number_of_Riders` | Riders requesting rides (demand) |
| `Number_of_Drivers` | Drivers available (supply) |
| `Location_Category` | Urban / Suburban / Rural |
| `Customer_Loyalty_Status` | Regular / Silver / Gold |
| `Number_of_Past_Rides` | Rides the customer has taken before |
| `Average_Ratings` | Customer's average rating |
| `Time_of_Booking` | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | Economy / Premium |
| `Expected_Ride_Duration` | Expected duration in minutes |
| `Historical_Cost_of_Ride` | Ride cost (target variable) |

## What the notebook covers

1. **Data exploration:** distributions, categorical trends, correlations
2. **Feature engineering:** demand-to-supply ratio, cost per minute, ride-history deciles
3. **Pricing mechanics:** how cost varies with demand pressure, location and time of day
4. **Preparation:** one-hot encoding, train/test split (80/20), scaling, multicollinearity check (VIF)
5. **Modelling:** Linear Regression, Random Forest and XGBoost
6. **Evaluation:** error analysis, ±10% pricing tolerance, partial dependence, SHAP

## Key findings

- **Demand pressure doesn't translate into fares linearly.** Median cost does not keep rising as the demand-to-supply ratio worsens; at the highest-shortage levels the median actually dips.
- **Ride duration is the main fare driver.** SHAP analysis shows it can shift a fare by about ₹152 per trip, while a premium vehicle adds only about ₹16 on average.
- **Customer tenure is not priced in.** Median cost per minute is almost flat between brand-new and veteran riders.
- **Demand and supply are only moderately correlated**, so the engineered demand-to-supply ratio captures them without much redundancy.

## Approach

- **Features:** demand-to-supply ratio, cost per minute (used for analysis only and excluded from the models to avoid target leakage), ride-history deciles.
- **Preprocessing:** one-hot encoding of categoricals, 80/20 train/test split (`random_state=42`), standard scaling fitted on the training set only.
- **Multicollinearity:** features with high VIF (`Loyalty_Decile`, `Number_of_Drivers` and the demand-ratio bucket dummies) were removed before modelling.
- **Models:** Linear Regression, Random Forest (100 trees) and XGBoost (100 trees), compared on RMSE and R².

## Results

| Model | RMSE (₹) | R² |
|---|---|---|
| Linear Regression | 67.60 | 0.8747 |
| Random Forest | 74.45 | 0.8480 |
| XGBoost | 76.43 | 0.8398 |

Linear Regression had the best RMSE and R² and the smallest worst-case pricing errors (about ₹164 overpriced and ₹158 underpriced). The tree models placed more predictions within ±10% of the actual fare (about 42% vs 37%) but produced larger worst-case errors, which matters for revenue risk.

## Getting started

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook dynamic_price.ipynb
```

The notebook expects `dynamic_pricing.csv` in the same folder.

## Project structure

```
├── dynamic_price.ipynb     # Analysis and modelling
├── dynamic_pricing.csv     # Dataset
├── requirements.txt
├── LICENSE
└── README.md
```

## Limitations

- The dataset is small (1,000 rows) and appears synthetic, so results may not reflect real-world pricing.
- Models predict *historical* fares, not revenue-optimal fares. No customer-response or booking-conversion data is available.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The dataset is CC0 (public domain) as published on Kaggle.

## Author

Apoorva Baranwal · [GitHub](https://github.com/ABaranwal03) · [LinkedIn](https://www.linkedin.com/in/apoorva-baranwal-a287aa24b)

## Acknowledgements

Dataset by [arashnic](https://www.kaggle.com/arashnic) on Kaggle.
