# Used Car Price Prediction

This project implements an end-to-end machine learning pipeline to predict used car prices. It features custom preprocessing for noisy, real-world data and uses an XGBoost Regressor to achieve high predictive accuracy.

## Project Overview
The pipeline handles raw data ingestion, cleans inconsistent string-based numerical features, standardizes categorical inputs, and groups rare vehicle models to mitigate noise.

## Model Performance
Our model demonstrates strong generalization on unseen data:

| Metric | Score |
| :--- | :--- |
| **Final Test $R^2$ Score** | **0.772** |
| **Training $R^2$ Score** | **0.897** |

## Key Features
* **Custom Preprocessing:** 
    * `NumericalStringCleaner`: Handles currency and unit-based string formatting.
    * `RareCategoryGrouper`: Automatically buckets low-frequency car models to reduce noise.
    * `AgeCalculator`: Derives vehicle age relative to the current year.
    * `TransmissionStandardizer`: Normalizes messy text data into binary categories.
* **Pipeline Architecture:** Integrated `ColumnTransformer` with `TargetEncoder` and `XGBRegressor` for robust training.

## Project Structure
```text
used-cars/
├── data/
│   └── used_cars.csv        # Input dataset
├── src/
│   └── preprocessing.py     # Custom transformer classes (BaseEstimator/TransformerMixin)
├── pipeline.py              # Main pipeline definition, training, and evaluation
└── README.md                # This file

## How to Run
1. Ensure your environment has the required dependencies (`scikit-learn`, `xgboost`, `pandas`, `numpy`).
2. Place the dataset in the `data/` folder as `used_cars.csv`.
3. Execute the training script:
   ```bash
   python pipeline.py

### A few quick tips:
*   **Formatting:** I have used Markdown's table and code block syntax so it renders beautifully on GitHub. 
*   **The "$R^2$":** I used LaTeX formatting for the $R^2$ symbol as per your requirement.
*   **Customization:** If you plan on adding more details later (like an image of your feature importance plot), you can simply drop a `.png` file in the folder and link to it in this file using `![Feature Importance](./results/feature_importance.png)`.