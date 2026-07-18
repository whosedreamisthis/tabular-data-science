# Student Sleep & Mental Health Analysis

An exploratory data analysis (EDA) project investigating the relationships between student lifestyle habits, academic performance, and mental well-being indicators such as burnout.

## Project Overview
This project analyzes a dataset of student metrics to identify behavioral patterns that correlate with self-reported burnout. By examining factors like sleep duration, screen time, and exercise, we aim to provide data-driven insights into the predictors of student wellness.

## Methodology
The analysis was performed using Python with `pandas` for data manipulation and `scikit-learn` for feature correlation analysis.

1.  **Data Preprocessing**: 
    *   Removed non-predictive identifiers (`student_id`).
    *   Converted categorical/boolean flags (`feels_burned_out`) into numeric representations for correlation analysis.
2.  **Analysis**:
    *   Calculated a correlation matrix to identify the strength of relationships between lifestyle variables and burnout metrics.
    *   Ranked predictors by their Pearson correlation coefficients.

## Key Findings
Our analysis identified several strong indicators related to student burnout:

*   **Primary Predictors**: Stress level ($r \approx 0.79$) and anxiety score ($r \approx 0.61$) are the strongest positive correlates with burnout.
*   **Lifestyle Impact**: 
    *   Higher screen time ($r \approx 0.41$) and social media usage ($r \approx 0.31$) are associated with increased burnout.
    *   Adequate sleep ($r \approx -0.39$) and regular exercise ($r \approx -0.24$) serve as significant mitigating factors.
*   **Academic Correlation**: A negative correlation between GPA and burnout ($r \approx -0.44$) suggests that students maintaining higher academic standing tend to report lower burnout levels.

## Technologies Used
*   **Language**: Python
*   **Libraries**: `pandas`, `scikit-learn`
*   **Environment**: Jupyter Notebook

## How to Run
1.  Ensure you have Python and the required libraries installed:
    ```bash
    pip install pandas scikit-learn
    ```
2.  Place the `student_sleep_mental_health_2026.csv` file in the project root.
3.  Run the analysis script:
    ```bash
    python analyze_wellness.py
    ```

---
*Developed as a portfolio project in 2026.*
