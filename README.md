# CCE Email Optimization Analysis

This project analyzes survey responses about emails sent by the Center for Civic Engagement (CCE) of AUCA. The goal is to understand what makes people more likely to read these emails and what content or formatting choices are preferred by respondents.

The analysis is implemented in [ds_mailings_optimization.py](ds_mailings_optimization.py) and uses the survey data in [Customer Feedback.csv](Customer%20Feedback.csv).

## Project Goal

The main objective of this project is to improve CCE mailing effectiveness by answering questions like:

- Which email formats and content styles are preferred?
- What factors are associated with people reading CCE emails?
- Are there different groups of respondents with different satisfaction patterns?

## How The Project Works

The script runs a full analysis pipeline:

1. It loads the survey CSV file and renames the columns into shorter analysis-friendly names.
2. It cleans the data by trimming text, fixing category labels, and removing columns that are not needed.
3. It converts survey answers into machine-readable values:
   - Likert-scale answers are mapped to numbers from 1 to 4.
   - Yes/no and ordered preference questions are encoded numerically.
   - Multi-select answers are expanded into binary indicator columns.
4. It creates a target variable called `target`:
   - `1` means the respondent reads CCE emails.
   - `0` means the respondent does not read them.
5. It performs **K-Modes clustering** to group respondents by categorical preference patterns.
6. It standardizes the feature matrix and applies **SMOTE** to balance the classes before modeling.
7. It trains a **Logistic Regression** model to estimate which features are associated with reading behavior.
8. It evaluates the model using:
   - 5-fold cross-validation
   - ROC AUC
   - classification report
9. It compares cluster satisfaction scores using a **Mann-Whitney U test**.
10. It exports visual summaries as PNG files into the `output/` folder.

## What You Achieved

This project produced a complete end-to-end survey analytics workflow and generated usable outputs for reporting.

From the latest successful run:

- Total engineered feature matrix: `46` rows and `32` features
- Readers: `43`
- Non-readers: `3`
- Clusters found: `3`
- Cluster sizes: `{0: 14, 1: 11, 2: 21}`
- Cross-validated accuracy: `0.954 +- 0.044`
- ROC AUC: `1.000`

You also generated three visual deliverables:

- [output/cce_dashboard.png](output/cce_dashboard.png)
- [output/cce_preferences.png](output/cce_preferences.png)
- [output/cce_keywords_features.png](output/cce_keywords_features.png)

## Main Outputs

### 1. Dashboard

`cce_dashboard.png` combines the core analytical results in one figure:

- K-Modes cost curve
- PCA cluster visualization
- Cluster satisfaction heatmap
- Logistic regression feature coefficients
- ROC curve
- Reading-frequency summary

### 2. Preferences Summary

`cce_preferences.png` shows the most preferred options for:

- title length
- emoji usage
- body length
- colored text
- link placement
- respondent role

### 3. Keywords And Features Summary

`cce_keywords_features.png` summarizes:

- the most attractive email-title keywords
- the most valued characteristics of a good email

## Statistical Findings

The Mann-Whitney comparison between Cluster 0 and Cluster 1 did not show statistically significant differences across the five satisfaction dimensions tested:

- Deadlines
- Formatting
- Engaging content
- Opportunities
- Overall satisfaction

In the latest run, all reported `p` values were above `0.05`.

## Files In This Project

- [ds_mailings_optimization.py](ds_mailings_optimization.py): main analysis script
- [Customer Feedback.csv](Customer%20Feedback.csv): survey dataset
- [run.log](run.log): latest console output from the script
- [output](output): generated charts

## How To Run

Install the required Python packages:

```bash
python3 -m pip install --user numpy pandas matplotlib scikit-learn scipy imbalanced-learn kmodes
```

Run the script from the project folder:

```bash
python3 ds_mailings_optimization.py
```

Generated charts will be saved in the `output/` directory.

## Notes

- The dataset is small and highly imbalanced because only `3` respondents were labeled as non-readers.
- Because of that class imbalance, the model scores should be interpreted carefully.
- The script is designed for local report generation rather than as a reusable package or web app.
