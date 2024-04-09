# Causal Inference Analysis: The Impact of Lighting on Accidents

## Project Overview
This project aims to investigate the causal relationship between lighting conditions and the occurrence of accidents. By leveraging CAUSAL ML, I seek to understand how different levels of lighting can influence the likelihood of accidents and provide insights for improving safety measures.

## Data Description
The dataset used in this analysis contains information about various accidents, including details about the lighting conditions at the time of each accident, the severity of the accidents, and other relevant factors.

## Methodology
A range of causal inference models were experienced to estimate the causal effect of lighting on accident outcomes. The analysis includes:

- Exploratory Data Analysis (EDA) to understand the data distribution and identify patterns.
- Data preprocessing and feature selection to prepare the dataset for causal analysis.
- Model Objective:
**Use case senario: Using road, light , weather conditions to predict the number of causlties in accidents.**
**Treatment: 'Light_Conditions_Darkness - lights lit'**
**other predictors: all attributes available** 
**y: number of causalties**

- Models:LGBM and XGBT


## Key Findings
## Features:
### Key Observations:

- **Top Features**: Both methods identified `Vehicle_Type_bus` and `Accident_Severity` as highly important features, but their order of importance is swapped. This indicates that regardless of the method, these two features are crucial in predicting the outcome.

- **Cluster Similarity**: Features related to cluster similarity(derived from latitude and longitude that represents geo data ) appear prominently in both methods, suggesting that the location of accidents play a significant role in the model. However, the specific clusters identified as important vary between the two methods, which could suggest differences in how each method evaluates the influence of correlated features.

- **Environmental Conditions**: Both methods highlight the significance of light and road surface conditions, reflecting their impact on the model's predictions. However, the permutation method seems to reduce the relative importance of specific light conditions and weather compared to the auto method, possibly due to the permutation method's ability to account for interactions between features more effectively.

- **Urban or Rural Area**: The feature `Urban_or_Rural_Area_Rural` is identified as important by both methods but ranks higher in the permutation method. This could indicate that the permutation method captures the contextual impact of the environment on the outcome more effectively.

- **Negative Importance**: The permutation method uniquely identifies a few features with slightly negative importance scores (e.g., `Light_Conditions_Darkness - lights lit`, `Weather_Conditions_Fog or mist`). Negative values can suggest that permuting these features leads to a slight improvement in the model's accuracy, indicating that their original contribution might be misleading or non-informative. This aspect is not captured in the auto method.

### Conclusion:

Both the auto and permutation methods offer valuable insights into which features are most influential in predicting the number of causalties. The auto method highlights features that are statistically significant within the model's structure, while the permutation method provides a more nuanced understanding of feature importance by evaluating the impact of each feature on the model's predictive performance in a more holistic manner. The slight discrepancies and negative importance scores observed with the permutation method underscore the complex interdependencies among features and their collective impact on model predictions.


## **Final Conclusion**

When comparing the feature importance results from the LGBM and XGBoost (XGBT) models on the test set, several key observations and differences stand out.

1. **Cluster Similarity Features**:
   - **LGBM**: Cluster similarity features like `Urban_or_Rural_Area_Rural` and `Accident_Severity` top the list, indicating these features are highly influential in LGBM's predictions. This suggests that the model heavily relies on the geographical and severity attributes for its predictions.
   - **XGBoost**: Similarly, cluster similarity features are very important but the order is different. For example, `Cluster 0 similarity` has the highest importance in XGBoost, unlike in LGBM where it's lower. This variance implies that XGBoost might be capturing different aspects of the data through these cluster similarities or weighing them differently.

2. **Geographical Features**:
   - **LGBM**: `Urban_or_Rural_Area_Rural` is the most important feature, significantly more than `Urban_or_Rural_Area_Urban`. This could indicate that rural areas have distinct characteristics or patterns that are highly predictive.
   - **XGBoost**: Shows a lower importance for `Urban_or_Rural_Area_Rural` and a negative importance for `Urban_or_Rural_Area_Urban`, which suggests a different interpretation of how urbanity influences the number of causalties. The negative importance might indicate an inverse relationship for some features in the XGBoost model.

3. **Vehicle and Road Type Features**:
   - **LGBM**: Vehicle type (`Vehicle_Type_bus`) and various road types have moderate to low importance. This suggests that while they are considered, they are not as critical as geographical or cluster similarity features.
   - **XGBoost**: `Vehicle_Type_bus` and `Road_Type_Slip road` have negative importance, suggesting that their presence (or the pattern they represent) might lead to a lower probability of the predicted outcome. This contrasts with LGBM, where these features had a positive, albeit small, importance.

4. **Weather and Light Conditions**:
   - Both models attribute low to negligible importance to weather conditions and light conditions, though XGBoost assigns negative importance to several of these features, indicating a potential predictive value in the absence (or presence) of these conditions.

5. **Negative Importance in XGBoost**:
   - XGBoost shows negative importance for several features, including some cluster similarities, vehicle types, and road conditions. Negative importance in XGBoost can suggest that the model finds a reverse correlation with the target variable. In contrast, LGBM's importances are all non-negative, which aligns with its method of calculating feature importance.


In summary, both models consider cluster similarities and geographical features as important, but they diverge significantly in how they evaluate the importance of specific features, including vehicle types, road types, and weather conditions. The original hypothesis of taking lights-lit as treatment does not affect the outcome as much as expected.



