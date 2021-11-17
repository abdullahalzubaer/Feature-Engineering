# Feature-Engineering


Part 1
---


Overview:

Dataset: concrete.csv

Model: Random Forest Regressor

Evaluated score by cross-validation

Metric: Mean absolute error

Error before creating 3 features: 8.2317

Error after creating 3 features: 7.8215

Observation: Adding informative features can increase the performance of the model.

Part 2
---

Objective: Identifying and ranking relevant features w.r.t target

Metric: Mutual Information (MI) -> It describes the relationships in terms of uncertainty. MI between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. That is, if we know the value of a feature, how much more confident would we be about the target? Uncertainty is measured using "entropy". The least possible MI is zero, the maximum possible MI is infinite, since MI is a logarithmic quantity it increases very slowly (above 2.0 are uncommon).

Remember: MI cannot detect the interaction between features, for that we can create new features (combining other features) and calculate its MI. MI is a univariate metric.

Example: Mutual Information between ExterQual and SalePrice

Dataset: Ames Housing

<img src="images/par2_plot1.jpg" width = "500" >

<!-- <img src="images/model_architecture.png" width = "500" > -->

Observation: We can observe the relationship between the feature "ExterQual" and the target, "SalePrice". We can interpret the plot as follows. Knowing the value of the ExterQual feature makes us more certain regarding the target. Because, each category in ExterQual roughly separates the range of Sale Price, helping the model to use this information to improve its performance.


Dataset: Automobile_data

Metric: Mutual Information

<img src="images/par2_plot2.jpg" width = "500" >

Observation: MI score for each feature in the dataset w.r.t the target. The higher the MI score, the better is that feature to predict the price of the car, i.e. the relationship is strong!


<img src="images/par2_plot3.jpg" width = "500" >

Observation: Plotting the relation between curb-weight and price by fitting a linear regression to curb-weight and price. As curb-weight increases, the price also increases. This shows strong relation, as also shown by Mutual Information Score

