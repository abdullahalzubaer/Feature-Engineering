import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

plt.style.use("ggplot")

"""Part 1"""

df = pd.read_csv("/content/concrete.csv")

X = df.copy()
y = X.pop("CompressiveStrength")

baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
print(baseline_score)
baseline_score = -1 * baseline_score.mean()
print(f"Mean Absolute Error for Baseline: {baseline_score:.4f}")

# Creating new three features (the features can come from domain knowledge or intuition)

X["FCRation"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRation"] = X["Water"] / X["Cement"]

model = RandomForestRegressor(criterion="absolute_error", random_state=2)
model_score = cross_val_score(model, X, y, scoring="neg_mean_absolute_error")
model_score = -1 * model_score.mean()
print(f"Mean Absolute Error for Model: {model_score:.4f}")


"""Part 2"""

df = pd.read_csv("/content/ames.csv")
plt.figure(figsize=(10, 6))
ax = sns.stripplot(data=df, x="ExterQual", y="SalePrice").set(
    title="Mutual Information of SalePrice vs ExterQual"
)
plt.savefig("par2_plot1.jpg", bbox_inches="tight", dpi=600)
