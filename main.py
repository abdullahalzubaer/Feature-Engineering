import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
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

# For Automobile_data dataset

# Converting the "?" to NaN while reading the data
df = pd.read_csv("/content/Automobile_data.csv", na_values="?")
df = df.dropna()
df.isnull().sum()
X = df.copy().reset_index()
X.drop("index", axis="columns", inplace=True)
y = X.pop("price")


for colname in X.select_dtypes("object"):
    X[colname], _ = X[
        colname
    ].factorize()  # Remember these features are not discrete NOT continuous!

"""Creating boolean series, need this for MI. MI treats discrete features differently from continuous features.
Consequently, we need to tell MI which are which"""
discrete_features = X.dtypes == int
discrete_features

# MI score in a function


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


make_mi_scores(X, y, discrete_features)


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))  # how many y values we have
    # taking the name of the y values which isthe index of mi_score series
    ticks = list(scores.index)
    plt.barh(width, scores, color="tab:blue")
    plt.yticks(width, ticks)  # providing the name of the features on the y axis
    plt.title("Mutual Information Scores")


plt.figure(figsize=(10, 6))
plot_mi_scores(mi_scores)
plt.savefig("par2_plot2.jpg", bbox_inches="tight", dpi=600)
