import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import numpy as np

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

# sns.set(rc={"figure.figsize":(10, 6)}) #width=3, #height=4
# sns.scatterplot(x="curb-weight", y="price", data=df)
# plt.savefig("par2_plot3.jpg",bbox_inches='tight', dpi=600 )

sns.lmplot(data=df, x="curb-weight", y="price", height=6, aspect=1.5)
plt.savefig("par2_plot3.jpg",bbox_inches='tight', dpi=600 )

sns.lmplot(data=df, x="horsepower", y="price", hue="fuel-type", height=6, aspect=1.5)
plt.savefig("par2_plot4.jpg",bbox_inches='tight', dpi=600 )

###

df = pd.read_csv("/content/ames.csv")
features = ["YearBuilt", "MoSold", "ScreenPorch", "LotFrontage", "GrLivArea", "GarageArea"]
# plt.setp(plot.get_xticklabels(), rotation=90)
a = sns.relplot(
    x="value", y="SalePrice", col="variable",
    data=df.melt(id_vars="SalePrice", value_vars=features), facet_kws=dict(sharex=False),
    col_wrap=3
);
plt.savefig("par2_plot5.jpg",bbox_inches='tight', dpi=600 )

def make_mi_scores(X,y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes] # creates a list of true false, needed for MI
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    return mi_scores.sort_values(ascending=False)

def plot_mi_scores(scores):
    width = np.arange(len(scores)) # how many y values we have
    ticks = list(scores.index) # taking the name of the y values which isthe index of mi_score series
    plt.barh(width,scores, color="tab:blue")
    plt.yticks(width,ticks) # providing the name of the features on the y axis
    plt.title("Mutual Information Scores")
    
plot_mi_scores(mi_scores.head(15))
plt.savefig("par2_plot6.jpg",bbox_inches='tight', dpi=600 )

sns.set(rc={"figure.figsize":(12, 4)})
sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen", height=6, aspect=1.5)
plt.savefig("par2_plot7.jpg",bbox_inches='tight', dpi=600 )

feature = "GrLivArea" # change it to other feature to witness if there is interaction or not

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
)
plt.savefig("par2_plot8.jpg",bbox_inches='tight', dpi=600 )

# Part 3: Creating Features

accidents = pd.read_csv("/content/accidents.csv")
autos = pd.read_csv("/content/autos.csv")
concrete = pd.read_csv("/content/concrete.csv")
customer = pd.read_csv("/content/customer.csv")


accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p) # appliyng log(x+1) for all the values in LongWindSpeed

fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(8,4))
fig.set_size_inches(10, 5)
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0]).set_title("Raw Values")
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]).set_title("Log Scale")
plt.savefig("par3_plot1.jpg",bbox_inches='tight', dpi=600 )

# Creating Features

# By aggregation
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
accidents.head(n=2)

components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer", "CoarseAggregate", "FineAggregate"]

# By splitting certain fatires
customer[["Type", "Level"]] = (customer["Policy"].str.split(" ", expand=True)) # expand let us create two features
customer[["Customer", "Policy", "Type", "Level" ]].head()
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

customer["State_Gender"] = customer["State"] + '_' +customer["Gender"] # Creating new feature from State and Gender
# Do the above if you believe there are informative interaction between these two features(remember you can check by MI and plots)

# Part 4: Creating Features using Clustering method
df = pd.read_csv("/content/housing.csv")
X = df.loc[:,["MedInc", "Latitude", "Longitude"]]
X.head()

kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X['Cluster'] = X['Cluster'].astype('category') # changing type to category or else it is int
plt.figure(figsize=(12,8))
sns.scatterplot(data=X, x="Longitude", y="Latitude", hue="Cluster")

