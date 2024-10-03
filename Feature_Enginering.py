import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("concrete.csv")
df.head()

X = df.copy()
y = X.pop("CompressiveStrength")

# Train and score baseline model
baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")

X = df.copy()
y = X.pop("CompressiveStrength")

# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score model on dataset with additional ratio features
model = RandomForestRegressor(criterion="absolute_error", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-whitegrid")

df = pd.read_csv("autos.csv")
df.head()
df['fuel_type'].value_counts()
df['make'].value_counts()


#The scikit-learn algorithm for MI treats discrete features differently 
# from continuous features. Consequently, you need to tell it which are which.
#  As a rule of thumb, anything that must have a float dtype is not discrete. 
# Categoricals (object or categorial dtype) can be treated as discrete by 
# giving them a label encoding.

X = df.copy()
y = X.pop("price")
X.info()
# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

X['fuel_type'].value_counts()
X['make'].value_counts()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int
X.info()

from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

sns.relplot(x="curb_weight", y="price", data=df);
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

accidents = pd.read_csv("accidents.csv")
autos = pd.read_csv("autos.csv")
concrete = pd.read_csv("concrete.csv")
customer = pd.read_csv("customer.csv")

autos["stroke_ratio"] = autos.stroke / autos.bore
autos[["stroke", "bore", "stroke_ratio"]].head()
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)

# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);

accidents['Amenity'].value_counts()

roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

accidents[roadway_features + ["RoadwayFeatures"]].head(10)


components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

concrete[components + ["Components"]].head(10)



customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

customer[["Policy", "Type", "Level"]].head(10)

autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()

customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

customer[["State", "Income", "AverageIncome"]].head(10)

customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)

customer[["State", "StateFreq"]].head(10)

# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

df_valid[["Coverage", "AverageClaim"]].head(10)


#Target Encoding
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")

autos[["make", "price", "make_encoded"]].head(10)


import featuretools as ft
data = ft.demo.load_mock_customer()

customers_df = data["customers"]
customers_df

sessions_df = data["sessions"]
sessions_df.sample(5)

transactions_df = data["transactions"]
transactions_df.sample(5)

# specify a dictionary containing each Dataframe in the dataset.
# If the dataset has an “id” column, we pass it along with the DataFrames

dataframes = {
    "customers": (customers_df, "customer_id"),
    "sessions": (sessions_df, "session_id", "session_start"),
    "transactions": (transactions_df, "transaction_id", "transaction_time"),
}

# we define the connections between the Dataframes.
# In this example we have two relationships:
relationships = [
    ("sessions", "session_id", "transactions", "session_id"),
    ("customers", "customer_id", "sessions", "customer_id"),
]

# we can generate features through DFS, 
# which requires three basic inputs:
# “DataFrames”, “Relationship list” and “Target DataFrame name”:

feature_matrix_customers, features_defs = ft.dfs(
    dataframes=dataframes,
    relationships=relationships,
    target_dataframe_name="customers",
)
feature_matrix_customers

feature = features_defs[18]
feature


### Another example
import featuretools as ft
import numpy as np
import pandas as pd

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")

print('train size', train.shape)
print('test size', test.shape)

# saving identifiers
test_Item_Identifier = test['Item_Identifier']
test_Outlet_Identifier = test['Outlet_Identifier']
sales = train['Item_Outlet_Sales']
train.drop(['Item_Outlet_Sales'], axis=1, inplace=True)

combi = train.append(test, ignore_index=True)
combi.isnull().sum()

# imputing missing data
combi['Item_Weight'].fillna(combi['Item_Weight'].mean(), inplace = True)
combi['Outlet_Size'].fillna("missing", inplace = True)


combi['Item_Fat_Content'].value_counts()

# dictionary to replace the categories
fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}
combi['Item_Fat_Content'] = combi['Item_Fat_Content'].replace(fat_content_dict, regex=True)

combi['id'] = combi['Item_Identifier'] + combi['Outlet_Identifier']
combi.drop(['Item_Identifier'], axis=1, inplace=True)

# creating and entity set 'es'
es = ft.EntitySet(id = 'sales')

# adding a dataframe 
es.add_dataframe(dataframe_name='mart',dataframe = combi, index = 'id')

es.normalize_dataframe(base_dataframe_name='mart', new_dataframe_name='outlet', index = 'Outlet_Identifier', 
additional_columns = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

combi['Outlet_Identifier'].value_counts()
    
feature_matrix, feature_names = ft.dfs(entityset=es, 
target_dataframe_name = 'mart', 
max_depth = 2, 
verbose = 1, 
n_jobs = 1)
     
feature_matrix.columns
feature_matrix.head()

combi.info()

#################  Featurewiz #########
##### SULOV ####
### Searching for the uncorrelated list of variables (SULOV):
##This method searches the uncorrelated list of variables
#to identify valid variable pairs, it considers the variable pair
#  with the lowest correlation and maximum MIS (Mutual Information Score)
#  rating for further processing.

##########Recursive XGBoost ########
#The variables identified in SULOV in the previous step are 
# recursively passed to XGBoost, and the features most relevant
#  to the target column are selected through XGBoost, combined, 
# and added as new features, and this process is iterated until 
# all valid features are generated.

from featurewiz import FeatureWiz
import pandas as pd
import os

dataset = pd.read_csv("adult.csv")

RANDOM_SEED = 99
target = 'income'
preds1 = [x for x in list(dataset) if x != target]
len(preds1)
type(preds1)
preds = ['age','occupation','marital-status']

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED,)
print(train.shape, test.shape)

dft = FeatureWiz(train, target, corr_limit=0.70,verbose=1, test_data=test,
                      feature_engg=["groupby",'target','interactions'], 
                     category_encoders=''
                     )

newX=dataset.drop('income',axis=1)
newX=dataset[preds]
newY=dataset['income']
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)


features = FeatureWiz(corr_limit=0.40, feature_engg=["groupby",'target','interactions'], category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=2)
X_train_selected = features.fit_transform(X_train, y_train)
X_test_selected = features.transform(X_test)
features.features  # the selected feature list #
# automated feature generation
import featurewiz as FW
outputs = FW.featurewiz(dataname=train, target=target, corr_limit=0.50, verbose=2, sep=',', 
                header=0, test_data='',feature_engg=['interactions','groupby','target'], category_encoders='',
                dask_xgboost_flag=False, nrows=None)





