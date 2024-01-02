## Developing the model ###

# Load Libraries
import pandas as pd

# Import data
kickstarter_df = pd.read_excel("Kickstarter.xlsx")

# Pre-Processing
kickstarter_df = kickstarter_df.dropna()
# Calculating the USD goal amount
kickstarter_df["goal_usd"] = kickstarter_df["goal"] * kickstarter_df["static_usd_rate"]
# Keeping only rows where state is either 'successful' or 'failed'
kickstarter_df = kickstarter_df[kickstarter_df["state"].isin(["successful", "failed"])]
# Dropping columns that are not available at the point of lauch date or irrelevant of the analysis
columns_to_drop = [
    "disable_communication", "country", "static_usd_rate", "goal", "name", "id", 
    "deadline", "state_changed_at", "created_at", "launched_at", "pledged", 
    "backers_count", "usd_pledged", "spotlight", "staff_pick", 
    "launch_to_state_change_days", "state_changed_at_weekday", 
    "state_changed_at_month", "state_changed_at_yr", "state_changed_at_hr", 
    "state_changed_at_day"]
kickstarter_df.drop(columns_to_drop, axis=1, inplace=True)
# List of categorical columns for one-hot encoding
category_list = [
    "state", "category", "deadline_weekday", "created_at_weekday", 
    "launched_at_weekday", "deadline_month", "deadline_day", "deadline_yr", 
    "created_at_month", "created_at_day", "created_at_yr", "created_at_hr", 
    "launched_at_month", "launched_at_day", "launched_at_yr", "launched_at_hr", 
    "currency"]
# Creating dummy variables
kickstarter_df_dummies = pd.get_dummies(kickstarter_df[category_list], drop_first=True)
kickstarter_df = pd.concat([kickstarter_df, kickstarter_df_dummies], axis=1)
kickstarter_df.drop(category_list, axis=1, inplace=True)

# Setup the variables

y = kickstarter_df["state_successful"]
X = kickstarter_df.drop(["state_successful"], axis=1)

# List of features to drop based on Lasso regression
Lasso_useless_features = [
    'category_Wearables', 'deadline_weekday_Saturday', 'deadline_weekday_Wednesday', 
    'created_at_weekday_Sunday', 'created_at_weekday_Thursday', 'launched_at_weekday_Saturday'
]
X.drop(Lasso_useless_features, axis=1, inplace=True)

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

# Build the model
from sklearn.ensemble import RandomForestClassifier
#best hyperparameters tested
rf = RandomForestClassifier(n_estimators=200, max_features=35, random_state=5)
model = rf.fit(X_train, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)


## Grading ##

# Import Grading Data
kickstarter_grading_df = pd.read_excel("Kickstarter-Grading-Sample.xlsx")

# Pre-Process Grading Data
kickstarter_grading_df = kickstarter_grading_df.dropna()
# Calculate goal in USD
kickstarter_grading_df["goal_usd"] = kickstarter_grading_df["goal"] * kickstarter_grading_df["static_usd_rate"]

# Drop rows with missing values
kickstarter_grading_df.dropna(inplace=True)

# Keep only 'successful' and 'failed' states
kickstarter_grading_df = kickstarter_grading_df[kickstarter_grading_df["state"].isin(["successful", "failed"])]
# Drop columns that are not needed
columns_to_drop = ["disable_communication", "country", "static_usd_rate", "goal", "name", "id", 
                   "deadline", "state_changed_at", "created_at", "launched_at", "pledged", 
                   "backers_count", "usd_pledged", "spotlight", "staff_pick", 
                   "launch_to_state_change_days", "state_changed_at_weekday", 
                   "state_changed_at_month", "state_changed_at_yr", "state_changed_at_hr", 
                   "state_changed_at_day"]
kickstarter_grading_df.drop(columns_to_drop, axis=1, inplace=True)

# One-Hot Encoding for Categorical Variables
kickstarter_grading_df_dummies = pd.get_dummies(kickstarter_grading_df[category_list], drop_first=True)
kickstarter_grading_df = pd.concat([kickstarter_grading_df, kickstarter_grading_df_dummies], axis=1)
kickstarter_grading_df.drop(category_list, axis=1, inplace=True)

# Setup the variables
Lasso_useless_features=['category_Wearables', 'deadline_weekday_Saturday', 'deadline_weekday_Wednesday', 'created_at_weekday_Sunday', 'created_at_weekday_Thursday', 'launched_at_weekday_Saturday']
X_grading = kickstarter_grading_df.drop(["state_successful"], axis=1)
y_grading = kickstarter_grading_df["state_successful"]
# Drop features identified as less useful by Lasso regression
X_grading.drop(Lasso_useless_features, axis=1, inplace=True)


# Apply the model previously trained to the grading data
y_grading_pred = model.predict(X_grading)

# Calculate the accuracy score
accuracy_score(y_grading, y_grading_pred)



##############Task 2: Clsuter Model#####################
#df=pd.read_excel("Kickstarter.xlsx")
#df["goal_usd"]=df["goal"]*df["static_usd_rate"]
#df.dropna(subset=['category'], inplace=True)
#dropped1392 rows with Non category
#df.dropna(subset=['name_len_clean'], inplace=True)
#df.dropna(subset=['blurb_len'], inplace=True)
#df.dropna(subset=['blurb_len_clean'], inplace=True)
#drop rows with state!=successful or failed
#df = df[df["state"].isin(["successful", "failed"])]
#df.drop(["static_usd_rate","goal","name","id","deadline",'state_changed_at',
#       'created_at', 'launched_at'],axis=1,inplace=True)
#category_list=["state","disable_communication","category","deadline_weekday","created_at_weekday","launched_at_weekday","deadline_month","deadline_day","deadline_yr","deadline_hr",'created_at_month',
#       'created_at_day', 'created_at_yr', 'created_at_hr', 'launched_at_month',
#       'launched_at_day', 'launched_at_yr', 'launched_at_hr',"country","currency", "state_changed_at_weekday","state_changed_at_month",
#       'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr',"pledged","staff_pick","spotlight"]

#non_categorical_columns = df.columns.difference(category_list)
#new_df=df.drop(category_list, axis=1)
#drop row 12197 since it is the only data point in cluster2
#new_df.drop([12197], axis=0, inplace=True)
#from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
#scaled_df = scaler.fit_transform(new_df)  #apply scaler to our dataset using transform
#scaled_df = pd.DataFrame(new_df, columns=new_df.columns) # transform into a dataframe and add column names
#scaled_df.describe()
#from sklearn.cluster import KMeans
#withinss = []
#for i in range (2,10):    #testing k: sqrt(n/2)
#    kmeans = KMeans(n_clusters=i)
#    model = kmeans.fit(scaled_df)
#    withinss.append(model.inertia_)
#from matplotlib import pyplot
#pyplot.plot(range(2,10),withinss)
#plt.title('Inertia vs Number of clusters')
#plt.xlabel('Number of clusters')
#plt.ylabel('Inertia')
#from sklearn.metrics import silhouette_score

#silhouette_scores = {}
#for n_clusters in range(2, 10):  #  test for 2 to 10 clusters
#    kmeans = KMeans(n_clusters=n_clusters)
#    cluster_labels = kmeans.fit_predict(scaled_df)
#    score = silhouette_score(scaled_df, cluster_labels)
#    silhouette_scores[n_clusters] = score

#best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
#best_score = silhouette_scores[best_n_clusters]

#print(f"For n_clusters = {best_n_clusters}, the best average silhouette score is: {best_score}")

#from sklearn.metrics import calinski_harabasz_score
#F_scores = {}
#for n_clusters in range(2, 10):  # Assuming we test for 2 to 10 clusters
#    kmeans = KMeans(n_clusters=n_clusters)
#    cluster_labels = kmeans.fit_predict(scaled_df)
#    score = calinski_harabasz_score(scaled_df, cluster_labels)
#    F_scores[n_clusters] = score

#best_n_clusters = max(F_scores, key=F_scores.get)
#best_score = F_scores[best_n_clusters]

#print(f"For n_clusters = {best_n_clusters}, the best F score is: {best_score}")

#kmeans = KMeans(n_clusters=4)
#model = kmeans.fit(scaled_df)
#model.inertia_
#scaled_df['cluster'] = model.labels_
#cluster1 = scaled_df[scaled_df['cluster'] == 0]
#cluster2 = scaled_df[scaled_df['cluster'] == 1]
#cluster3 = scaled_df[scaled_df['cluster'] == 2]
#cluster4 = scaled_df[scaled_df['cluster'] == 3]

#import matplotlib.pyplot as plt
#import seaborn as sns


# Extracting only the numerical columns (excluding the 'cluster' column)
#numerical_columns = scaled_df.columns.drop('cluster')

# Define a color map for the clusters
#colors = ['red', 'green', 'blue', 'purple']

# Creating scatter plots for each pair of numerical columns
#for i in range(len(numerical_columns)):
#    for j in range(i + 1, len(numerical_columns)):
#        plt.figure(figsize=(10, 6))
        # Plot each cluster with a different color
#       for cluster_num in range(4):
#            cluster_data = scaled_df[scaled_df['cluster'] == cluster_num]
#            plt.scatter(cluster_data[numerical_columns[i]], cluster_data[numerical_columns[j]], color=colors[cluster_num], label=f'Cluster {cluster_num}')

#        plt.xlabel(numerical_columns[i])
#        plt.ylabel(numerical_columns[j])
#        plt.title(f'Scatter plot of {numerical_columns[i]} vs {numerical_columns[j]}')
#        plt.legend()
#        plt.show()

