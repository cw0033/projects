df = read.csv("Gun_violence.csv")
summary(df)
View(df)
############################dropping non_used columns##########################
# List of columns to drop
columns_to_drop <- c("incident_url", "sources", "source_url","incident_url_fields_missing","gun_stolen","location_description","participant_relationship","participant_name","state_house_district","congressional_district","state_senate_district")
df <- select(df, -one_of(columns_to_drop))
na_count <- df %>%
  summarize_each(funs(sum(is.na(.))))
print(na_count)
############################# drop all na values###############################
# Convert blank values to NA
df[df == ""] <- NA
df<- na.omit(df)


###########################Only work on Illinois################################
library(dplyr)
df_Illinois <- df %>% 
  filter(state == "Illinois")

View(df_Illinois)

####################### clustering longitude and latitude ######################
coordinates <- df_Illinois[, c("longitude", "latitude")]

# Determine the maximum number of clusters to consider
max_clusters <- 20

# hold the Withiness for each k
wss_values <- numeric(max_clusters)

# Loop over 1 to max_clusters to compute WSS for each k
for (k in 1:max_clusters) {
  set.seed(123) 
  kmeans_result <- kmeans(coordinates, centers = k)
  wss_values[k] <- kmeans_result$tot.withinss
}

# Plot the Elbow Plot
plot(1:max_clusters, wss_values, type = "b", xlab = "Number of Clusters", ylab = "Total Within-Cluster Sum of Squares", main = "Elbow Method for Determining Optimal k")

######add column of location cluster########
# Assuming coordinates are already defined and cleaned from NA values
set.seed(123)  
optimal_k <- 4

# Perform K-means clustering with 5 clusters
kmeans_result <- kmeans(coordinates, centers = optimal_k)

# Add the cluster assignment to dataframe
df_Illinois$cluster <- kmeans_result$cluster
head(df_Illinois)


#######################Count number of suspects in each row(incident)##########
# Function to count occurrences of "Subject-Suspect"
count_subject_suspect <- function(s) {
  sum(grepl("Subject-Suspect", unlist(strsplit(s, "\\|\\|"))))
}

# Apply this function to each row of the participant_type column
df_Illinois$n_subject_suspect <- sapply(df_Illinois$participant_type, count_subject_suspect)




#####################Dealing with participant_age_group#########################

df_Illinois <- df_Illinois %>%
  mutate(teens_involved = ifelse(grepl("Teen", participant_age_group), 1, 0))


############Dealing with participant type and age-fetch victims' avg age of each case########
# Function to calculate the average age based on participant type
calculate_average_age_fixed_v2 <- function(types_str, ages_str, target_type) {
  # Splitting the strings into arrays
  types <- unlist(strsplit(types_str, "\\|\\|"))
  ages <- unlist(strsplit(ages_str, "\\|\\|"))
  
  # Extracting individual types and ages
  type_info <- lapply(types, function(x) strsplit(x, "::")[[1]])
  age_info <- lapply(ages, function(x) strsplit(x, "::")[[1]])
  
  # Ensuring equal length by trimming the longer list
  min_length <- min(length(type_info), length(age_info))
  type_info <- type_info[1:min_length]
  age_info <- age_info[1:min_length]
  
  # Selecting ages based on the target type
  valid_ages <- sapply(1:min_length, function(i) {
    if(length(type_info[[i]]) > 1 && length(age_info[[i]]) > 1 && type_info[[i]][2] == target_type) {
      return(as.numeric(age_info[[i]][2]))
    }
    return(NA)
  })
  
  # Returning the average age, handling cases with no selected ages
  valid_ages <- valid_ages[!is.na(valid_ages)]
  if (length(valid_ages) > 0) mean(valid_ages) else NA
}


# Adding the 'victim_age' column
df_Illinois$victim_age <- mapply(calculate_average_age_fixed_v2, 
                                 df_Illinois$participant_type, 
                                 df_Illinois$participant_age, 
                                 MoreArgs = list(target_type = "Victim"))

# Adding the 'non_victim_age' column
df_Illinois$non_victim_age <- mapply(calculate_average_age_fixed_v2, 
                                     df_Illinois$participant_type, 
                                     df_Illinois$participant_age, 
                                     MoreArgs = list(target_type = "Subject-Suspect"))



table_victim_age <- table(is.na(df_Illinois$victim_age))
print("NA summary for victim_age:")
print(table_victim_age)
#######1161null value in victim age; 7089 non_nulls
table_non_victim_age <- table(is.na(df_Illinois$non_victim_age))
print("NA summary for non_victim_age:")
print(table_non_victim_age)
#######62665null values in non_victim_age, 1986 non-nulls

#########drop non_victim_age and na values in col victim_age#######
library(dplyr)
columns_to_drop2<-c("address","latitude","longitude","participant_age","participant_age_group","participant_status","non_victim_age","participant_gender","participant_type","city_or_county","state")
df_Illinois <- df_Illinois %>% 
  select(-columns_to_drop2)
df_Illinois <- df_Illinois[!is.na(df_Illinois$victim_age), ]


#######drop non_victim_age column, since there are too many null values#########
################################################################################



#####developing new columns: drug_involed and gang_involved from "incident_characteristics","notes"######
# drug_involved column
df_Illinois$drug_involved <- apply(df_Illinois[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("drug", x, ignore.case = TRUE))
})

# gang_involved column
df_Illinois$gang_involved <- apply(df_Illinois[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("gang", x, ignore.case = TRUE))
})


#################Add column: Bar/School/Drive-by/Home Invasion#################

df_Illinois$bar <- apply(df_Illinois[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("bar", x, ignore.case = TRUE))
})
df_Illinois$home_invasion <- apply(df_Illinois[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("home invasion", x, ignore.case = TRUE))
})
df_Illinois$school <- apply(df_Illinois[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("school", x, ignore.case = TRUE))
})
df_Illinois$drive_by <- apply(df_Illinois[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("drive-by", x, ignore.case = TRUE))
})


columns_to_drop3<-c("notes","gun_type","incident_characteristics","incident_id")
df_Illinois <- df_Illinois %>% 
  select(-columns_to_drop3)



##########derive new columns : month and weekday from date col#################

library(lubridate)

# Convert the date column to Date type if it's not already
df_Illinois$date <- as.Date(df_Illinois$date)

# Extract month and weekday
df_Illinois$month <- month(df_Illinois$date, label = TRUE)  
df_Illinois$weekday <- wday(df_Illinois$date, label = TRUE) 

df_Illinois <- df_Illinois %>% 
  select(-date)

###################end of dataframe creation for df_Illinois###################
###############################################################################

##############################start building model#############################


##dummify month and weekday; transform n_killed to "fatal"
df_Illinois$month <- as.factor(df_Illinois$month)
df_Illinois$weekday <- as.factor(df_Illinois$weekday)
df_Illinois$fatal <- ifelse(df_Illinois$n_killed > 0, 1, 0)
df_Illinois <- df_Illinois %>% 
#  select(-n_killed)
library(dplyr)
df_Illinois <- df_Illinois %>% mutate(fatal = as.factor(fatal))


#####run random forest and find the best number of trees without feature selection###
library(randomForest)
library(caret)
set.seed(123) 
splitIndex <- createDataPartition(df_Illinois$fatal, p = 0.7, list = FALSE)
train_data <- df_Illinois[splitIndex, ]
test_data <- df_Illinois[-splitIndex, ]

# Loop over ntrees

rf_results <- list()

for (i in seq(100, 500, by=100)) {
  set.seed(123)  
  rf <- randomForest(fatal ~ n_injured + n_guns_involved + cluster + n_subject_suspect +
                       teens_involved + victim_age + drug_involved + gang_involved +
                       bar + home_invasion + school + drive_by + month + weekday,
                     ntree=i, data=train_data, importance=TRUE, na.action=na.omit)
  
  predictions <- predict(rf, newdata = test_data)
  cm <- confusionMatrix(predictions, test_data$fatal)
  
  # Store the model's confusion matrix and accuracy in the list
  rf_results[[paste("ntree", i)]] <- list(
    confusion_matrix = cm$table,
    accuracy = cm$overall['Accuracy']
  )
}

# Print Accuracy for each model
for (i in seq(100, 500, by=100)) {
  cat("Number of trees:", i, "\n")
  cat("Confusion Matrix:\n")
  print(rf_results[[paste("ntree", i)]]$confusion_matrix)
  cat("Accuracy:", rf_results[[paste("ntree", i)]]$accuracy, "\n\n")
}

#############plot accuracy vs ntrees############
results_df <- data.frame(
  num_trees = seq(100, 500, by = 100),
  accuracy = sapply(rf_results, function(x) x$accuracy)
)

# Plot
ggplot(results_df, aes(x = num_trees, y = accuracy)) +
  geom_line() +  # Line plot
  geom_point() +  # Add points
  xlab("Number of Trees") +
  ylab("Accuracy") +
  ggtitle("Accuracy vs Number of Trees in RandomForest")




#####feature importance from rf model#######
rf_150 <- randomForest(fatal ~ n_injured + n_guns_involved + cluster + n_subject_suspect + teens_involved + victim_age + 
                         drug_involved + gang_involved + bar + home_invasion + school + drive_by + month + weekday, 
                       ntree=150, data=df_Illinois, importance=TRUE, na.action=na.omit)
importance(rf_150)
varImpPlot(rf_150)
#####result from rf importance score: n_injured,n_subject_suspect, n_guns_involved, teens_involved, victim_age,drug_involved, bar, home invasion are more important####


##PCA 

# columns to include in PCA
vars <- c("n_injured", "n_guns_involved", "n_subject_suspect", "victim_age")

# Selecting the specified columns and ensuring they are numeric
df_selected <- df_Illinois[, vars]
df_selected <- data.frame(lapply(df_selected, as.numeric))

# Handling missing values with na.omit 
df_selected <- na.omit(df_selected)

# Perform PCA
pca_result <- prcomp(df_selected, scale = TRUE)

# Print summary of PCA
summary(pca_result)

##plot pca
# Proportion of Variance Explained
pve <- pca_result$sdev^2 / sum(pca_result$sdev^2)
par(mfrow=c(1,2))
plot(pve,ylim=c(0,1))
plot(cumsum(pve),ylim=c(0,1))



########test OOB in rf n_trees=50~500###########
rf <- randomForest(fatal ~ n_injured + n_guns_involved + cluster + n_subject_suspect + teens_involved + victim_age + 
                         drug_involved + gang_involved + bar + home_invasion + school + drive_by + month + weekday, 
                       ntree=500, data=df_Illinois, importance=TRUE, na.action=na.omit,do.trace=50)
###########result:best tree is also ntree=150 using OOB to evaluate#######

library(randomForest)
library(caret) # for additional model performance metrics

# Fit the first random forest model
rf <- randomForest(fatal ~ n_injured + n_guns_involved + cluster + n_subject_suspect + 
                     teens_involved + victim_age + drug_involved + gang_involved + bar + 
                     home_invasion + school + drive_by + month + weekday,
                   ntree=150, data=df_Illinois, importance=TRUE, na.action=na.omit)

# Fit the second random forest model with fewer features
rf_feature <- randomForest(fatal ~ n_injured + n_subject_suspect + 
                             teens_involved + victim_age + 
                             bar + home_invasion, 
                           ntree=150, data=df_Illinois, importance=TRUE, na.action=na.omit)

# Compare OOB error rates
oob_error_rf <- rf$err.rate[150, 'OOB']
oob_error_rf_feature <- rf_feature$err.rate[150, 'OOB']

# Print the OOB error rates
print(paste("OOB error rate for rf:", oob_error_rf))
print(paste("OOB error rate for rf_feature:", oob_error_rf_feature))

#  the outcome 'fatal' is a factor (classification task)
# get confusion matrix for each model:
confusion_rf <- rf$confusion
confusion_rf_feature <- rf_feature$confusion

# Print confusion matrices
print("Confusion matrix for rf:")
print(confusion_rf)
print("Confusion matrix for rf_feature:")
print(confusion_rf_feature)

# Calculate accuracy for each model
accuracy_rf <- sum(diag(confusion_rf)) / sum(confusion_rf)
accuracy_rf_feature <- sum(diag(confusion_rf_feature)) / sum(confusion_rf_feature)

# Print the accuracy for each model
print(paste("Accuracy for rf:", accuracy_rf))
print(paste("Accuracy for rf_feature:", accuracy_rf_feature))










###########################################################################
#####################prediction model:n_killed#############################
##########################################################################

rf_results <- list()

for (i in seq(100, 500, by=100)) {
  set.seed(123)
  rf_n <- randomForest(n_killed ~ n_injured + n_guns_involved + cluster + n_subject_suspect + teens_involved + victim_age + 
                       drug_involved + gang_involved + bar + home_invasion + school + drive_by + month + weekday, 
                     ntree=i, data=df_Illinois, importance=TRUE, na.action=na.omit)
  
  # Store the model and its final MSE and %Var explained in the list
  rf_results[[paste("ntree", i)]] <- list(
    mse_final = rf_n$mse[length(rf_n$mse)],
    var_explained_final = rf_n$rsq[length(rf_n$rsq)]
  )
}

# Print MSE and %Var explained for each model
for (i in seq(100, 500, by=100)) {
  cat("Number of trees:", i, "\n")
  cat("Mean of squared residuals:", rf_results[[paste("ntree", i)]]$mse_final, "\n")
  cat("% Var explained:", rf_results[[paste("ntree", i)]]$var_explained_final * 100, "%\n")
  cat("\n")
}
##################result: best tree, ntree=400, MSE:0.09915248, R-squared: 59.47005 %##############
#use OOB to find best tree
rf_n <- randomForest(n_killed ~ n_injured + n_guns_involved + cluster + n_subject_suspect + teens_involved + victim_age + 
                       drug_involved + gang_involved + bar + home_invasion + school + drive_by + month + weekday, 
                     ntree=500, data=df_Illinois, importance=TRUE, na.action=na.omit,do.trace=50)
#############result:ntree=400 is the best tree, MSE:0.09909 , r-squared:40.51#########

#####feature importance from rf_n model#######
rf_n_400 <- randomForest(n_killed ~ n_injured + n_guns_involved + cluster + n_subject_suspect + teens_involved + victim_age + 
                         drug_involved + gang_involved + bar + home_invasion + school + drive_by + month + weekday, 
                       ntree=400, data=df_Illinois, importance=TRUE, na.action=na.omit)
importance(rf_n_400)
varImpPlot(rf_n_400)

###########model performance of rf_n_400 and fr_n_400_feature###################
####cross validation####
library(caret)

# Assuming df_Illinois is your full dataset and n_killed is the response variable
set.seed(123) # for reproducibility

# Create indices to split the data into training and test set
# Let's say we want to keep 80% of the data for training and 30% for testing
trainIndex <- createDataPartition(df_Illinois$n_killed, p = .7, 
                                  list = FALSE, 
                                  times = 1)

# Create the training and test sets
trainData <- df_Illinois[trainIndex, ]
testData <- df_Illinois[-trainIndex, ]

#train random forest models on the training set
rf_n_400 <- randomForest(n_killed ~ n_injured + n_guns_involved + cluster + n_subject_suspect + 
                           teens_involved + victim_age + drug_involved + gang_involved + 
                           bar + home_invasion + school + drive_by + month + weekday, 
                         ntree=400, data=trainData, importance=TRUE, na.action=na.omit)

rf_n_400_feature <- randomForest(n_killed ~ n_injured + n_subject_suspect + teens_involved + 
                                   victim_age + bar + home_invasion, 
                                 ntree=400, data=trainData, importance=TRUE, na.action=na.omit)

# Evaluate the performance 
predictions_rf_n_400 <- predict(rf_n_400, testData)
predictions_rf_n_400_feature <- predict(rf_n_400_feature, testData)

# Calculate RMSE 
rmse_rf_n_400 <- sqrt(mean((testData$n_killed - predictions_rf_n_400)^2))
rmse_rf_n_400_feature <- sqrt(mean((testData$n_killed - predictions_rf_n_400_feature)^2))

# Calculate MSE 
mse_rf_n_400 <- mean((testData$n_killed - predictions_rf_n_400)^2)
mse_rf_n_400_feature <- mean((test_data$n_killed - predictions_rf_n_400_feature)^2)

# Calculate MAE 
mae_rf_n_400 <- mean(abs(testData$n_killed - predictions_rf_n_400))
mae_rf_n_400_feature <- mean(abs(testData$n_killed - predictions_rf_n_400_feature))

# Calculate R-squared 
r_squared_rf_n_400 <- cor(testData$n_killed, predictions_rf_n_400)^2
r_squared_rf_n_400_feature <- cor(testData$n_killed, predictions_rf_n_400_feature)^2

# Print the metrics
cat("Model with all features:\n")
cat("RMSE:", rmse_rf_n_400, "\n")
cat("MSE:", mse_rf_n_400, "\n")
cat("MAE:", mae_rf_n_400, "\n")
cat("R-squared:", r_squared_rf_n_400, "\n\n")

cat("Model with selected features:\n")
cat("RMSE:", rmse_rf_n_400_feature, "\n")
cat("MSE:", mse_rf_n_400_feature, "\n")
cat("MAE:", mae_rf_n_400_feature, "\n")
cat("R-squared:", r_squared_rf_n_400_feature, "\n")

# stoe everything into df

# named list for each model with its metrics
metrics_all_features <- list(
  RMSE = rmse_rf_n_400,
  MSE = mse_rf_n_400,
  MAE = mae_rf_n_400,
  R_squared = r_squared_rf_n_400
)

metrics_selected_features <- list(
  RMSE = rmse_rf_n_400_feature,
  MSE = mse_rf_n_400_feature,
  MAE = mae_rf_n_400_feature,
  R_squared = r_squared_rf_n_400_feature
)

# Combine the lists into a data frame
performance_comparison <- data.frame(
  All_Features = unlist(metrics_all_features),
  Selected_Features = unlist(metrics_selected_features)
)

# Transpose the data frame so that the metrics are rows and models are columns
performance_comparison <- t(performance_comparison)
colnames(performance_comparison) <- c("RMSE", "MSE", "MAE", "R_squared")

# Convert the matrix back to a data frame for printing
performance_comparison <- as.data.frame(performance_comparison)

# Print the data frame
print(performance_comparison)

# Compare OOB error rates
rf_n_400 <- randomForest(n_killed ~ n_injured + n_guns_involved + cluster + n_subject_suspect + teens_involved + victim_age + 
                           drug_involved + gang_involved + bar + home_invasion + school + drive_by + month + weekday, 
                         ntree=400, data=df_Illinois, importance=TRUE, na.action=na.omit,do.trace=50)
rf_n_400_feature <- randomForest(n_killed ~ n_injured + n_subject_suspect + teens_involved + victim_age + 
                            bar + home_invasion, 
                         ntree=400, data=df_Illinois, importance=TRUE, na.action=na.omit,do.trace=50)

