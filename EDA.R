df = read.csv("/Users/emilywu/Desktop/MMA/MGSC 661/Final Project/Gun_violence.csv")
summary(df)
View(df)

#############count state incident#########
# Load necessary libraries
library(ggplot2)
library(dplyr)

# Count the number of rows for each state
state_counts <- df %>% 
  group_by(state) %>% 
  summarise(count = n())

# Create a bar chart
ggplot(state_counts, aes(x = state, y = count)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "State", y = "Count", title = "Incidents in Each State")

########count n_killed by state#########
# Load the dplyr package

# Calculate the average number of people killed by state
average_killed_by_state <- df %>%
  group_by(state) %>%
  summarise(average_killed = mean(n_killed, na.rm = TRUE))

ggplot(average_killed_by_state, aes(x = state, y = average_killed)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "State", y = "Average Number Killed", title = "Average Number of People Killed by State")

###########average n_injured by state###########
average_injured_by_state <- df %>%
  group_by(state) %>%
  summarise(average_injured = mean(n_injured, na.rm = TRUE))

ggplot(average_injured_by_state, aes(x = state, y = average_injured)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "State", y = "Average Number Injured", title = "Average Number of People Injured by State")


############################dropping non_used columns##########################
# List of columns you want to drop
columns_to_drop <- c("incident_url", "sources", "source_url","incident_url_fields_missing","gun_stolen","location_description","participant_relationship","participant_name")
# Dropping the columns

df <- select(df, -one_of(columns_to_drop))
na_count <- df %>%
  summarize_each(funs(sum(is.na(.))))

print(na_count)
############################# drop all na values###############################
# Convert blank values to NA
df[df == ""] <- NA
# Now use na.omit() to remove rows with NA values
df<- na.omit(df)
# Optionally, check the dimensions of the cleaned dataframe
print(dim(df))




####################### clustering longitude and latitude ######################
coordinates <- df[, c("longitude", "latitude")]

# Determine the maximum number of clusters you want to consider
max_clusters <- 20

# Initialize a vector to hold the Withiness for each k
wss_values <- numeric(max_clusters)

# Loop over 1 to max_clusters to compute WSS for each k
for (k in 1:max_clusters) {
  set.seed(123) # for reproducibility
  kmeans_result <- kmeans(coordinates, centers = k)
  wss_values[k] <- kmeans_result$tot.withinss
}

# Plot the Elbow Plot
plot(1:max_clusters, wss_values, type = "b", xlab = "Number of Clusters", ylab = "Total Within-Cluster Sum of Squares", main = "Elbow Method for Determining Optimal k")

######add column of location cluster########
# Assuming coordinates are already defined and cleaned from NA values
set.seed(123)  # for reproducibility
optimal_k <- 4

# Perform K-means clustering with 5 clusters
kmeans_result <- kmeans(coordinates, centers = optimal_k)

# Add the cluster assignment to your dataframe
df$cluster <- kmeans_result$cluster

# Check the first few rows of the modified dataframe
head(df)

#######################Count number of suspects in each row(incident)##########
# Function to count occurrences of "Subject-Suspect"
count_subject_suspect <- function(s) {
  sum(grepl("Subject-Suspect", unlist(strsplit(s, "\\|\\|"))))
}

# Apply this function to each row of the participant_type column
df$subject_suspect_count <- sapply(df$participant_type, count_subject_suspect)





#####################Dealing with participant_age_group#########################

df <- df %>%
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
df$victim_age <- mapply(calculate_average_age_fixed_v2, 
                                 df$participant_type, 
                                 df$participant_age, 
                                 MoreArgs = list(target_type = "Victim"))

# Adding the 'non_victim_age' column
df$non_victim_age <- mapply(calculate_average_age_fixed_v2, 
                                     df$participant_type, 
                                     df$participant_age, 
                                     MoreArgs = list(target_type = "Subject-Suspect"))




table_victim_age <- table(is.na(df$victim_age))
print("NA summary for victim_age:")
print(table_victim_age)
#######1161null value in victim age; 7089 non_nulls
table_non_victim_age <- table(is.na(df$non_victim_age))
print("NA summary for non_victim_age:")
print(table_non_victim_age)
#######62665null values in non_victim_age, 1986 non-nulls

#########drop non_victim_age and na values in col victim_age#######

columns_to_drop2<-c("address","latitude","longitude","participant_age","participant_age_group","participant_status","participant_gender","participant_type")
df <- df %>% 
  select(-columns_to_drop2)
df <- df[!is.na(df$victim_age), ]


#######drop non_victim_age column, since there are too many null values#########
################################################################################



#####developing new columns: drug_involed and gang_involved from "incident_characteristics","notes"######
# For the drug_involved column
df$drug_involved <- apply(df[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("drug", x, ignore.case = TRUE))
})

# For the gang_involved column
df$gang_involved <- apply(df[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("gang", x, ignore.case = TRUE))
})


#################Add column: Bar/School/Drive-by/Home Invasion#################

df$bar <- apply(df[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("bar", x, ignore.case = TRUE))
})
df$home_invasion <- apply(df[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("home invasion", x, ignore.case = TRUE))
})
df$school <- apply(df[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("school", x, ignore.case = TRUE))
})
df$drive_by <- apply(df[, c("incident_characteristics", "notes")], 1, function(x) {
  any(grepl("drive-by", x, ignore.case = TRUE))
})

columns_to_drop3<-c("notes","gun_type","incident_characteristics","incident_id")

df <- df %>% 
  select(-columns_to_drop3)



##########derive new columns : month and weekday from date col#################
install.packages("lubridate")
library(lubridate)
# Assuming your dataframe is named df_Illinois and your date column is named "date"

# Convert the date column to Date type if it's not already
df$date <- as.Date(df$date)

# Extract month and weekday
df$month <- month(df$date, label = TRUE)  # label = TRUE will give month names instead of numbers


df$weekday <- wday(df$date, label = TRUE) # label = TRUE will give weekday names instead of numbers

df <- df %>% 
  select(-date)

# Write df_Illinois to a CSV file
write.csv(df_Illinois, "/Users/emilywu/Desktop/MMA/MGSC 661/Final Project/df_Illinois.csv", row.names = FALSE)

###################end of dataframe creation for df_Illinois###################
###############################################################################

##############################start building model#############################


##dummify month and weekday; transform n_killed to "fatal"
# Columns you want to convert to factors
columns_to_factorize <- c("state", "city_or_county", "congressional_district",
                          "state_house_district", "state_senate_district",
                          "teens_involved", "drug_involved", "gang_involved",
                          "month", "weekday")

# Convert each column to a factor
for (column_name in columns_to_factorize) {
  df[[column_name]] <- as.factor(df[[column_name]])
}

# Optionally, check the structure of the dataframe to confirm the changes
str(df[columns_to_factorize])

df$fatal <- ifelse(df$n_killed > 0, 1, 0)



View(df)

#################plot scatter plot between numerical predictors###############
# First, install and load the GGally package if you haven't already
install.packages("GGally")
library(GGally)

# Assuming 'df' is your dataframe

# Use ggpairs to create a scatter plot matrix
# Note: ggpairs only works with numerical data for scatter plots, 
# so we'll only select numerical columns from your dataframe
numerical_cols <- df[, sapply(df, is.numeric)]

# Draw the scatter plot matrix
ggpairs(numerical_cols)

###################plot bar chart for categorical predictors###################
# Columns to plot
columns_to_plot <- c("teens_involved", "drug_involved", "gang_involved", "bar", 
                     "home_invasion", "school", "drive_by")















# Loop through each column and create a percentage bar plot
for (col in columns_to_plot) {
  # Calculate the percentages
  df_percent <- df %>%
    group_by(!!sym(col)) %>%
    summarise(Count = n()) %>%
    mutate(Percentage = Count / sum(Count) * 100)
  
  # Plot
  p <- ggplot(df_percent, aes_string(x = col, y = 'Percentage', fill = col)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(labels = c("0" = "False", "1" = "True")) +
    xlab(col) +
    ylab("Percentage") +
    ggtitle(paste("Percentage Bar Chart of", col)) +
    theme_minimal() +
    scale_fill_manual(values = c("skyblue", "pink"))
  
  # Print the plot
  print(p)
}




#############create percentage bar chart of categorical vars vs fatal##########
# Loop through each column and create a percentage bar plot
library(ggplot2)
library(dplyr)
library(tidyr)


columns_to_plot <- c("teens_involved", "drug_involved", "gang_involved", "bar", 
                     "home_invasion", "school", "drive_by")

# First, ensure all columns are of the same type (factor in this case)
df <- df %>%
  mutate(across(all_of(columns_to_plot), as.factor))

# Pivot the data to long format
long_df <- df %>% 
  pivot_longer(cols = columns_to_plot, names_to = "category", values_to = "value") %>%
  mutate(value = as.factor(value), fatal = as.factor(fatal))

# Calculate the percentages of fatal and non-fatal incidents for each category
df_percent <- long_df %>%
  group_by(category, value, fatal) %>%
  summarise(Count = n(), .groups = 'drop') %>%
  mutate(Percentage = Count / sum(Count) * 100)

# Create the combined bar chart with facets for each category
combined_plot <- ggplot(df_percent, aes(x = value, y = Percentage, fill = fatal)) +
  geom_bar(stat = "identity", position = "fill") +
  facet_wrap(~category, scales = "free_x") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Category Value", y = "Percentage", fill = "Outcome") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_manual(values = c("skyblue", "pink"))

# Print the combined plot
print(combined_plot)



