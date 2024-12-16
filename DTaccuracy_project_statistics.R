#Loaded necessary packages 
install.packages("dplyr") # installing this package will be used for making data manipulation easier.
library(rpart)            # used for creating recursive partitioning and regression trees (decision trees)
library(caret)            # provides functions for training and plotting classification and regression models.
library(partykit)         # provides tools for creating and visualizing decision trees and other tree-based models.
library(reshape2)         # is used for reshaping data, such as converting data from wide format to long format and vice-versa, its part of data manipulation task.
library(ggplot2)          # which is used for creating complex and multi-layered graphics.
library(gridExtra)        # provides functions to arrange multiple grid-based graphics by combining multiple plots and other graphical objects into a single layout. 
library(dplyr)            # used for data manipulation tasks
set.seed(5)



#---------------------------------------------------------------------
                  #DIABETES DATASET ANALYSIS
#---------------------------------------------------------------------


#--TITLE: FUNCTIONS OF THE METHODS FOR ERROR ESTIMATION--

# Proposed Method (denoted as: Jeff_error), combining quantized Bayes error and sampling error.
# Minimum Error Pruning (denoted as: Cestnik_error).
# Pessimistic Error Pruning (denoted as: Quinlan_error).
# K-fold Cross-Validation with K fols = {2, 5, 10} (denoted as: CV).


Jeff_error=function(tree, train){
  #Quantized Bayes Error
  leaf_assignments = predict(tree, train, type = "class")
  
  emp_loss = sum(leaf_assignments != train$Class) / nrow(train)
  
  #SAMPLING ERROR 
  prior_parameter = 0.5
  number_classes = length(unique(train$Class))
  visual_tree = as.party(tree)
  
  # Get the terminal node indices for each observation in the training set
  predicted_terminal_nodes_train = as.numeric(predict(visual_tree, newdata = train, type = "node"))
  
  # Create a data frame combining terminal nodes and actual classes for the training set
  train_results = data.frame(Terminal_Node = predicted_terminal_nodes_train, Actual_Class = train$Class)
  
  # Count the occurrences of each class in each terminal node using table()
  train_class_counts = with(train_results, table(Terminal_Node, Actual_Class))
  
  # Convert class_counts_train to a data frame
  train_class_counts_df = as.data.frame(train_class_counts)
  
  # Pivot the data using dcast to get the desired matrix format
  positive_samples_y = dcast(data = train_class_counts_df, Terminal_Node ~ Actual_Class, value.var = "Freq")
  
  # Replace NA values with 0
  positive_samples_y[is.na(positive_samples_y)] = 0
  
  # Set the Terminal_Node column as row names
  rownames(positive_samples_y) = positive_samples_y$Terminal_Node
  positive_samples_y$Terminal_Node = NULL
  
  total_positive_samples =rowSums(positive_samples_y)
  
  estimated_posterior_proby = (positive_samples_y+prior_parameter)/(total_positive_samples+(number_classes*prior_parameter))
  
  var_estimated_pos=(estimated_posterior_proby*(1-estimated_posterior_proby))/(1+total_positive_samples)
  
  bias_estimated_pos=(estimated_posterior_proby-(positive_samples_y/total_positive_samples))^2
  
  pmf=total_positive_samples/nrow(train)
  
  #generalization error estimation
  
  second_part=((var_estimated_pos+bias_estimated_pos)^0.5)*pmf
  
  sum=sum(second_part)
  
  JEFF_error= emp_loss+sum
  return(JEFF_error)
}
Cestnik_error = function(tree, train) {
  # Leaf assignments
  leaf_assignments = predict(tree, train, type = "class")
  
  emp_loss = sum(leaf_assignments != train$Class) / nrow(train)
  
  # Create confusion matrix
  cm = table(leaf_assignments, train$Class)
  
  # Calculate error estimate
  n = sum(cm)
  p_e = 1 - sum(diag(cm)) / n
  k = length(unique(train$Class))
  N = nrow(train)
  
  cestnik_error = (p_e + (k - 1) / (2 * N)) / (1 + (k - 1) / N)
  return(cestnik_error)
}
Quinlan_error = function(tree, train) {
  # Leaf assignments
  leaf_assignments = predict(tree, train, type = "class")
  
  emp_loss = sum(leaf_assignments != train$Class) / nrow(train)
  
  # Create confusion matrix
  cm = table(leaf_assignments, train$Class)
  
  # Calculate error estimate
  n = sum(cm)
  e = 1 - sum(diag(cm)) / n
  k = length(unique(train$Class))
  
  quinlan_error = e + sqrt(e * (1 - e) / n)
  return(quinlan_error)
}
CV <- function(dats, n.folds, depth_p){
  folds = list()
  fold.size = nrow(dats)/n.folds
  remain = 1:nrow(dats)
  results = matrix(0, n.folds, 2)
  
  for (i in 1:n.folds){
    select = sample(remain, fold.size, replace = FALSE)
    folds[[i]] = select
    
    if (i == n.folds){
      folds[[i]] = remain
    }
    remain = setdiff(remain, select)
  }
  
  for (i in 1:n.folds){
    indis = folds[[i]]
    train = dats[-indis, ]
    validation = dats[indis, ]
    
    tree = rpart(Class ~ ., data = train, method = "class", parms=list(split="gini"),
                 control = rpart.control(minsplit=2, minbucket=1, cp=0, maxdepth = depth_p)) 
    
    predicted = predict(tree, validation, type = "class")
    error = sum(predicted != validation$Class) / nrow(validation)
    depth = max(rpart:::tree.depth(as.numeric(rownames(tree$frame))))
    
    results[i, 1] = error
    results[i, 2] = depth
  }
  
  valori_buoni = results[, 1][which(results[, 2] == depth_p)]
  if (length(valori_buoni != 0)){
    mean_error = mean(valori_buoni)
  } else {
    mean_error = NA
  }
  
  return(mean_error)
}

#Upoading diabetes dataset for the binary class and performing data manipulation
df_diabets = read.csv("/Users/lenovo/OneDrive/Desktop/Statisti_DTaccuracy/diabetes.csv", sep = ",", header = TRUE)
colnames(df_diabets)[colnames(df_diabets) == "Outcome"] ="Class"
df_diabets$Class = as.factor(df_diabets$Class)


dataframe = df_diabets     # assigning the specific dataset to a generic variable for flexibility purposes
num_splits = 50            # repeating 50 randomly the split of the data per each ratio


# creation of lists to put the results of the error estimated per each 
jeff_results = list(); cestnik_results = list(); quinlan_results = list(); CV2_results = list(); CV5_results = list(); CV10_results = list(); test_results = list()


depths= seq.int(1, 25, by = 1) # identifying a range going from 1 to 25 of tree depths
r = c(0.9, 0.5, 0.1)           # the split ratio of train and test
plots_list = list()



#--TITLE: DECISION TREE ERROR ESTIMATION ANALYSIS FOR DIFFERENT TRAIN/TEST RATIOS AND DEPTHS--


# Outer Loop over Proportions and Depths
for (prop in r) {
  for (depth in depths) {
    
    # Initialize Lists to Store Results for Each Method
    splits_jeff = list()
    splits_cestnik = list()
    splits_quinlan = list()
    splits_test = list()
    splits_CV2 = list()
    splits_CV5 = list()
    splits_CV10 = list()
    
    # Inner Loop for Splits
    for (i in 1:num_splits){
      # Calculate Number of Samples for Training Set
      num_samples = floor(prop * nrow(dataframe))
      # Randomly Sample indices for Training Set
      indices = sample(1:nrow(dataframe), num_samples)
      # Create Training and Testing Sets
      train_data = dataframe[indices, ]
      test_data = dataframe[-indices, ]
      
      # Perform Cross-Validation with 2, 5, and 10 Folds
      cv2_estimation = CV(train_data, 2, depth)
      cv5_estimation = CV(train_data, 5, depth)
      cv10_estimation = CV(train_data, 10, depth)
      
      # Store Cross-Validation Results
      splits_CV2[[i]] = list(estimation = cv2_estimation)
      splits_CV5[[i]] = list(estimation = cv5_estimation)
      splits_CV10[[i]] = list(estimation = cv10_estimation)
      
      # Train a Decision Tree Model
      model = rpart(Class ~ ., data = train_data, method = "class", parms=list(split="gini"),
                    control = rpart.control(minsplit=2, minbucket=1, cp=0, maxdepth = depth))
      
      # Determine Actual Depth of Trained Model
      depth_model = max(rpart:::tree.depth(as.numeric(rownames(model$frame))))
      
      # Check if Model Depth Matches Desired Depth
      if (depth_model == depth) {
        # Make Predictions on Test Data
        predictions = predict(model, newdata = test_data, type = "class")
        # Estimate Errors Using Different Methods
        estimation_jeff = Jeff_error(model, train_data)
        estimation_cestnik = Cestnik_error(model, train_data)
        estimation_quinlan = Quinlan_error(model, train_data)
      } else {
        # If Depth Doesn't Match, Set Estimates to NA
        predictions = NA
        estimation_jeff = NA
        estimation_cestnik = NA
        estimation_quinlan = NA
      }
      # Calculate Empirical Error
      test_estimation = mean(predictions != test_data$Class)
      
      # Store Error Estimates for Each Method
      splits_jeff[[i]] = list(estimation = estimation_jeff)
      splits_cestnik[[i]] = list(estimation = estimation_cestnik)
      splits_quinlan[[i]] = list(estimation = estimation_quinlan)
      splits_test[[i]] = list(estimation = test_estimation)
    }
    
    # Save Split Results for Each Depth
    jeff_results[[as.character(depth)]] = splits_jeff
    cestnik_results[[as.character(depth)]] = splits_cestnik
    quinlan_results[[as.character(depth)]] = splits_quinlan
    test_results[[as.character(depth)]] = splits_test
    CV2_results[[as.character(depth)]] = splits_CV2
    CV5_results[[as.character(depth)]] = splits_CV5
    CV10_results[[as.character(depth)]] = splits_CV10
  }
  
  # Calculate Mean Performance for Each Method Across Depths
  mean_performance_jeff = sapply(jeff_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_cestnik = sapply(cestnik_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_quinlan = sapply(quinlan_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_test = sapply(test_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_CV2 = sapply(CV2_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_CV5 = sapply(CV5_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_CV10 = sapply(CV10_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  
  # Combine Mean Performances into a Data Frame
  data <- data.frame(depths = depths,
                     Jeff = mean_performance_jeff,
                     Cestnik = mean_performance_cestnik,
                     Quinlan = mean_performance_quinlan,
                     Test = mean_performance_test,
                     CV2 = mean_performance_CV2,
                     CV5 = mean_performance_CV5,
                     CV10 = mean_performance_CV10)
  
  # Determine Maximum Depth with Complete Cases for Each Method
  max_depth = max(max(data$depths[complete.cases(data$Jeff)]),
                  max(data$depths[complete.cases(data$Cestnik)]),
                  max(data$depths[complete.cases(data$Quinlan)]),
                  max(data$depths[complete.cases(data$Test)]),
                  max(data$depths[complete.cases(data$CV2)]),
                  max(data$depths[complete.cases(data$CV5)]),
                  max(data$depths[complete.cases(data$CV10)]))
}

  
  # Melt the data frame for easier plotting
  melted_data = melt(data, id.vars = "depths", variable.name = "Method")
  
  # Create the plot
  current_plot = ggplot(melted_data, aes(x = depths, y = value, color = Method)) +
    geom_line() +
    geom_point() +
    labs(x = "Tree depth", y = "Generalization Error", color = "Method") +
    ggtitle(paste("diabet (r:", prop, ")")) +
    scale_x_continuous(limits = c(1, max_depth)) +
    theme_minimal()
  
  plots_list[[as.character(prop)]] = current_plot


# Arrange the plots in a grid
grid.arrange(grobs = plots_list, ncol = 2)



# Initialize parameters and result storage lists
r = c(0.9, 0.5, 0.1)         # Proportions for train/test splits
jeff_results2 = list()       # Storage for Jeff results
cestnik_results2 = list()    # Storage for Cestnik results
quinlan_results2 = list()    # Storage for Quinlan results
CV2_results2 = list()        # Storage for 2-fold cross-validation results
CV5_results2 = list()        # Storage for 5-fold cross-validation results
CV10_results2 = list()       # Storage for 10-fold cross-validation results
test_results2 = list()       # Storage for test results

pearson_jeff = c()           # Pearson correlation for Jeff method
pearson_cestnik = c()        # Pearson correlation for Cestnik method
pearson_quinlan = c()        # Pearson correlation for Quinlan method
pearson_CV2 = c()            # Pearson correlation for 2-fold CV
pearson_CV5 = c()            # Pearson correlation for 5-fold CV
pearson_CV10 = c()           # Pearson correlation for 10-fold CV

# Loop over train/test split proportions
for (prop in r) {
  splits_jeff_ = list()      # Storage for Jeff results for current proportion
  splits_cestnik_ = list()   # Storage for Cestnik results for current proportion
  splits_quinlan_ = list()   # Storage for Quinlan results for current proportion
  splits_CV2_ = list()       # Storage for 2-fold CV results for current proportion
  splits_CV5_ = list()       # Storage for 5-fold CV results for current proportion
  splits_CV10_ = list()      # Storage for 10-fold CV results for current proportion
  
  test_for_pearson = c()     # Storage for test results for Pearson correlation
  p_pearson_jeff = c()       # Storage for Pearson correlation values for Jeff method
  p_pearson_cestnik = c()    # Storage for Pearson correlation values for Cestnik method
  p_pearson_quinlan = c()    # Storage for Pearson correlation values for Quinlan method
  p_pearson_CV2 = c()        # Storage for Pearson correlation values for 2-fold CV
  p_pearson_CV5 = c()        # Storage for Pearson correlation values for 5-fold CV
  p_pearson_CV10 = c()       # Storage for Pearson correlation values for 10-fold CV
  
  # Loop for repeated random sampling
  for (i in 1:num_splits) {
    # Randomly sample indices for the training set
    num_samples = floor(prop * nrow(dataframe))
    indices = sample(1:nrow(dataframe), num_samples)
    train_data = dataframe[indices, ]   
    test_data = dataframe[-indices, ]   
    
    # Train a decision tree model
    model = rpart(Class ~ ., data = train_data, method = "class", parms=list(split="gini"))
    predictions = predict(model, newdata = test_data, type = "class")
    test_estimation = mean(predictions != test_data$Class)          # Error rate on test set
    test_for_pearson = append(test_for_pearson, test_estimation)    # Store test error for Pearson correlation
    
    # Perform cross-validation with 2, 5, and 10 folds
    for (n in c(2, 5, 10)){
      train_ctrl = trainControl(method = "cv", number = n)
      start_time_cv = Sys.time()
      cv_model = train(Class ~ ., data = train_data, method = "rpart",
                       trControl = train_ctrl, parms = list(split = "gini"))
      cv_estimation = 1 - mean(cv_model$results$Accuracy)           # Error rate from cross-validation
      end_time_cv = Sys.time()
      cv_mse_obs = (test_estimation - cv_estimation)^2              # MSE observation
      
      time_cv = end_time_cv - start_time_cv                         # Computation time
      if (n == 2){
        splits_CV2_[[i]] = list(estimation = cv_mse_obs, time = time_cv)
        p_pearson_CV2 = append(p_pearson_CV2, cv_estimation)
      } else if(n == 5){
        splits_CV5_[[i]] = list(estimation = cv_mse_obs, time = time_cv)
        p_pearson_CV5 = append(p_pearson_CV5, cv_estimation)
      } else {
        splits_CV10_[[i]] = list(estimation = cv_mse_obs, time = time_cv)
        p_pearson_CV10 = append(p_pearson_CV10, cv_estimation)
      }
    }
    
    # Calculate error estimation for Jeff method
    start_time_jeff = Sys.time()
    estimation_jeff = Jeff_error(model, train_data)
    end_time_jeff = Sys.time()
    
    # Calculate error estimation for Cestnik method
    start_time_cestnik = Sys.time()
    estimation_cestnik = Cestnik_error(model, train_data)
    end_time_cestnik = Sys.time()
    
    # Calculate error estimation for Quinlan method
    start_time_quinlan = Sys.time()
    estimation_quinlan = Quinlan_error(model, train_data)
    end_time_quinlan = Sys.time()
    
    # Store Pearson correlation values
    p_pearson_jeff = append(p_pearson_jeff, estimation_jeff)
    p_pearson_cestnik = append(p_pearson_cestnik, estimation_cestnik)
    p_pearson_quinlan = append(p_pearson_quinlan, estimation_quinlan)
    
    # Calculate MSE observations
    mse_jeff_obs = (test_estimation - estimation_jeff)^2
    mse_cestnik_obs = (test_estimation - estimation_cestnik)^2
    mse_quinlan_obs = (test_estimation - estimation_quinlan)^2
    
    # Calculate computation times
    time_jeff = end_time_jeff - start_time_jeff
    time_cestnik = end_time_cestnik - start_time_cestnik
    time_quinlan = end_time_quinlan - start_time_quinlan
    
    # Store results for each method
    splits_jeff_[[i]] = list(estimation = mse_jeff_obs, time = time_jeff)
    splits_cestnik_[[i]] = list(estimation = mse_cestnik_obs, time = time_cestnik)
    splits_quinlan_[[i]] = list(estimation = mse_quinlan_obs, time = time_quinlan)
  }
  
  # Store results for current proportion
  jeff_results2[[as.character(prop)]] = splits_jeff_
  cestnik_results2[[as.character(prop)]] = splits_cestnik_
  quinlan_results2[[as.character(prop)]] = splits_quinlan_
  CV2_results2[[as.character(prop)]] = splits_CV2_
  CV5_results2[[as.character(prop)]] = splits_CV5_
  CV10_results2[[as.character(prop)]] = splits_CV10_
  
  # Calculate and store Pearson correlations for each method
  cor_jeff = cor(test_for_pearson, p_pearson_jeff, method = "pearson")
  pearson_jeff = append(pearson_jeff, cor_jeff)
  cor_cestnik = cor(test_for_pearson, p_pearson_cestnik, method = "pearson")
  pearson_cestnik = append(pearson_cestnik, cor_cestnik)
  cor_quinlan = cor(test_for_pearson, p_pearson_quinlan, method = "pearson")
  pearson_quinlan = append(pearson_quinlan, cor_quinlan)
  cor_CV2 = cor(test_for_pearson, p_pearson_CV2, method="pearson")
  pearson_CV2 = append(pearson_CV2, cor_CV2)
  cor_CV5 = cor(test_for_pearson, p_pearson_CV5, method="pearson")
  pearson_CV5 = append(pearson_CV5, cor_CV5)
  cor_CV10 = cor(test_for_pearson, p_pearson_CV10, method="pearson")
  pearson_CV10 = append(pearson_CV10, cor_CV10)
}


#--TITLE: COMPREHENSIVE ANALYSIS OF MSE, PEARSON CORRELATION, AND COMPUTATION TIME FOR DIFFERENT ERROR ESTIMATION METHODS--

### FUNCTIONS TO CALCULATE MSE AND STANDARD DEVIATION

# Function to calculate the average estimation
calculate_average = function(nested_list) {
  estimations = sapply(nested_list, function(x) x$estimation)
  average = mean(estimations)
  return(average)
}

# Function to calculate the standard deviation of estimations
calculate_std = function(nested_list){
  estimations = sapply(nested_list, function(x) x$estimation)
  std = sd(estimations)
  return (std)
}

# Calculate MSE for each method
jeff = lapply(jeff_results2, calculate_average)
mse_jeff = unlist(jeff)

cestnik = lapply(cestnik_results2, calculate_average)
mse_cestnik = unlist(cestnik)

quinlan = lapply(quinlan_results2, calculate_average)
mse_quinlan = unlist(quinlan)

CV2 = lapply(CV2_results2, calculate_average)
mse_CV2 = unlist(CV2)

CV5 = lapply(CV5_results2, calculate_average)
mse_CV5 = unlist(CV5)

CV10 = lapply(CV10_results2, calculate_average)
mse_CV10 = unlist(CV10)

# Calculation of standard deviation of MSE for each method
std_jeff = lapply(jeff_results2, calculate_std)
std_dev_mse_jeff = unlist(std_jeff)

std_cestnik = lapply(cestnik_results2, calculate_std)
std_dev_mse_cestnik = unlist(std_cestnik)

std_quinlan = lapply(quinlan_results2, calculate_std)
std_dev_mse_quinlan = unlist(std_quinlan)

std_CV2 = lapply(CV2_results2, calculate_std)
std_dev_mse_CV2 = unlist(std_CV2)

std_CV5 = lapply(CV5_results2, calculate_std)
std_dev_mse_CV5 = unlist(std_CV5)

std_CV10 = lapply(CV10_results2, calculate_std)
std_dev_mse_CV10 = unlist(std_CV10)

#### CREATION OF MSE GRAPHS

# Create a data frame with MSE and standard deviations
plot_data = data.frame(
  Proportion = r,
  MSE_Jeff = mse_jeff,
  SD_Jeff = std_dev_mse_jeff,
  MSE_Cestnik = mse_cestnik,
  SD_Cestnik = std_dev_mse_cestnik,
  MSE_Quinlan = mse_quinlan,
  SD_Quinlan = std_dev_mse_quinlan,
  MSE_CV2 = mse_CV2,
  SD_CV2 = std_dev_mse_CV2,
  MSE_CV5 = mse_CV5,
  SD_CV5 = std_dev_mse_CV5,
  MSE_CV10 = mse_CV10,
  SD_CV10 = std_dev_mse_CV10
)

# Creating the plots using ggplot2
plot0 = ggplot(plot_data, aes(x = Proportion)) +
  geom_line(aes(y = MSE_Jeff, color="MSE_Jeff"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_Jeff - SD_Jeff, ymax = MSE_Jeff + SD_Jeff, color="MSE_Jeff"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_Cestnik, color="MSE_Cestnik"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_Cestnik - SD_Cestnik, ymax = MSE_Cestnik + SD_Cestnik, color="MSE_Cestnik"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_Quinlan, color="MSE_Quinlan"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_Quinlan - SD_Quinlan, ymax = MSE_Quinlan + SD_Quinlan, color="MSE_Quinlan"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_CV2, color="MSE_CV2"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_CV2 - SD_CV2, ymax = MSE_CV2 + SD_CV2, color="MSE_CV2"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_CV5, color="MSE_CV5"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_CV5 - SD_CV5, ymax = MSE_CV5 + SD_CV5, color="MSE_CV5"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_CV10, color="MSE_CV10"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_CV10 - SD_CV10, ymax = MSE_CV10 + SD_CV10, color="MSE_CV10"),
                width = 0.1, size = 0.5) +
  scale_color_manual(values=c("MSE_Jeff"="blue", "MSE_Cestnik"="orange", "MSE_Quinlan"="purple", "MSE_CV2"="red", "MSE_CV5"="yellow", "MSE_CV10"="green")) +
  labs(title = "DIABETES MSE",
       x = "#train/#data ratio",
       y = "MSE vs measured error") +
  scale_x_continuous(breaks = r) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Print the plot
print(plot0)

#### CREATION OF PEARSON CORRELATION GRAPHS

# Create a data frame with Pearson correlations
plot_pearson = data.frame(
  Proportion = r,
  Correlation_Jeff = pearson_jeff,
  Correlation_Cestnik = pearson_cestnik,
  Correlation_Quinlan = pearson_quinlan,
  Correlation_CV2 = pearson_CV2,
  Correlation_CV5 = pearson_CV5,
  Correlation_CV10 = pearson_CV10
)

# Create the plot using ggplot2
plot2 = ggplot(plot_pearson, aes(x = Proportion)) +
  geom_line(aes(y = Correlation_Jeff, color="Correlation_Jeff"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_Cestnik, color="Correlation_Cestnik"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_Quinlan, color="Correlation_Quinlan"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_CV2, color="Correlation_CV2"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_CV5, color="Correlation_CV5"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_CV10, color="Correlation_CV10"), linetype = "solid", size = 1) +
  scale_color_manual(values=c("Correlation_Jeff"="blue", "Correlation_Cestnik"="orange", "Correlation_Quinlan"="purple", "Correlation_CV2"="red", "Correlation_CV5"="yellow", "Correlation_CV10"="green")) +
  labs(title = "DIABETES PEARSON CORRELATION",
       x = "#train/#data ratio",
       y = "Pearson Correlation") +
  scale_x_continuous(breaks = r) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Print the plot
print(plot2)

#### CALCULATION OF COMPUTATION TIME COMPLEXITY

# Function to calculate the average computation time
calculate_time = function(nested_list) {
  estimations = sapply(nested_list, function(x) x$time)
  time = mean(estimations)
  return(time)
}

# Calculate computation time for each method
Jeff_time = unlist(lapply(jeff_results2, calculate_time))
Cestnik_time = unlist(lapply(cestnik_results2, calculate_time))
Quinlan_time = unlist(lapply(quinlan_results2, calculate_time))
CV2_time = unlist(lapply(CV2_results2, calculate_time))
CV5_time = unlist(lapply(CV5_results2, calculate_time))
CV10_time = unlist(lapply(CV10_results2, calculate_time))

# Create a matrix for the proportions and times
time_matrix = rbind(Jeff_time, Cestnik_time, Quinlan_time, CV2_time, CV5_time, CV10_time)
colnames(time_matrix) = c("0.9", "0.5", "0.1")

# Create a bar plot with axis labels and title
barplot(time_matrix, beside = TRUE,
        legend.text = rownames(time_matrix),
        names.arg = colnames(time_matrix), 
        args.legend = list(cex = 0.75, x="topleft", inset = c(-0.045, -0.45)), 
        col = rainbow(nrow(time_matrix)),
        xlab = "Train/Test Ratio",
        ylab = "Computation Time (seconds)")

# Add title for clarity
title("Computation Time for Different Methods")



#---------------------------------------------------------------------
                     #ECOLI DATASET ANALYSIS
#---------------------------------------------------------------------


# Importing ecoli dataset for the multiclass case and performing data manipulation

file_path <- "/Users/lenovo/OneDrive/Desktop/Statisti_DTaccuracy/ecoli.data"
df_ecoli <- read.table(file_path, sep = "", header = FALSE, stringsAsFactors = FALSE)
colnames(df_ecoli) <- c("SequenceName", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "Class")
df_ecoli$Class <- as.factor(as.numeric(as.factor(df_ecoli$Class)))
df_ecoli$SequenceName <- as.numeric(as.factor(df_ecoli$SequenceName))

# Adaptation of the code for the ecoli dataset
dataframe = df_ecoli
num_splits = 50

# creation of lists to put the results of the error estimated per each 
jeff_results = list(); cestnik_results = list(); quinlan_results = list(); CV2_results = list(); CV5_results = list(); CV10_results = list(); test_results = list()



depths= seq.int(1, 25, by = 1) # identifying a range going from 1 to 25 of tree depths
r = c(0.9, 0.5, 0.1)           # the split ratio of train and test
num_splits = 50                # repeating 50 randomly the split of the data per each ratio
plots_list = list()

#--TITLE: DECISION TREE ERROR ESTIMATION ANALYSIS FOR DIFFERENT TRAIN/TEST RATIOS AND DEPTHS--


# Outer Loop over Proportions and Depths
for (prop in r) {
  for (depth in depths) {
    # Initialize Lists to Store Results for Each Method
    splits_jeff = list()
    splits_cestnik = list()
    splits_quinlan = list()
    splits_test = list()
    splits_CV2 = list()
    splits_CV5 = list()
    splits_CV10 = list()
    
    # Inner Loop for Splits
    for (i in 1:num_splits){
      # Calculate Number of Samples for Training Set
      num_samples = floor(prop * nrow(dataframe))
      # Randomly Sample Indices for Training Set
      indices = sample(1:nrow(dataframe), num_samples)
      # Create Training and Testing Sets
      train_data = dataframe[indices, ]
      test_data = dataframe[-indices, ]
      
      print(paste("Proportion:", prop, "Depth:", depth, "Split:", i, "Train size:", nrow(train_data), "Test size:", nrow(test_data)))
      
      if (nrow(train_data) == 0 || nrow(test_data) == 0) {
        print("Error: train_data or test_data is empty!")
        next
      }
      
      # Perform Cross-Validation with 2, 5, and 10 Folds
      cv2_estimation = CV(train_data, 2, depth)
      cv5_estimation = CV(train_data, 5, depth)
      cv10_estimation = CV(train_data, 10, depth)
      
      # Store Cross-Validation Results
      splits_CV2[[i]] = list(estimation = cv2_estimation)
      splits_CV5[[i]] = list(estimation = cv5_estimation)
      splits_CV10[[i]] = list(estimation = cv10_estimation)
      
      # Train a Decision Tree Model
      model = rpart(Class ~ ., data = train_data, method = "class", parms = list(split = "gini"),
                    control = rpart.control(minsplit = 2, minbucket = 1, cp = 0, maxdepth = depth))
      
      # Determine Actual Depth of Trained Model
      depth_model = max(rpart:::tree.depth(as.numeric(rownames(model$frame))))
      
      # Check if Model Depth Matches Desired Depth
      if (depth_model == depth) {
        # Make Predictions on Test Data
        predictions = predict(model, newdata = test_data, type = "class")
        # Estimate Errors Using Different Methods
        estimation_jeff = Jeff_error(model, train_data)
        estimation_cestnik = Cestnik_error(model, train_data)
        estimation_quinlan = Quinlan_error(model, train_data)
      } else {
        # If Depth Doesn't Match, Set Estimates to NA
        predictions = NA
        estimation_jeff = NA
        estimation_cestnik = NA
        estimation_quinlan = NA
      }
      
      # Calculate Empirical Error
      test_estimation = mean(predictions != test_data$Class)
      
      # Store Error Estimates for Each Method
      splits_jeff[[i]] = list(estimation = estimation_jeff)
      splits_cestnik[[i]] = list(estimation = estimation_cestnik)
      splits_quinlan[[i]] = list(estimation = estimation_quinlan)
      splits_test[[i]] = list(estimation = test_estimation)
    }
    # Save Split Results for Each Depth
    jeff_results[[as.character(depth)]] = splits_jeff
    cestnik_results[[as.character(depth)]] = splits_cestnik
    quinlan_results[[as.character(depth)]] = splits_quinlan
    test_results[[as.character(depth)]] = splits_test
    CV2_results[[as.character(depth)]] = splits_CV2
    CV5_results[[as.character(depth)]] = splits_CV5
    CV10_results[[as.character(depth)]] = splits_CV10
  }
  
  # Calculate Mean Performance for Each Method Across Depths
  mean_performance_jeff = sapply(jeff_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_cestnik = sapply(cestnik_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_quinlan = sapply(quinlan_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_test = sapply(test_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_CV2 = sapply(CV2_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_CV5 = sapply(CV5_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  mean_performance_CV10 = sapply(CV10_results, function(metrics) mean(sapply(metrics, function(split) split$estimation), na.rm = TRUE))
  
  # Combine Mean Performances into a Data Frame  
  data = data.frame(depths = depths,
                    Jeff = mean_performance_jeff,
                    Cestnik = mean_performance_cestnik,
                    Quinlan = mean_performance_quinlan,
                    Test = mean_performance_test,
                    CV2 = mean_performance_CV2,
                    CV5 = mean_performance_CV5,
                    CV10 = mean_performance_CV10)
  
  # Determine Maximum Depth with Complete Cases for Each Method
  max_depth = max(max(data$depths[complete.cases(data$Jeff)]),
                  max(data$depths[complete.cases(data$Cestnik)]),
                  max(data$depths[complete.cases(data$Quinlan)]),
                  max(data$depths[complete.cases(data$Test)]),
                  max(data$depths[complete.cases(data$CV2)]),
                  max(data$depths[complete.cases(data$CV5)]),
                  max(data$depths[complete.cases(data$CV10)]))
  
  if (nrow(data) == 0) {
    print("Error: data frame for plotting is empty!")
    next
  }
  
  # Melt the data frame for easier plotting
  melted_data = melt(data, id.vars = "depths", variable.name = "Method")
  
  # Create the plot
  current_plot = ggplot(melted_data, aes(x = depths, y = value, color = Method)) +
    geom_line() +
    geom_point() +
    labs(x = "Tree depth", y = "Generalization Error", color = "Method") +
    ggtitle(paste("ecoli (r:", prop, ")")) +
    scale_x_continuous(limits = c(1, max_depth)) +  # Set x-axis limits
    theme_minimal()
  
  plots_list[[as.character(prop)]] = current_plot
}

# Print the plots
for (prop in names(plots_list)) {
  print(plots_list[[prop]])
}
# Arrange the plots in a grid
grid.arrange(grobs = plots_list, ncol = 2)  # Adjust ncol as needed

# Questi vengono utilizzati per i grafici MSE/PEARSON
# Function to perform stratified sampling
stratified_sampling <- function(df, prop) {
  train_indices <- createDataPartition(df$Class, p = prop, list = FALSE)
  train_data <- df[train_indices, ]
  test_data <- df[-train_indices, ]
  return(list(train_data = train_data, test_data = test_data))
}

#Creation of the graphs  per MSE/Pearson
r = c(0.9, 0.5, 0.1)
jeff_results2 = list()
cestnik_results2 = list()
quinlan_results2 = list()
CV2_results2 = list()
CV5_results2 = list()
CV10_results2 = list()
test_results2 = list()

pearson_jeff = c()
pearson_cestnik = c()
pearson_quinlan = c()
pearson_CV2 = c()
pearson_CV5 = c()
pearson_CV10 = c()

for (prop in r) {
  splits_jeff_ = list()
  splits_cestnik_ = list()
  splits_quinlan_ = list()
  splits_CV2_ = list()
  splits_CV5_ = list()
  splits_CV10_ = list()
  
  test_for_pearson = c()
  p_pearson_jeff = c()
  p_pearson_cestnik = c()
  p_pearson_quinlan = c()
  p_pearson_CV2 = c()
  p_pearson_CV5 = c()
  p_pearson_CV10 = c()
  
  for (i in 1:num_splits) {
    sample <- stratified_sampling(dataframe, prop)
    train_data <- sample$train_data
    test_data <- sample$test_data
    
    print(paste("Proportion:", prop, "Split:", i, "Train size:", nrow(train_data), "Test size:", nrow(test_data)))
    
    if (nrow(train_data) == 0 || nrow(test_data) == 0) {
      print("Error: train_data or test_data is empty!")
      next
    }
    
    # Check for missing factor levels in the training data
    if (length(setdiff(levels(dataframe$Class), levels(train_data$Class))) > 0) {
      print("Warning: Missing factor levels in the training data!")
      next
    }
    
    model = rpart(Class ~ ., data = train_data, method = "class", parms = list(split = "gini"))
    predictions = predict(model, newdata = test_data, type = "class")
    test_estimation = mean(predictions != test_data$Class)
    test_for_pearson = append(test_for_pearson, test_estimation)
    
    for (n in c(2, 5, 10)){
      train_ctrl = trainControl(method = "cv", number = n)  
      start_time_cv = Sys.time()
      cv_model = train(Class ~ ., data = train_data, method = "rpart",
                       trControl = train_ctrl, parms = list(split = "gini"))
      cv_estimation = 1 - mean(cv_model$results$Accuracy)
      end_time_cv = Sys.time()
      cv_mse_obs = (test_estimation - cv_estimation)^2
      
      time_cv = end_time_cv - start_time_cv
      if (n == 2){
        splits_CV2_[[i]] = list(estimation = cv_mse_obs, time = time_cv)
        p_pearson_CV2 = append(p_pearson_CV2, cv_estimation)
      } else if(n == 5){
        splits_CV5_[[i]] = list(estimation = cv_mse_obs, time = time_cv)
        p_pearson_CV5 = append(p_pearson_CV5, cv_estimation)
      } else {
        splits_CV10_[[i]] = list(estimation = cv_mse_obs, time = time_cv)
        p_pearson_CV10 = append(p_pearson_CV10, cv_estimation)
      }
    }
    
    start_time_jeff = Sys.time()
    estimation_jeff = Jeff_error(model, train_data)
    end_time_jeff = Sys.time()
    
    start_time_cestnik = Sys.time()
    estimation_cestnik = Cestnik_error(model, train_data)
    end_time_cestnik = Sys.time()
    
    start_time_quinlan = Sys.time()
    estimation_quinlan = Quinlan_error(model, train_data)
    end_time_quinlan = Sys.time()
    
    p_pearson_jeff = append(p_pearson_jeff, estimation_jeff)
    p_pearson_cestnik = append(p_pearson_cestnik, estimation_cestnik)
    p_pearson_quinlan = append(p_pearson_quinlan, estimation_quinlan)
    
    mse_jeff_obs = (test_estimation - estimation_jeff)^2
    mse_cestnik_obs = (test_estimation - estimation_cestnik)^2
    mse_quinlan_obs = (test_estimation - estimation_quinlan)^2
    
    time_jeff = end_time_jeff - start_time_jeff
    time_cestnik = end_time_cestnik - start_time_cestnik
    time_quinlan = end_time_quinlan - start_time_quinlan
    
    splits_jeff_[[i]] = list(estimation = mse_jeff_obs, time = time_jeff)
    splits_cestnik_[[i]] = list(estimation = mse_cestnik_obs, time = time_cestnik)
    splits_quinlan_[[i]] = list(estimation = mse_quinlan_obs, time = time_quinlan)
  }
  
  print(paste("Finished processing proportion:", prop))
  print(paste("Length of test_for_pearson:", length(test_for_pearson)))
  print(paste("Length of p_pearson_jeff:", length(p_pearson_jeff)))
  
  if (length(test_for_pearson) == 0 || length(p_pearson_jeff) == 0) {
    print(paste("Skipping proportion:", prop, "due to insufficient data."))
    next
  }
  
  jeff_results2[[as.character(prop)]] = splits_jeff_
  cestnik_results2[[as.character(prop)]] = splits_cestnik_
  quinlan_results2[[as.character(prop)]] = splits_quinlan_
  CV2_results2[[as.character(prop)]] = splits_CV2_
  CV5_results2[[as.character(prop)]] = splits_CV5_
  CV10_results2[[as.character(prop)]] = splits_CV10_
  
  cor_jeff = cor(test_for_pearson, p_pearson_jeff, method = "pearson")
  pearson_jeff = append(pearson_jeff, cor_jeff)
  cor_cestnik = cor(test_for_pearson, p_pearson_cestnik, method = "pearson")
  pearson_cestnik = append(pearson_cestnik, cor_cestnik)
  cor_quinlan = cor(test_for_pearson, p_pearson_quinlan, method = "pearson")
  pearson_quinlan = append(pearson_quinlan, cor_quinlan)
  cor_CV2 = cor(test_for_pearson, p_pearson_CV2, method = "pearson")
  pearson_CV2 = append(pearson_CV2, cor_CV2)
  cor_CV5 = cor(test_for_pearson, p_pearson_CV5, method = "pearson")
  pearson_CV5 = append(pearson_CV5, cor_CV5)
  cor_CV10 = cor(test_for_pearson, p_pearson_CV10, method = "pearson")
  pearson_CV10 = append(pearson_CV10, cor_CV10)
}

#--TITLE: ANALYSIS OF MSE, PEARSON CORRELATION, AND COMPUTATION TIME FOR DIFFERENT ERROR ESTIMATION METHODS--

### FUNCTIONS TO CALCULATE MSE AND STANDARD DEVIATION

# Function to calculate the average estimation
calculate_average = function(nested_list) {
  estimations = sapply(nested_list, function(x) x$estimation)
  average = mean(estimations)
  return(average)
}


# Function to calculate the standard deviation of estimations
calculate_std = function(nested_list){
  estimations = sapply(nested_list, function(x) x$estimation)
  std = sd(estimations)
  return (std)
}

# Calculate MSE for each method
jeff = lapply(jeff_results2, calculate_average)
mse_jeff = unlist(jeff)

cestnik = lapply(cestnik_results2, calculate_average)
mse_cestnik = unlist(cestnik)

quinlan = lapply(quinlan_results2, calculate_average)
mse_quinlan = unlist(quinlan)

CV2 = lapply(CV2_results2, calculate_average)
mse_CV2 = unlist(CV2)

CV5 = lapply(CV5_results2, calculate_average)
mse_CV5 = unlist(CV5)

CV10 = lapply(CV10_results2, calculate_average)
mse_CV10 = unlist(CV10)

# Calculation of standard deviation of MSE for each method
std_jeff = lapply(jeff_results2, calculate_std)
std_dev_mse_jeff = unlist(std_jeff)

std_cestnik = lapply(cestnik_results2, calculate_std)
std_dev_mse_cestnik = unlist(std_cestnik)

std_quinlan = lapply(quinlan_results2, calculate_std)
std_dev_mse_quinlan = unlist(std_quinlan)

std_CV2 = lapply(CV2_results2, calculate_std)
std_dev_mse_CV2 = unlist(std_CV2)

std_CV5 = lapply(CV5_results2, calculate_std)
std_dev_mse_CV5 = unlist(std_CV5)

std_CV10 = lapply(CV10_results2, calculate_std)
std_dev_mse_CV10 = unlist(std_CV10)

#### CREATION OF MSE GRAPHS

# Create a data frame with MSE and standard deviations

plot_data = data.frame(
  Proportion = r,
  MSE_Jeff = mse_jeff,
  SD_Jeff = std_dev_mse_jeff,
  MSE_Cestnik = mse_cestnik,
  SD_Cestnik = std_dev_mse_cestnik,
  MSE_Quinlan = mse_quinlan,
  SD_Quinlan = std_dev_mse_quinlan,
  MSE_CV2 = mse_CV2,
  SD_CV2 = std_dev_mse_CV2,
  MSE_CV5 = mse_CV5,
  SD_CV5 = std_dev_mse_CV5,
  MSE_CV10 = mse_CV10,
  SD_CV10 = std_dev_mse_CV10
)

# Creating the plots using ggplot2
plot0 = ggplot(plot_data, aes(x = Proportion)) +
  geom_line(aes(y = MSE_Jeff, color="MSE_Jeff"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_Jeff - SD_Jeff, ymax = MSE_Jeff + SD_Jeff, color="MSE_Jeff"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_Cestnik, color="MSE_Cestnik"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_Cestnik - SD_Cestnik, ymax = MSE_Cestnik + SD_Cestnik, color="MSE_Cestnik"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_Quinlan, color="MSE_Quinlan"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_Quinlan - SD_Quinlan, ymax = MSE_Quinlan + SD_Quinlan, color="MSE_Quinlan"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_CV2, color="MSE_CV2"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_CV2 - SD_CV2, ymax = MSE_CV2 + SD_CV2, color="MSE_CV2"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_CV5, color="MSE_CV5"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_CV5 - SD_CV5, ymax = MSE_CV5 + SD_CV5, color="MSE_CV5"),
                width = 0.1, size = 0.5) +
  geom_line(aes(y = MSE_CV10, color="MSE_CV10"), linetype = "solid", size = 1) +
  geom_errorbar(aes(ymin = MSE_CV10 - SD_CV10, ymax = MSE_CV10 + SD_CV10, color="MSE_CV10"),
                width = 0.1, size = 0.5) +
  scale_color_manual(values=c("MSE_Jeff"="blue", "MSE_Cestnik"="orange", "MSE_Quinlan"="purple", "MSE_CV2"="red", "MSE_CV5"="yellow", "MSE_CV10"="green")) +
  labs(title = "ecoli mse",
       x = "#train/#data ratio",
       y = "MSE vs measured error") +
  scale_x_continuous(breaks = r) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title

# Print the plot
print(plot0)

#### CREATION OF PEARSON CORRELATION GRAPHS

# Create a data frame with Pearson correlations

plot_pearson = data.frame(
  Proportion = r,
  Correlation_Jeff = pearson_jeff,
  Correlation_Cestnik = pearson_cestnik,
  Correlation_Quinlan = pearson_quinlan,
  Correlation_CV2 = pearson_CV2,
  Correlation_CV5 = pearson_CV5,
  Correlation_CV10 = pearson_CV10
)

# Create the plot using ggplot2
plot2 = ggplot(plot_pearson, aes(x = Proportion)) +
  geom_line(aes(y = Correlation_Jeff, color="Correlation_Jeff"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_Cestnik, color="Correlation_Cestnik"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_Quinlan, color="Correlation_Quinlan"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_CV2, color="Correlation_CV2"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_CV5, color="Correlation_CV5"), linetype = "solid", size = 1) +
  geom_line(aes(y = Correlation_CV10, color="Correlation_CV10"), linetype = "solid", size = 1) +
  scale_color_manual(values=c("Correlation_Jeff"="blue", "Correlation_Cestnik"="orange", "Correlation_Quinlan"="purple", "Correlation_CV2"="red", "Correlation_CV5"="yellow", "Correlation_CV10"="green")) +
  labs(title = "ecoli pearson correlation",
       x = "#train/#data ratio",
       y = "pearson corr") +
  scale_x_continuous(breaks = r) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title

# Print the plot
print(plot2)

#### CALCULATION OF COMPUTATION TIME COMPLEXITY

# Function to calculate the average computation time
calculate_time = function(nested_list) {
  estimations = sapply(nested_list, function(x) x$time)
  time = mean(estimations)
  return(time)
}

# Calculate computation time for each method
Jeff_time = unlist(lapply(jeff_results2, calculate_time))
Cestnik_time = unlist(lapply(cestnik_results2, calculate_time))
Quinlan_time = unlist(lapply(quinlan_results2, calculate_time))
CV2_time = unlist(lapply(CV2_results2, calculate_time))
CV5_time = unlist(lapply(CV5_results2, calculate_time))
CV10_time = unlist(lapply(CV10_results2, calculate_time))

# Create a matrix for the proportions and times
time_matrix <- rbind(Jeff_time, Cestnik_time, Quinlan_time, CV2_time, CV5_time, CV10_time)
colnames(time_matrix) <- c("0.9", "0.5", "0.1")

# Create a bar plot with axis labels and title

barplot(time_matrix, beside = TRUE,
        legend.text = rownames(time_matrix),
        names.arg = colnames(time_matrix), 
        args.legend = list(cex = 0.75, x="topleft", inset = c(-0.045, -0.45)), 
        col = rainbow(nrow(time_matrix)),
        xlab = "Train/Test Ratio",
        ylab = "Computation Time (seconds)")

# Add title for clarity
title("Computation Time for Different Methods")
