#Assignemnt 2 CS699


#Problem 1 
#| results: hold

# 1. Load the observation data
# We represent the categorical features for people P4, P5, and P9
people <- data.frame(
  ID        = c("P4", "P5", "P9"),
  job       = c("management", "blue-collar", "entrepreneur"),
  marital   = c("married", "single", "married"),
  education = c("tertiary", "secondary", "tertiary"),
  default   = c("no", "no", "no"),
  housing   = c("yes", "yes", "yes"),
  loan      = c("yes", "no", "no"),
  contact   = c("unknown", "unknown", "unknown"),
  stringsAsFactors = FALSE
)

# 2. Define the nominal columns to be used for comparison
nom_cols <- c("job", "marital", "education", "default", "housing", "loan", "contact")

# 3. Define the Scaled Hamming Distance function
# Proportion of mismatches = (Number of mismatches) / (Total number of features)
hamming_dist <- function(id_a, id_b, df, cols) {
  row_a <- df[df$ID == id_a, cols]
  row_b <- df[df$ID == id_b, cols]
  
  # mean() of a logical vector returns the proportion of TRUE values (mismatches)
  return(mean(row_a != row_b))
}

# 4. Calculate distances from P4 to P5 and P4 to P9
dist_P4_P5 <- hamming_dist("P4", "P5", people, nom_cols)
dist_P4_P9 <- hamming_dist("P4", "P9", people, nom_cols)

# 5. Determine which person is closer to P4
closer_person <- ifelse(dist_P4_P5 < dist_P4_P9, "P5", "P9")

# 6. Display the results
list(
  distance_P4_P5 = round(dist_P4_P5, 3),
  distance_P4_P9 = round(dist_P4_P9, 3),
  closer_to_P4   = closer_person
)


#  OTHER SIMPLE SOLUTION
d(P4, P5) = 4/7 = 0.571     #mismatches / all

d(P4, P9 )= 2/7 = 0.285       #mismatches / all

# So WE CONCLUDE THAT  P4 IS CLOSER TO P9



#Problem 2
# Load the two-row Z-values data set and compute Euclidean & Manhattan distances
A <- data.frame(
  Object = c("O1", "O2"),
  A1 = c(88, 97),
  A2 = c(47, 63),
  A3 = c(32,18),
  A4 = c(6,  4))

# Keep only the numeric attributes for distance calculations
A_values <- A[, -1]          # drop the identifier column

# Euclidean distance — dist() defaults to Euclidean for numeric data frames
euclid_O1_O2 <- as.numeric(dist(A_values)[1])

# Manhattan distance — specify method = "manhattan"
manh_O1_O2 <- as.numeric(dist(A_values, method = "manhattan")[1])

# Return both distances in a tidy structure
list(euclidean = euclid_O1_O2,
     manhattan = manh_O1_O2)


# Problem 3
df <- read.csv("accidents1030.csv")
dim(df)
# Ensure the class column is categorical so caret treats it correctly.
df$MAX_SEV <- factor(df$MAX_SEV)
#1)- Generate training and. Use 1/3 of the data in the holdout
#1. Train / hold-out split 1/3 and 2/3
# initial_split()/// stratified on MAX_SEV, matching the hold-out
library(rsample)
set.seed(42) # to  Make it precisely reproducible.
split  <- initial_split(df, prop = 2/3, strata = MAX_SEV)
train  <- training(split)
holdout <- testing(split)

#(2). Fit a k-Nearest Neighbor model.

library(caret)
ctrl <- trainControl(method = "cv",    # 10-fold CV
                     number = 10,
                     classProbs = FALSE,
                     summaryFunction = defaultSummary)

knn_mod <- train(MAX_SEV ~ ., data = train,
                 method      = "knn",
                 trControl   = ctrl,
                 preProcess  = c("center", "scale"),  # center and scale mandated by lecture
                 tuneLength  = 30)                    # search 30 odd k’s

# Best k chosen by caret
best_k <- knn_mod$bestTune$k

## k-Nearest Neighbor Code - Predict, Confusion Matrix, Accuracy

pred <- predict(knn_mod, newdata = holdout, type = "raw")

# confusionMatrix() // 
cm <- confusionMatrix(data = pred,
                      reference = holdout$MAX_SEV,
                      positive  = "OnTime")   # pick the “positive” label as needed

list(
  best_k          = best_k,
  confusion_table = cm$table,
  accuracy        = cm$overall["Accuracy"]
)


#(3). Does this seem to be a good model? Discuss why or why not.
"  the module accuracy is 45.22% while 46% is better than random guessing, it is quite low
for a machine learning model, meaning it is wrong more than half the time
the confusion matrix shows that the model completely fails to predict the “fatal” class,
misclassifying all fatal accidents as either no-injury or non-fatal. This indicates poor 
class discrimination and suggests that the model is dominated by majority classes, likely
due to class imbalance and the large value of k selected or the 11 features provided 
might not be the right predictors for accident severity.Given the importance of correctly 
identifying severe accidents,this model would not be suitable for practical use."




### Proplem (4)
#(1) Remove observations with MAX_SEV = "no-injury"
df <- read.csv("accidents1030.csv")
df <- subset(df, MAX_SEV != "no-injury")
df$MAX_SEV <- factor(df$MAX_SEV)
dim(df)

#(2) generate training and holdout partitions (1/3 holdout

library(rsample)
set.seed(31)

split <- initial_split(df, prop = 2/3, strata = MAX_SEV)
train <- training(split)
hold  <- testing(split)

table(train$MAX_SEV)
table(hold$MAX_SEV)


#3) Fit a logistic regression model on the original training data


logit_orig <- glm(MAX_SEV ~ ., data = train, family = "binomial")

prob_orig <- predict(logit_orig, newdata = hold, type = "response")

pred_orig <- factor(
  ifelse(prob_orig >= 0.5, "non-fatal", "fatal"),
  levels = levels(hold$MAX_SEV)
)

library(caret)
cm_orig <- confusionMatrix(pred_orig, hold$MAX_SEV, positive = "fatal")
cm_orig

cm_orig$overall["Accuracy"]

#F-score

tp <- cm_orig$table["fatal", "fatal"]
fp <- cm_orig$table["fatal", "non-fatal"]
fn <- cm_orig$table["non-fatal", "fatal"]

precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
F1        <- 2 * precision * recall / (precision + recall)

c(precision = precision, recall = recall, F1 = F1)

#This model is trained on imbalanced data, so it tends to favor the majority class (non-fatal).

#(4) Does this seem to be a good model?

# this is not a good model 
#Although it has high overall accuracy, this is misleading due to severe class imbalance.
#The model has very low recall (10%) for fatal accidents, meaning it fails to identify most fatal cases.
#In safety-related problems, missing fatal cases is unacceptable, so accuracy alone is not sufficient
#High accuracy (94%) is misleading here. The model is simply predicting the majority class 
#"non-fatal") for almost every case. Because it only caught 1 out of 10 actual fatal accidents, it
#has very little predictive utility for safety purposes


#(5)Which resampling technique is less likely to work, and why?
  
#Under-sampling is less likely to work
#The fatal class is very small, undersampling would discard many non-fatal observations
#This causes severe information loss.
" We only have about 28 fatal cases in your training set. If we used under-sampling, 
we  would have to throw away almost all of your non-fatal data to match that number. 
we would end up with a tiny dataset, which is not enough for 
the model to learn reliable patterns without overfitting"

##(6) Apply the more appropriate resampling method (over-sampling)
library(ROSE)
set.seed(33)

train_over <- ovun.sample(
  MAX_SEV ~ .,
  data = train,
  method = "over",
  p = 0.5
)$data

table(train$MAX_SEV)
table(train_over$MAX_SEV)

#Over-sampling duplicates minority class observations so that both classes contribute equally during training.

##(7) Fit logistic regression on balanced data & evaluate on original holdout

logit_over <- glm(MAX_SEV ~ ., data = train_over, family = "binomial")

prob_over <- predict(logit_over, newdata = hold, type = "response")

pred_over <- factor(
  ifelse(prob_over >= 0.5, "non-fatal", "fatal"),
  levels = levels(hold$MAX_SEV)
)

cm_over <- confusionMatrix(pred_over, hold$MAX_SEV, positive = "fatal")
cm_over
#accuracy
cm_over$overall["Accuracy"]
# F-score
tp <- cm_over$table["fatal", "fatal"]
fp <- cm_over$table["fatal", "non-fatal"]
fn <- cm_over$table["non-fatal", "fatal"]

precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
F1        <- 2 * precision * recall / (precision + recall)

c(precision = precision, recall = recall, F1 = F1)
#The model is trained on balanced data but evaluated on the original holdout, ensuring a fair comparison.

##(8) Compare models from (3) and (7)

#Metric	            Original Model  	Over-sampled    ModeChange
#Accuracy	          93.9%	            27.37%	         Much worse
#Recall (Fatal)	    10.00%          	40.00%	          4× better
#Precision (Fatal)  33.33%	          3.13%	            Much worse
#F1-Score	           0.154	          0.058	           Worse
#Which is better?
  
#For catching fatal accidents: Over-sampled model is better (catches 4× more fatal cases)

#For overall accuracy: Original model is better

#For minimizing false alarms: Original model is better

#Even though the Accuracy dropped significantly (from 94% down to 27%), the recall improved
#from 10% to 40%. In a safety context, correctly identifying a fatality is much more important
#than overall accuracy. The second model is trying to find dangerous conditions
#rather than just guessing the safest outcome every time.
#However, if balancing false positives and false negatives is important, the original model has a higher F1-score.

##(9) Variable importance for the balanced logistic regression model

#  remove Intercept and get top 3
importance <- sort(abs(coef(logit_over)), decreasing = TRUE)
importance_no_int <- importance[names(importance) != "(Intercept)"]
importance_no_int 
head(importance_no_int, 3) #Top 3 predictors


#Problem 5
# Load necessary libraries
library(tidyverse)
library(caret)       # for data splitting and performance metrics
library(glmnet)      # for LASSO/ridge regression
library(Metrics)     # for MAE and RMSE
# 1. Load the dataset
df <- read.csv("powdermetallurgy.csv")
# Check dataset dimensions
dim(df)
# [1] 6253 8
#. Generate training and holdout partitions (1/3 holdout)
set.seed(42)
split <- createDataPartition(df$Shrinkage, p = 2/3, list = FALSE)
train <- df[split, ]
hold  <- df[-split, ]
# 2. Fit a multiple linear regression model
lm_model <- lm(Shrinkage ~ ., data = train)
# Make predictions on holdout
pred_lm <- predict(lm_model, newdata = hold)
#Compute MAE and RMSE
mae_lm <- mae(hold$Shrinkage, pred_lm)
rmse_lm <- rmse(hold$Shrinkage, pred_lm)
mae_lm
rmse_lm
#3) Does this seem to be a good model?
# yes this a good model
sd(df$Shrinkage)
 
#Comparison to baseline: the standard deviation of Shrinkage in the dataset is 
#approximately 0.617. Our model's RMSE (0.249) is significantly lower than the 
#standard deviation, which indicates the model is explaining a large portion of
#the variance rather than just predicting the mean.
#Error magnitude: An MAE of 0.200 means that on average, the predictions are 
#within 0.2 units of the actual value. Given that the Shrinkage values range from
# -2.2 to 0.76, an error of 0.2 is relatively small and suggests high predictive 
#accuracy for a manufacturing context.
# 4. Fit a LASSO regression model
# Prepare data for glmnet (needs numeric matrix)
x_train <- model.matrix(Shrinkage ~ ., train)[, -1]  # remove intercept column
y_train <- train$Shrinkage
x_hold  <- model.matrix(Shrinkage ~ ., hold)[, -1]
y_hold  <- hold$Shrinkage

set.seed(42)
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)  # LASSO (alpha=1)
# Best lambda
best_lambda <- lasso_model$lambda.min
# Predict on holdout
pred_lasso <- predict(lasso_model, s = best_lambda, newx = x_hold)
# Compute MAE and RMSE
mae_lasso <- mae(y_hold, pred_lasso)
rmse_lasso <- rmse(y_hold, pred_lasso)
mae_lasso
rmse_lasso
# 5. Compare models
cat("Linear Regression: MAE =", mae_lm, ", RMSE =", rmse_lm, "\n")
cat("LASSO Regression: MAE =", mae_lasso, ", RMSE =", rmse_lasso, "\n")
# The multiple linear regression and LASSO models produced nearly identical
# performance on the holdout data. Both models achieved almost the same MAE
# (0.199) and RMSE (0.248), indicating that regularization did not provide
# a meaningful improvement in predictive accuracy.
# This result is expected because the dataset is relatively large (6,253
# observations) compared to the small number of predictors, and there is
# limited multicollinearity. As a result, the ordinary least squares solution
# is already stable, and cross-validation selected a very small penalty,
# causing the LASSO model to closely approximate the linear regression model.


# Proplem 6
# classify the objects using the logistic regression model 
#The model equation for class "yes" is:
# Define the coefficients
intercept <- -3.485
coef_A1 <- 0.045
coef_A2 <- 0.003
threshold <- 0.5

# Define the unseen objects
objects <- data.frame(
  ID = c("O1", "O2"),
  A1 = c(47, 65),
  A2 = c(213, 276)
)

# Calculate the log-odds (z)
objects$z <- intercept + (coef_A1 * objects$A1) + (coef_A2 * objects$A2)

# Calculate the probability (p) using the sigmoid function
objects$prob <- 1 / (1 + exp(-objects$z))

# Classify based on the threshold
objects$Class <- ifelse(objects$prob >= threshold, "yes", "no")

# Display results
print(objects[, c("ID", "prob", "Class")])

#Object	A1	A2	 Logit	 Probability	Class (Threshold=0.5)
#O1	    47	213	-0.731	 0.325	        No
#O2	    65	276	 0.268	  0.567       	Yes

