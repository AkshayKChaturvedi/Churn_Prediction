library(precrec)
library(e1071)
library(randomForest)
library(rpart)
library(gbm)
library(ggplot2)
library(psych)
library(reshape2)

train_data <- read.csv('C:/Users/Dell/Downloads/Train_data.csv')

test_data <- read.csv('C:/Users/Dell/Downloads/Test_data.csv')

# Remove three discrete numerical variables from train and test set
train_data <- train_data[,setdiff(names(train_data),c('total.intl.calls', 'number.vmail.messages', 'number.customer.service.calls'))]

test_data <- test_data[,setdiff(names(test_data),c('total.intl.calls', 'number.vmail.messages', 'number.customer.service.calls'))]

cat <- list('state', 'area.code', 'phone.number', 'international.plan', 'voice.mail.plan', 'Churn')

num <- setdiff(names(train_data),cat)

# --------------------------------Generic Functions-----------------------------

clean_data <- function(data, remove='None'){
  cleansed_data <- data[,setdiff(names(data),c('state', 'phone.number', 'area.code'))]
  if (remove == 'charges'){
      cleansed_data <- cleansed_data[,setdiff(names(cleansed_data),c('total.night.charge', 'total.day.charge', 'total.intl.charge',
                                                               'total.eve.charge'))]} 
    
  if (remove == 'minutes'){
      cleansed_data <- cleansed_data[,setdiff(names(cleansed_data),c('total.night.minutes', 'total.day.minutes', 'total.intl.minutes',
                                                               'total.eve.minutes'))]}
  cleansed_data$Churn <- as.integer(cleansed_data$Churn)
  cleansed_data$international.plan <- as.integer(cleansed_data$international.plan)
  cleansed_data$voice.mail.plan <- as.integer(cleansed_data$voice.mail.plan)
  
  cleansed_data$Churn[cleansed_data$Churn == 1] <- 0
  cleansed_data$Churn[cleansed_data$Churn == 2] <- 1
  
  cleansed_data$international.plan[cleansed_data$international.plan == 1] <- 0
  cleansed_data$international.plan[cleansed_data$international.plan == 2] <- 1
  
  cleansed_data$voice.mail.plan[cleansed_data$voice.mail.plan == 1] <- 0
  cleansed_data$voice.mail.plan[cleansed_data$voice.mail.plan == 2] <- 1
  
  cleansed_data$Churn <- as.factor(cleansed_data$Churn)
  
  return(cleansed_data)
}


outlier_removal <- function(un_data, cols){
  data <- un_data
  for (i in cols){
    print(i)
    if (class(data[,i]) != 'integer' | class(data[,i]) != 'numeric'){
      data$i <- as.integer(data[,i])
    }
    percentiles <- quantile(data[,i], c(.75, .25))
    q75 <- percentiles[[1]]
    q25 <- percentiles[[2]]
    iqr <- q75 - q25
    min <- q25 - (1.5*iqr)
    max <- q75 + (1.5*iqr)
    print(paste(q75,q25,iqr,min,max))
    data <- data[!(data[,i] < min | data[,i] > max), ]
    print(nrow(data))
  }
  return(data)
}

standardize_data <- function(data, source){
  
  # data : data frame to be transformed
  # source : data frame whose mean and standard deviation will be 
  # used to transform the 'data' mentioned above
  
  if (ncol(data) != ncol(source)){
    print('Please make sure data and source have same number of columns')
    return
  }
  col_means <- sapply(source, mean)
  print(col_means)
  col_sd <- sapply(source, sd)
  print(col_sd)
  for (i in 1:ncol(data)){
    data[,i] <- (data[,i] - col_means[i])/col_sd[i] 
  }
  return(data)
}

# --------------------------Exploratory Data Analysis--------------------------

target_distribution <- table(train_data$Churn)

# Unique value counts of target variable
print(target_distribution)

train_num <- train_data[,num]

train_num_melt <- melt(train_num)

train_cat <- train_data[,as.character(cat)]

for (i in 1:5){
  print(names(train_cat)[i])
  print(chisq.test(table(train_data$Churn, train_cat[,i])))
}

multi.hist(train_num, main = '', dcol=c('white','black'), 
           dlty=c("solid", "solid"), bcol = 'white', density=TRUE)

cor_mat <- cor(train_num)

melted_cormat <- melt(cor_mat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + theme(
    axis.text.x = element_text(
      angle = 45, vjust = 1, size = 10, hjust = 1), 
    axis.title.x = element_blank(), axis.title.y = element_blank())

ggplot(train_num_melt, aes(x = 0, y = value)) +
  geom_boxplot() + facet_wrap(~variable, scales='free') + 
  theme(axis.title.x=element_blank())

# -----------------------Predictive Modeling Experiments----------------------- 
  
clean_train <- clean_data(train_data)

# Dataset without any outliers
# clean_train <- outlier_removal(clean_train, num)

x_train <- clean_train[,!(names(clean_train) == 'Churn')]

y_train <- clean_train$Churn

clean_test <- clean_data(test_data)

x_test <- clean_test[,!(names(clean_test) == 'Churn')]

x_test <- standardize_data(x_test, x_train)
                   
y_test <- clean_test$Churn

clean_test[,!(names(clean_test) == 'Churn')] <- x_test

x_train <- standardize_data(x_train, x_train)

clean_train[,!(names(clean_train) == 'Churn')] <- x_train

# ****************************Logistic Regression******************************

set.seed(1)

lr <- glm(Churn ~ ., data = clean_train, family=binomial(link='logit'))

lr_predictions_train_prob <- predict(lr, type = 'response', x_train)

lr_predictions_train <- ifelse(lr_predictions_train_prob > 0.5, 1, 0)

lr_accuracy_train <- sum(lr_predictions_train == y_train)/nrow(clean_train)

lr_average_precision_train <- evalmod(scores = lr_predictions_train_prob, labels = y_train)

lr_predictions_test_prob <- predict(lr, x_test, type = 'response')

lr_predictions_test <- ifelse(lr_predictions_test_prob > 0.5, 1, 0)

lr_accuracy_test <- sum(lr_predictions_test == y_test)/nrow(clean_test)

lr_average_precision_test <- evalmod(scores = lr_predictions_test_prob, labels = y_test)

# ********************************Naive Bayes**********************************

set.seed(1)

nb <- naiveBayes(Churn ~ ., data = clean_train)

nb_predictions_train_prob <- predict(nb, type = 'raw', x_train)[, 2]

nb_predictions_train <- ifelse(nb_predictions_train_prob > 0.5, 1, 0)

nb_accuracy_train <- sum(nb_predictions_train == y_train)/nrow(clean_train)

nb_average_precision_train <- evalmod(scores = nb_predictions_train_prob, labels = y_train)

nb_predictions_test_prob <- predict(nb, x_test, type = 'raw')[, 2]

nb_predictions_test <- ifelse(nb_predictions_test_prob > 0.5, 1, 0)

nb_accuracy_test <- sum(nb_predictions_test == y_test)/nrow(clean_test)

nb_average_precision_test <- evalmod(scores = nb_predictions_test_prob, labels = y_test)

# **************************Support Vector Machine*****************************

set.seed(1)

svm_classifier <- svm(Churn ~ ., data = clean_train, probability=TRUE, type='C-classification')

svm_predictions_train_prob_temp <- predict(svm_classifier, probability=TRUE, x_train)

svm_predictions_train_prob <- attributes(svm_predictions_train_prob_temp)$probabilities[, 2]

svm_predictions_train <- ifelse(svm_predictions_train_prob > 0.5, 1, 0)

svm_accuracy_train <- sum(svm_predictions_train == y_train)/nrow(clean_train)

svm_average_precision_train <- evalmod(scores = svm_predictions_train_prob, labels = y_train)

svm_predictions_test_prob_temp <- predict(svm_classifier, x_test, probability=TRUE)

svm_predictions_test_prob <- attributes(svm_predictions_test_prob_temp)$probabilities[, 2]

svm_predictions_test <- ifelse(svm_predictions_test_prob > 0.5, 1, 0)

svm_accuracy_test <- sum(svm_predictions_test == y_test)/nrow(clean_test)

svm_average_precision_test <- evalmod(scores = svm_predictions_test_prob, labels = y_test)

# ***************************Gradient Boosted Tree*****************************

# Gradient Boosted Tree in R requires target to be integer rather 
# than factor that's why instead of using clean_train and clean_test
# itself, data has been prepared below for Gradient Boosted Tree only

gbt_train <- clean_train
 
gbt_train$Churn <- as.integer(gbt_train$Churn)
 
gbt_train$Churn[gbt_train$Churn == 1] <- 0
gbt_train$Churn[gbt_train$Churn == 2] <- 1

gbt_x_train <- gbt_train[,!(names(gbt_train) == 'Churn')]
 
gbt_y_train <- gbt_train$Churn
 
gbt_test <- clean_test

gbt_test$Churn <- as.integer(gbt_test$Churn)

gbt_test$Churn[gbt_test$Churn == 1] <- 0
gbt_test$Churn[gbt_test$Churn == 2] <- 1

gbt_x_test <- gbt_test[,!(names(gbt_test) == 'Churn')]
 
gbt_y_test <- gbt_test$Churn

set.seed(1)

gbt <- gbm(Churn ~ ., data = gbt_train, distribution = 'bernoulli', n.trees=100, shrinkage=0.1, interaction.depth=2)

gbt_predictions_train_prob <- predict(gbt, type = 'response', gbt_x_train, n.trees=100)

gbt_predictions_train <- ifelse(gbt_predictions_train_prob > 0.5, 1, 0)

gbt_accuracy_train <- sum(gbt_predictions_train == gbt_y_train)/nrow(gbt_train)

gbt_average_precision_train <- evalmod(scores = gbt_predictions_train_prob, labels = gbt_y_train)

gbt_predictions_test_prob <- predict(gbt, gbt_x_test, type = 'response', n.trees=100)

gbt_predictions_test <- ifelse(gbt_predictions_test_prob > 0.5, 1, 0)

gbt_accuracy_test <- sum(gbt_predictions_test == gbt_y_test)/nrow(gbt_test)

gbt_average_precision_test <- evalmod(scores = gbt_predictions_test_prob, labels = gbt_y_test)

# ********************************Decision Tree********************************

set.seed(1)

dt <- rpart(Churn ~ ., data = clean_train, method = 'class', parms = list(split = "information"))

dt_predictions_train_prob <- predict(dt, type = 'prob', x_train)[, 2]

dt_predictions_train <- ifelse(dt_predictions_train_prob > 0.5, 1, 0)

dt_accuracy_train <- sum(dt_predictions_train == y_train)/nrow(clean_train)

dt_average_precision_train <- evalmod(scores = dt_predictions_train_prob, labels = y_train)

dt_predictions_test_prob <- predict(dt, x_test, type = 'prob')[, 2]

dt_predictions_test <- ifelse(dt_predictions_test_prob > 0.5, 1, 0)

dt_accuracy_test <- sum(dt_predictions_test == y_test)/nrow(clean_test)

dt_average_precision_test <- evalmod(scores = dt_predictions_test_prob, labels = y_test)

# ********************************Random Forest********************************

set.seed(1)

rf <- randomForest(Churn ~ ., data = clean_train)

rf_predictions_train_prob <- predict(rf, type = 'prob', x_train)[, 2]

rf_predictions_train <- ifelse(rf_predictions_train_prob > 0.5, 1, 0)

rf_accuracy_train <- sum(rf_predictions_train == y_train)/nrow(clean_train)

rf_average_precision_train <- evalmod(scores = rf_predictions_train_prob, labels = y_train)

rf_predictions_test_prob <- predict(rf, x_test, type = 'prob')[, 2]

rf_predictions_test <- ifelse(rf_predictions_test_prob > 0.5, 1, 0)

rf_accuracy_test <- sum(rf_predictions_test == y_test)/nrow(clean_test)

rf_average_precision_test <- evalmod(scores = rf_predictions_test_prob, labels = y_test)

# **************************Precision-Recall Curves*****************************

labels <- join_labels(y_test, y_test, y_test, y_test, y_test, y_test)
scores <- join_scores(lr_predictions_test_prob, nb_predictions_test_prob, 
                      svm_predictions_test_prob, gbt_predictions_test_prob,
                      dt_predictions_test_prob, rf_predictions_test_prob)

msmdat3 <- mmdata(scores, labels, modnames = c('Logistic Regression',
                                               'Naive Bayes', 'Support Vector Machine', 
                                               'Gradient Boosted Tree', 'Decision Tree', 
                                               'Random Forest'), dsids = c(1, 2, 3, 4, 5, 6))
mscurves <- evalmod(msmdat3)

autoplot(mscurves, 'PRC')

# ******Write the predictions in a csv file with features used******

clean_test$rf_predictions_test <- rf_predictions_test

write.csv(clean_test, 
          'C:/Users/Dell/Desktop/R_Churn_test_set_with_predictions.csv', 
          row.names=FALSE)

# ***********Print performance metrics of the best model************

sprintf('Training set accuracy of the best model i.e. Random Forest 
        is : %f', rf_accuracy_train)

sprintf('Test set accuracy of the best model i.e. Random Forest 
        is : %f', rf_accuracy_test)

sprintf('Average precision score of the best model i.e. Random Forest 
        on training set is : %f', attr(rf_average_precision_train, 'aucs')[2,4])

sprintf('Average precision score of the best model i.e. Random Forest 
        on test set is : %f', attr(rf_average_precision_test, 'aucs')[2,4])

# *************************************End**************************************
