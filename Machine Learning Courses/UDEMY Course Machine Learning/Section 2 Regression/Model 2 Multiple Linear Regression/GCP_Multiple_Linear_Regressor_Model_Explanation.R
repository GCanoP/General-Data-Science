# ================================================
# ========== MULTIPLE LINEAR REGRESSION ==========
# ================================================

# Import dataset. 
datasets = read.csv("50_Startups.csv")

# Coding the cathegorical variables. 
datasets$State = factor(datasets$State,
                        levels = c(unique(datasets$State)),
                        labels = c(sequence(length(unique(datasets$State)))))

# Split the dataset. 
library(caTools)
set.seed(123)
split = sample.split(datasets$Profit, SplitRatio = 0.8)
training_set = subset(datasets, split == TRUE)
testing_set = subset(datasets, split == FALSE)

# Fitting the model. 
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
                data = training_set)

# Predicting Results.
y_pred = predict(regression, newdata = testing_set)

# Building a backward elimination model.
BackwardElimination <- function(x, s1){
  numVars = length(x)
  for(i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x )
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > s1){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars -1
  }
  return(summary(regressor))
}


SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
BackwardElimination(training_set, SL)








