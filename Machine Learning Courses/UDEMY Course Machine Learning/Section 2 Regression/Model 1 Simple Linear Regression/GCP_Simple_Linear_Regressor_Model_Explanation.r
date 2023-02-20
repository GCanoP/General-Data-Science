# ===================================================
# ========== SIMPLE LINEAR REGRESSION IN R ==========
# ===================================================

# Import the dataset. 
dataset = read.csv("Salary_Data.csv")

# Split the datset. 
library(caTools)
set.seed(123)
split =  sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Create the linear regression model. 

regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# Prediction with test dataset. 
y_pred = predict(regressor, newdata = testing_set)

# Graphs for Prediction. 
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), color = "blue") +
  ggtitle("Salary  - Years of Experience") + 
  xlab(" Years of Experience") + 
  ylab("Salary")
