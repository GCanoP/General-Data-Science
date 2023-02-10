# Coding Example - Date January 27, 2023.
# Machine Learning Course UDEMY - Gerardo Cano Perea. 
# Basic Template for Data Managment. 

# Import dataset. 
dataset = read.csv('Data.csv')

# Replace NA values with the mean value of the column. 
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Categorical Variables. 
dataset$Country = factor(dataset$Country, 
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c(unique(dataset$Purchased)),
                           labels = c(sequence(length(unique(dataset$Purchased)))))

# Split training and test datasets. 
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Scaling Values.
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3]= scale(testing_set[,2:3])










