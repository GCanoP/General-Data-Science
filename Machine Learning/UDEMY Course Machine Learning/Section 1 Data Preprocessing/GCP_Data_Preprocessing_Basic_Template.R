# Coding Example - Date January 27, 2023.
# Machine Learning Course UDEMY - Gerardo Cano Perea. 
# Basic Template for Data Managment. 

# Import Dataset. 
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


