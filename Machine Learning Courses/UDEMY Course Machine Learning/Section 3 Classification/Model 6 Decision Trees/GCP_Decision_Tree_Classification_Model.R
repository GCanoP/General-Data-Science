# =====================================================================
# ========== DECISION TREES MODEL FOR CASSIFICATION PROBLEMS ==========
# =====================================================================

# Import the Dataset.
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[,3:5]

# Coding Variables as Factors. 
dataset$Purchased = factor(dataset$Purchased, levels = c(unique(dataset$Purchased)))

# Split the Dataset. 
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Create the Data Model.
library(rpart)
classifier = rpart(formula = Purchased ~ ., data = training_set)

# Predicting Results
y_pred = predict(classifier, newdata = testing_set[, -3], type = "class")

# Confusion Matrix. 
cm = table(testing_set[,3], y_pred)

# Data Visualisation. 
library(ElemStatLearn)
# IN PROGRESS

# Representación del árbol de clasificación
plot(classifier)
text(classifier)

