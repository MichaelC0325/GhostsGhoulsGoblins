library(tidymodels)
library(embed) #target encoding
library(vroom)

missSet <- vroom("trainWithMissingValues.csv")
trainSet <- vroom("train.csv")

#colMeans(is.na(missSet))

missSet <- missSet %>%
  mutate(across(where(is.character), as.factor))

my_recipe <- recipe(type~., data = missSet) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, type), neighbors = 5) |> 
  step_impute_knn(rotting_flesh, impute_with = imp_vars(hair_length, color, type), neighbors = 5) |> 
  step_impute_knn(bone_length, impute_with = imp_vars(rotting_flesh, hair_length, has_soul, color, type), neighbors = 5)

preped <- prep(my_recipe)
imputedSet <- bake(preped, new_data = missSet)

rmse_vec(trainSet[is.na(missSet)], imputedSet[is.na(missSet)])
