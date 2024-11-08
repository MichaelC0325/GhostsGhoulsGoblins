library(tidymodels)
library(embed) #target encoding
library(vroom)
library(parsnip)
library(parallel)

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


# Random Forest -----------------------------------------------------------

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$type <- as.factor(trainData$type)

my_recipe <- recipe(type~., data = trainData) |> 
  step_mutate_at(color, fn = factor)


#model
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#workflow
my_workflow <- workflow() %>%
  add_model(forest_mod) %>%
  add_recipe(my_recipe)

#grid of tuning values
grid_vals <- grid_regular(mtry(range = c(1, ncol(trainData) - 1)),
                          min_n(),
                          levels = 10)


#K-fold cross-validation
cv_folds <- vfold_cv(trainData, v = 5, repeats = 1)

#give computer more cores to work with
cl <- makePSOCKcluster(6)
doParallel::registerDoParallel(cl)
# Tune the model using grid search
tune_results <- tune_grid(
  my_workflow,
  resamples = cv_folds,
  grid = grid_vals,
  control = control_grid(save_pred = TRUE), #control = control_grid(verbose = True) doesn't work with parallel
  metrics = metric_set(accuracy)
)
#Stop the cluster
stopCluster(cl) #invalid connection error will happen if not stopped

#best tuning results
best_params <- select_best(tune_results, metric = "accuracy")

# Finalize workflow with best parameters
final_workflow <- finalize_workflow(my_workflow, best_params)

# Fit final model on training data
final_fit <- fit(final_workflow, data = trainData)

# Make predictions on the test set
predictions <- predict(final_fit, new_data = testData, type = "class")

# View predictions
predictions

## Format the Predictions for Submission to Kaggle
kaggle_submission <- predictions %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(id, .pred_class) |> 
  mutate(type = .pred_class) |> 
  select(-.pred_class)

## Write out the file
vroom_write(x=kaggle_submission, file="./RandomForest2.csv", delim=",")

