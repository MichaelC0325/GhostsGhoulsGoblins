library(tidymodels)
library(embed) #target encoding
library(vroom)
library(parsnip)
library(parallel)
library(dbarts)
library(bonsai)
library(lightgbm)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$type <- as.factor(trainData$type)

my_recipe <- recipe(type~., data = trainData) |> 
  step_mutate_at(color, fn = factor)


#Bart model
bart_mod <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")
#boost model
boost_mod <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

#workflow
my_workflow <- workflow() %>%
  add_model(boost_mod) %>%
  add_recipe(my_recipe)

#grid
grid_vals <- grid_regular(
  tree_depth(range = c(1, 10)),    # Depth of each tree: limit from 1 to 10 for more shallow trees
  trees(range = c(1, 50)),          # Number of trees in the boosting process
  learn_rate(range = c(0.001, 0.1)),# Learning rate: small steps to prevent overfitting
  levels = 10
)


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
vroom_write(x=kaggle_submission, file="./Boost2.csv", delim=",")
