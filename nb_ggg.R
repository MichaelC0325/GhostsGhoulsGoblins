library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(naivebayes)
library(parsnip)
library(parallel)
library(rules)
library(themis)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$type <- as.factor(trainData$type)

my_recipe <- recipe(type~., data = trainData) |> 
  step_mutate_at(color, fn = factor) |> 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) |> 
  step_zv(all_numeric_predictors()) |> 
  step_range(all_numeric_predictors(), min = 0, max = 1) |> 
  step_smote(all_outcomes(), neighbors = 3)
  

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
# Set up cross-validation
cv_folds <- vfold_cv(trainData, v = 5)

# Define tuning grid
nb_grid <- grid_regular(
  Laplace(),
  smoothness(),
  levels = 5
)

# Tune the model
tuned_results <- tune_grid(
  nb_wf,
  resamples = cv_folds,
  grid = nb_grid,
  metrics = metric_set(accuracy)
)

# Select the best parameters based on accuracy
best_params <- select_best(tuned_results, metric = "accuracy")

# Finalize the workflow with the best parameters
final_nb_wf <- finalize_workflow(nb_wf, best_params)

# Fit the final model on the entire training data
final_model_nb <- fit(final_nb_wf, data = trainData)

# Make predictions on the test set
predictions <- predict(final_model_nb, new_data = testData, type = "class")

## Format the Predictions for Submission to Kaggle
kaggle_submission <- predictions %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(id, .pred_class) |> 
  mutate(type = .pred_class) |> 
  select(-.pred_class)

## Write out the file
vroom_write(x=kaggle_submission, file="./NaiveBayes9.csv", delim=",")
