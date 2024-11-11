install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
keras::install_keras()


trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$type <- as.factor(trainData$type)

nn_recipe <- recipe(formula=type~., data=trainData) %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn = factor) |> 
  step_dummy(color) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]


#preped <- prep(nn_recipe)
#bake(preped, new_data = trainData)


nn_model <- mlp(hidden_units = tune(),
                epochs = 50 #or 100 or 250
) %>%
set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels=10)

nn_workflow <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)


tuned_nn <- tune_grid(
  nn_workflow,
  resamples = cv_folds,
  grid = nn_tuneGrid,
  metrics = metric_set(accuracy)
)

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want
#best tuning results
best_params <- select_best(tuned_nn, metric = "accuracy")

# Finalize workflow with best parameters
final_workflow <- finalize_workflow(nn_workflow, best_params)

# Fit final model on training data
final_fit <- fit(final_workflow, data = trainData)

# Make predictions on the test set
predictions <- predict(final_fit, new_data = testData, type = "class")

## Format the Predictions for Submission to Kaggle
kaggle_submission <- predictions %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(id, .pred_class) |> 
  mutate(type = .pred_class) |> 
  select(-.pred_class)

## Write out the file
vroom_write(x=kaggle_submission, file="./nn.csv", delim=",")


# graph -------------------------------------------------------------------------

# Extract the results from the tuning grid
tuning_results <- tuned_nn %>%
  collect_metrics() %>%
  filter(.metric == "accuracy")

# Plot accuracy against hidden_units
accuracy_plot <- ggplot(tuning_results, aes(x = hidden_units, y = mean)) +
  geom_point() +
  geom_line() +
  labs(
    title = "Cross-Validation Accuracy by Hidden Units",
    x = "Hidden Units",
    y = "Mean Accuracy"
  ) +
  theme_minimal()

# Display the plot
print(accuracy_plot)
