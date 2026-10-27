
# ============================================================
# LAB 7: CLASSIFICATION WITH NEURAL NETWORK IN R
# BIT33603 DATA MINING
# Student Name: MOHAMMED ZOALKEFL MOHAMMED HUSSEN
# Matric Number: AI240276
# ============================================================

# Install neuralnet if necessary
if (!requireNamespace("neuralnet", quietly = TRUE)) {
  install.packages("neuralnet")
}

library(neuralnet)

# Clear environment (except loaded packages)
rm(list = ls())

# Load Iris dataset
data(iris)

# ============================================================
# PART A: LAB ACTIVITIES (80% TRAINING, 20% TESTING)
# ============================================================

# ACTIVITY 1: DATA PREPARATION
cat("\n===== ACTIVITY 1: DATA PREPARATION =====\n")

print(head(iris))
print(summary(iris))

set.seed(245)

train_indices <- sample(
  1:nrow(iris),
  floor(0.80 * nrow(iris))
)

train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]

cat("Total records:", nrow(iris), "\n")
cat("Training records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")

# ============================================================
# ACTIVITY 2: BUILD NEURAL NETWORK
# ============================================================

cat("\n===== ACTIVITY 2: BUILD NEURAL NETWORK =====\n")

species_names <- levels(iris$Species)

# Convert target species to numeric indicator columns
for (species in species_names) {
  train_data[[species]] <- as.numeric(
    train_data$Species == species
  )
}

nn_formula <- as.formula(
  paste(
    paste(species_names, collapse = " + "),
    "~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
  )
)

set.seed(245)

model <- neuralnet(
  formula = nn_formula,
  data = train_data,
  hidden = c(4, 2),
  linear.output = FALSE,
  stepmax = 1e7
)

print(model$result.matrix)

# ============================================================
# ACTIVITY 3: EVALUATE THE MODEL
# ============================================================

cat("\n===== ACTIVITY 3: MODEL EVALUATION =====\n")

pred <- predict(model, test_data)

prediction_label <- factor(
  species_names[max.col(pred)],
  levels = species_names
)

conf_matrix <- table(
  Actual = test_data$Species,
  Predicted = prediction_label
)

cat("\nCONFUSION MATRIX:\n")
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

tp <- diag(conf_matrix)

precision <- tp / pmax(colSums(conf_matrix), 1)
recall <- tp / pmax(rowSums(conf_matrix), 1)

f_score <- ifelse(
  precision + recall > 0,
  2 * precision * recall / (precision + recall),
  0
)

metrics <- data.frame(
  Species = species_names,
  Precision = round(precision, 4),
  Recall = round(recall, 4),
  F_Score = round(f_score, 4)
)

cat("\nAccuracy:", round(accuracy, 4), "\n")
cat("Accuracy (%):", round(accuracy * 100, 2), "%\n")

print(metrics)

# ============================================================
# ACTIVITY 4: VISUALIZE NEURAL NETWORK
# ============================================================

cat("\n===== ACTIVITY 4: NEURAL NETWORK PLOT =====\n")

plot(model, rep = "best")

# Save the network plot
png("Lab7_NeuralNetwork_80_20.png",
    width = 1200, height = 800)

plot(model, rep = "best")

dev.off()

# ============================================================
# PART B: FINAL ASSESSMENT
# 70% TRAINING AND 30% TESTING
# ============================================================

cat("\n===== FINAL ASSESSMENT: 70/30 SPLIT =====\n")

# STEP 1: DATA SPLITTING
set.seed(245)

train_indices_70 <- sample(
  1:nrow(iris),
  floor(0.70 * nrow(iris))
)

train_data_70 <- iris[train_indices_70, ]
test_data_70 <- iris[-train_indices_70, ]

cat("Total Records:", nrow(iris), "\n")
cat("Training Records:", nrow(train_data_70), "\n")
cat("Testing Records:", nrow(test_data_70), "\n")

# ============================================================
# STEP 2: BUILD CLASSIFICATION MODEL
# ============================================================

species_names_70 <- levels(iris$Species)

for (species in species_names_70) {
  train_data_70[[species]] <- as.numeric(
    train_data_70$Species == species
  )
}

nn_formula_70 <- as.formula(
  paste(
    paste(species_names_70, collapse = " + "),
    "~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
  )
)

set.seed(245)

model_70 <- neuralnet(
  formula = nn_formula_70,
  data = train_data_70,
  hidden = c(4, 2),
  linear.output = FALSE,
  stepmax = 1e7
)

cat("\nNEURAL NETWORK TRAINING RESULTS:\n")
print(model_70$result.matrix)

# ============================================================
# STEP 3: MODEL EVALUATION
# ============================================================

pred_70 <- predict(model_70, test_data_70)

prediction_label_70 <- factor(
  species_names_70[max.col(pred_70)],
  levels = species_names_70
)

conf_matrix_70 <- table(
  Actual = test_data_70$Species,
  Predicted = prediction_label_70
)

cat("\nCONFUSION MATRIX (70:30):\n")
print(conf_matrix_70)

accuracy_70 <- sum(diag(conf_matrix_70)) /
  sum(conf_matrix_70)

tp_70 <- diag(conf_matrix_70)

precision_70 <- tp_70 /
  pmax(colSums(conf_matrix_70), 1)

recall_70 <- tp_70 /
  pmax(rowSums(conf_matrix_70), 1)

f_score_70 <- ifelse(
  precision_70 + recall_70 > 0,
  2 * precision_70 * recall_70 /
    (precision_70 + recall_70),
  0
)

metrics_70 <- data.frame(
  Species = species_names_70,
  Precision = round(precision_70, 4),
  Recall = round(recall_70, 4),
  F_Score = round(f_score_70, 4)
)

cat("\nFINAL ASSESSMENT RESULTS:\n")
cat("Accuracy:", round(accuracy_70, 4), "\n")
cat("Accuracy (%):",
    round(accuracy_70 * 100, 2), "%\n")

print(metrics_70)

# ============================================================
# STEP 4: VISUALIZE NEURAL NETWORK (70:30)
# ============================================================

plot(model_70, rep = "best")

# Save final assessment network plot
png("Lab7_NeuralNetwork_70_30.png",
    width = 1200, height = 800)

plot(model_70, rep = "best")

dev.off()

cat("\n===== LAB 7 COMPLETED =====\n")
