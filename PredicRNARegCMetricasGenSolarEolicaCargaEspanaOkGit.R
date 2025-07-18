# --- 1. INSTALACIÓN Y CARGA DE LIBRERÍAS ---

# Instalar paquetes (si no están instalados)
install.packages("tidyverse")     # Para manipulación de datos y gráficos
install.packages("lubridate")     # Para manejo de fechas
install.packages("neuralnet")     # Implementación de Redes Neuronales Artificiales (RNA)

# Cargar librerías necesarias
library(dplyr)
library(neuralnet)   # Para RNA
library(lubridate)
library(ggplot2)
library(tidyr)

# --- 2. LECTURA Y PREPARACIÓN DE DATOS (Sin cambios) ---

# Leer y preparar los datos
url1 <- "https://raw.githubusercontent.com/vneumannufprbr/TrabajosRStudio/main/energy_dataset.csv"
data <- read.csv(url1, stringsAsFactors = FALSE)

# Parsear y ordenar la columna de tiempo
data <- data %>%
  mutate(
    time = ymd_hms(time) # Parsear sin especificar la zona horaria
  ) %>%
  arrange(time)

# --- 3. DEFINICIÓN DE PARÁMETROS Y FUNCIONES AUXILIARES (Sin cambios en esta sección) ---

# Seleccionar variables objetivo
targets <- c("generation.solar", "generation.wind.onshore", "total.load.actual")
results <- list()
metrics <- list()

# Parámetros del modelo
window_size <- 24 # 24 horas, anterior 168 horas, Ventana de 7 días (24*7)
test_size <- 24 # 24 horas, anterior: 30 * 24, 30 días para evaluación
forecast_horizon <- 24 # 24 horas, anterior: 30 * 24, 30 días para pronóstico futuro

# Función para crear características
create_features <- function(serie, window) {
  n <- length(serie)
  features <- matrix(NA, nrow = n - window, ncol = window)
  for(i in 1:window) {
    features[, i] <- serie[i:(n - window + i - 1)]
  }
  target <- serie[(window + 1):n]
  colnames(features) <- paste0("X", 1:window)
  return(data.frame(features, target))
}

# Función para calcular métricas
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2, na.rm = TRUE))
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  ss_res <- sum((actual - predicted)^2, na.rm = TRUE)
  ss_tot <- sum((actual - mean(actual))^2, na.rm = TRUE)
  r_squared <- 1 - (ss_res/ss_tot)
  
  return(data.frame(R2 = r_squared, RMSE = rmse, MAE = mae))
}

# --- 4. EVALUACIÓN Y PRONÓSTICO CON REDES NEURONALES (RNA) ---

# Bucle para cada variable objetivo
for(target_var in targets) {
  cat("\nProcesando variable:", target_var, "\n")
  
  # Extraer serie completa y eliminar NAs
  serie_full <- data[[target_var]] %>% na.omit() %>% as.numeric()
  
  # 1. Evaluación en conjunto de prueba -------------------------------------------------
  # Dividir en entrenamiento y prueba
  n_full <- length(serie_full)
  train_series <- serie_full[1:(n_full - test_size)]
  test_series <- serie_full[(n_full - test_size + 1):n_full]
  
  # Crear matriz de entrenamiento
  train_data <- create_features(train_series, window_size)
  
  # Escalar datos de entrenamiento a [0, 1]
  maxs_train <- apply(train_data, 2, max)
  mins_train <- apply(train_data, 2, min)
  scale_train <- maxs_train - mins_train
  scale_train[scale_train == 0] <- 1
  scaled_train_data <- as.data.frame(scale(train_data, center = mins_train, scale = scale_train))
  
  # *** INICIO CORRECCIÓN 1: Asegurar que no hay valores no finitos ***
  scaled_train_data[] <- lapply(scaled_train_data, function(x) ifelse(is.finite(x), x, 0))
  # *** FIN CORRECCIÓN 1 ***
  
  # Crear la fórmula para la red neuronal
  predictor_vars <- colnames(scaled_train_data)[1:window_size]
  formula_str <- paste("target ~", paste(predictor_vars, collapse = " + "))
  nn_formula <- as.formula(formula_str)
  
  # Entrenar modelo de Red Neuronal
  cat("Entrenando RNA en datos de prueba... (puede tardar unos minutos)\n")
  nn_model <- neuralnet(nn_formula, 
                        data = scaled_train_data, 
                        hidden = c(2), # puede probar valores mayores (3,4,5...10) pero puede llevar horas o días para procesar
                        linear.output = TRUE, 
                        stepmax = 1e6)
  
  # Pronóstico recursivo para el conjunto de prueba
  last_window <- tail(train_series, window_size) 
  test_predictions <- numeric(test_size)
  
  mins_features <- mins_train[1:window_size]
  maxs_features <- maxs_train[1:window_size]
  min_target <- mins_train["target"]
  max_target <- maxs_train["target"]
  
  for(i in 1:test_size) {
    scale_features <- maxs_features - mins_features
    scale_features[scale_features == 0] <- 1
    last_window_scaled <- (last_window - mins_features) / scale_features
    
    test_input <- as.data.frame(t(last_window_scaled))
    colnames(test_input) <- predictor_vars
    
    pred_scaled <- predict(nn_model, newdata = test_input)
    
    scale_target <- max_target - min_target
    if (scale_target == 0) scale_target <- 1
    pred_actual <- pred_scaled[1,1] * scale_target + min_target
    
    # Validar la predicción ***
    if (!is.finite(pred_actual)) {
      pred_actual <- 0 # Si la predicción es NA/Inf, la reemplazamos por 0
    }
 
    test_predictions[i] <- pred_actual
    last_window <- c(last_window[-1], pred_actual)
  }
  
  # Calcular métricas de evaluación
  actual_test <- test_series[1:test_size]
  metrics[[target_var]] <- calculate_metrics(actual_test, test_predictions)
  
  # 2. Pronóstico futuro usando toda la data --------------------------------------------
  cat("Entrenando RNA en datos completos para pronóstico futuro...\n")
  
  full_data <- create_features(serie_full, window_size)
  
  maxs_full <- apply(full_data, 2, max)
  mins_full <- apply(full_data, 2, min)
  scale_full <- maxs_full - mins_full
  scale_full[scale_full == 0] <- 1
  scaled_full_data <- as.data.frame(scale(full_data, center = mins_full, scale = scale_full))
  scaled_full_data[] <- lapply(scaled_full_data, function(x) ifelse(is.finite(x), x, 0))
  
  nn_model_full <- neuralnet(nn_formula, 
                             data = scaled_full_data, 
                             hidden = c(2), # puede probar valores mayores (3,4,5...10, o más capas) pero puede llevar horas o días para procesar
                             linear.output = TRUE, 
                             stepmax = 1e6)
  
  last_window_full <- tail(serie_full, window_size)
  future_predictions <- numeric(forecast_horizon)
  
  mins_features_full <- mins_full[1:window_size]
  maxs_features_full <- maxs_full[1:window_size]
  min_target_full <- mins_full["target"]
  max_target_full <- maxs_full["target"]
  
  for(i in 1:forecast_horizon) {
    scale_features_full <- maxs_features_full - mins_features_full
    scale_features_full[scale_features_full == 0] <- 1
    last_window_full_scaled <- (last_window_full - mins_features_full) / scale_features_full
    
    test_input_full <- as.data.frame(t(last_window_full_scaled))
    colnames(test_input_full) <- predictor_vars
    
    pred_scaled <- predict(nn_model_full, newdata = test_input_full)
    
    scale_target_full <- max_target_full - min_target_full
    if (scale_target_full == 0) scale_target_full <- 1
    pred_actual <- pred_scaled[1,1] * scale_target_full + min_target_full
    
    if (!is.finite(pred_actual)) {
      pred_actual <- 0
    }
 
    future_predictions[i] <- pred_actual
    last_window_full <- c(last_window_full[-1], pred_actual)
  }
  
  results[[target_var]] <- future_predictions
}

# --- 5. VISUALIZACIÓN DE RESULTADOS ---

cat("\nMétricas de Evaluación en Conjunto de Prueba (usando RNA):\n")
for(target_var in targets) {
  cat("\nVariable:", target_var, "\n")
  print(metrics[[target_var]])
}

last_date <- tail(data$time, 1)
future_dates <- seq(last_date + hours(1), by = "hour", length.out = forecast_horizon)

forecast_df <- data.frame(
  time = future_dates,
  solar = results$generation.solar,
  wind_onshore = results$generation.wind.onshore,
  total_load = results$total.load.actual
)

forecast_df %>%
  gather(key = "variable", value = "value", -time) %>%
  ggplot(aes(x = time, y = value, color = variable)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(title = "Pronóstico a 24 horas usando Redes Neuronales (RNA)",
       x = "Fecha", y = "Valor") +
  theme_minimal()
