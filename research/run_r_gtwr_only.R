# run_r_gtwr_only.R — GTWR only, n=2000 sample
# Rscript research/run_r_gtwr_only.R

suppressPackageStartupMessages({ library(GWmodel); library(sp) })

args_vec <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", args_vec, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(sub("--file=", "", file_arg)) else "research"
root         <- normalizePath(file.path(script_dir, ".."))
data_dir     <- file.path(root, "data", "processed")
results_path <- file.path(data_dir, "baseline_results.txt")

log_msg <- function(msg) {
  ts  <- format(Sys.time(), "[%H:%M:%S]")
  out <- paste0(ts, " ", msg)
  cat(out, "\n", sep = "")
  cat(out, "\n", sep = "", file = results_path, append = TRUE)
}
log_metrics <- function(label, r2, rmse, mae, elapsed = NULL) {
  t_str <- if (!is.null(elapsed)) sprintf("  elapsed: %.1fs", elapsed) else ""
  log_msg(sprintf("  %-28s R²=%.4f  RMSE=%.4f  MAE=%.4f%s", label, r2, rmse, mae, t_str))
}
calc_metrics <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2)
  list(r2   = 1 - ss_res / sum((y_true - mean(y_true))^2),
       rmse = sqrt(mean((y_true - y_pred)^2)),
       mae  = mean(abs(y_true - y_pred)))
}
inv_scale <- function(y_s, mn, sd) y_s * sd + mn
latlon_to_metres <- function(lat, lon) {
  R <- 6371000; lat0 <- 37 * pi / 180
  cbind(R * (lon * pi / 180) * cos(lat0), R * (lat * pi / 180))
}

scalers       <- read.csv(file.path(data_dir, "r_scaler_params.csv"))
gtwr_pan_mean <- scalers$pan_mean[scalers$dataset == "gtwr"]
gtwr_pan_std  <- scalers$pan_std [scalers$dataset == "gtwr"]

log_msg(paste0("\n", strrep("=", 60)))
log_msg("GTWR (R/GWmodel) — daily 1 deg grid, UTC timestamps, n=2000 sample")
log_msg(strrep("=", 60))
log_msg("  Key change vs thesis: UTC timestamps (timezone bug fixed)")

gtwr_raw <- read.csv(file.path(data_dir, "gtwr_sample.csv"))
train_g  <- gtwr_raw[gtwr_raw$split == "train", ]
test_g   <- gtwr_raw[gtwr_raw$split == "test",  ]
log_msg(sprintf("  Train: %d rows    Test: %d rows", nrow(train_g), nrow(test_g)))

train_sp_g <- SpatialPointsDataFrame(latlon_to_metres(train_g$lat, train_g$lon), train_g)
test_sp_g  <- SpatialPointsDataFrame(latlon_to_metres(test_g$lat,  test_g$lon),  test_g)

doy_max <- max(gtwr_raw$doy)
t_train <- (train_g$doy / doy_max) * 1e6
t_test  <- (test_g$doy  / doy_max) * 1e6

gtwr_formula <- mean_pan_scaled ~ lightning_3d_count + fire_3d_count + mean_co

log_msg("  Selecting GTWR bandwidth (adaptive bisquare, CV)...")
t0 <- proc.time()["elapsed"]
st_bw <- bw.gtwr(gtwr_formula, data=train_sp_g, obs.tv=t_train,
                  approach="CV", kernel="bisquare", adaptive=TRUE,
                  longlat=FALSE, lamda=0.05)
log_msg(sprintf("  Optimal st.bw: %.2f  (%.1fs)", st_bw, proc.time()["elapsed"] - t0))

log_msg("  Fitting GTWR, predicting at test locations...")
t1 <- proc.time()["elapsed"]
gtwr_fit <- gtwr(gtwr_formula, data=train_sp_g,
                 regression.points=test_sp_g,
                 obs.tv=t_train, reg.tv=t_test,
                 st.bw=st_bw, kernel="bisquare", adaptive=TRUE,
                 longlat=FALSE, lamda=0.05)
yp_te <- inv_scale(as.numeric(gtwr_fit$SDF$yhat), gtwr_pan_mean, gtwr_pan_std)
yt_te <- inv_scale(test_g$mean_pan_scaled,         gtwr_pan_mean, gtwr_pan_std)
m_te  <- calc_metrics(yt_te, yp_te)
log_metrics("GTWR R (Test)", m_te$r2, m_te$rmse, m_te$mae,
            elapsed=proc.time()["elapsed"] - t1)

log_msg("  Fitting GTWR on train only for train metrics...")
t2 <- proc.time()["elapsed"]
gtwr_tr <- gtwr(gtwr_formula, data=train_sp_g, obs.tv=t_train,
                st.bw=st_bw, kernel="bisquare", adaptive=TRUE,
                longlat=FALSE, lamda=0.05)
yp_tr <- inv_scale(as.numeric(gtwr_tr$SDF$yhat), gtwr_pan_mean, gtwr_pan_std)
yt_tr <- inv_scale(train_g$mean_pan_scaled,       gtwr_pan_mean, gtwr_pan_std)
m_tr  <- calc_metrics(yt_tr, yp_tr)
log_metrics("GTWR R (Train)", m_tr$r2, m_tr$rmse, m_tr$mae,
            elapsed=proc.time()["elapsed"] - t2)

log_msg("  Thesis GTWR:                  R2=0.227  RMSE=0.184  MAE=0.071  (LOO-CV, 6% data, timezone bug)")
log_msg(sprintf("  Total elapsed: %.1f min", (proc.time()["elapsed"] - t0) / 60))
log_msg("GTWR complete.")
