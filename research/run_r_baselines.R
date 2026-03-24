# run_r_baselines.R — GWR and GTWR using R's GWmodel package
# ===========================================================
# Uses preprocessed data exported by export_for_r.py (log1p + StandardScaler)
# so results are directly comparable to the Python GNN and Python GWR runs.
#
# Models:
#   GWR   — monthly 1° spatial, 80/20 spatial holdout, adaptive bisquare
#   GTWR  — daily 1° grid, n=6000 sample, UTC timestamps, adaptive bisquare
#            Prediction at test locations via regression.points argument
#
# Key difference vs thesis:
#   Timezone bug fixed (UTC, not 12AM Pacific) — temporal distances in GTWR
#   were corrupted in the original dissertation.
#
# Results appended to data/processed/baseline_results.txt
#
# Run:
#   Rscript research/run_r_baselines.R

suppressPackageStartupMessages({
  library(GWmodel)
  library(sp)
})

# ---------------------------------------------------------------------------
# Paths — works whether called from repo root or research/
# ---------------------------------------------------------------------------
args_vec <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", args_vec, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(sub("--file=", "", file_arg))
} else {
  script_dir <- "research"
}
root         <- normalizePath(file.path(script_dir, ".."))
data_dir     <- file.path(root, "data", "processed")
results_path <- file.path(data_dir, "baseline_results.txt")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_msg <- function(msg) {
  ts  <- format(Sys.time(), "[%H:%M:%S]")
  out <- paste0(ts, " ", msg)
  cat(out, "\n", sep = "")
  cat(out, "\n", sep = "", file = results_path, append = TRUE)
}

log_metrics <- function(label, r2, rmse, mae, elapsed = NULL) {
  t_str <- if (!is.null(elapsed)) sprintf("  elapsed: %.1fs", elapsed) else ""
  log_msg(sprintf("  %-28s R²=%.4f  RMSE=%.4f  MAE=%.4f%s",
                   label, r2, rmse, mae, t_str))
}

calc_metrics <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  list(
    r2   = 1 - ss_res / ss_tot,
    rmse = sqrt(mean((y_true - y_pred)^2)),
    mae  = mean(abs(y_true - y_pred))
  )
}

inv_scale <- function(y_scaled, pan_mean, pan_std) {
  y_scaled * pan_std + pan_mean
}

latlon_to_metres <- function(lat, lon) {
  R    <- 6371000
  lat0 <- 37 * pi / 180          # CONUS centre
  x    <- R * (lon * pi / 180) * cos(lat0)
  y    <- R * (lat * pi / 180)
  cbind(x, y)
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
cat(sprintf("\n%s\n", strrep("=", 60)), file = results_path, append = TRUE)
cat(sprintf("R run started: %s\n", Sys.time()),  file = results_path, append = TRUE)
cat(sprintf("GWmodel version: %s\n", packageDescription("GWmodel")$Version),
    file = results_path, append = TRUE)
cat(sprintf("%s\n", strrep("=", 60)), file = results_path, append = TRUE)

scalers       <- read.csv(file.path(data_dir, "r_scaler_params.csv"))
gwr_pan_mean  <- scalers$pan_mean[scalers$dataset == "gwr"]
gwr_pan_std   <- scalers$pan_std [scalers$dataset == "gwr"]
gtwr_pan_mean <- scalers$pan_mean[scalers$dataset == "gtwr"]
gtwr_pan_std  <- scalers$pan_std [scalers$dataset == "gtwr"]

# ===========================================================================
# GWR
# ===========================================================================
log_msg(paste0("\n", strrep("=", 60)))
log_msg("GWR (R/GWmodel) — monthly 1 degree grid, adaptive bisquare, AICc")
log_msg(strrep("=", 60))

train_raw <- read.csv(file.path(data_dir, "gwr_train.csv"))
test_raw  <- read.csv(file.path(data_dir, "gwr_test.csv"))
log_msg(sprintf("  Train cells: %d    Test cells: %d", nrow(train_raw), nrow(test_raw)))

train_coords <- latlon_to_metres(train_raw$lat, train_raw$lon)
test_coords  <- latlon_to_metres(test_raw$lat,  test_raw$lon)

train_sp <- SpatialPointsDataFrame(train_coords, train_raw)
test_sp  <- SpatialPointsDataFrame(test_coords,  test_raw)

gwr_formula <- mean_pan_scaled ~ lightning_3d_count + fire_3d_count +
                                  mean_co + lat_scaled + lon_scaled

log_msg("  Selecting GWR bandwidth (adaptive bisquare, AICc)...")
t0  <- proc.time()["elapsed"]
bw_gwr <- bw.gwr(
  gwr_formula,
  data     = train_sp,
  approach = "AICc",
  kernel   = "bisquare",
  adaptive = TRUE,
  longlat  = FALSE
)
log_msg(sprintf("  Optimal bandwidth: %.0f neighbours  (%.1fs)",
                bw_gwr, proc.time()["elapsed"] - t0))

log_msg("  Fitting GWR on train set...")
t1 <- proc.time()["elapsed"]
gwr_fit <- gwr.basic(
  gwr_formula,
  data     = train_sp,
  bw       = bw_gwr,
  kernel   = "bisquare",
  adaptive = TRUE,
  longlat  = FALSE
)
yp_tr <- inv_scale(as.numeric(gwr_fit$SDF$yhat), gwr_pan_mean, gwr_pan_std)
yt_tr <- inv_scale(train_raw$mean_pan_scaled,     gwr_pan_mean, gwr_pan_std)
m     <- calc_metrics(yt_tr, yp_tr)
log_metrics("GWR R (Train)", m$r2, m$rmse, m$mae,
            elapsed = proc.time()["elapsed"] - t1)

log_msg("  Predicting GWR on test set (gwr.predict)...")
t2 <- proc.time()["elapsed"]
gwr_pred <- gwr.predict(
  gwr_formula,
  data        = train_sp,
  predictdata = test_sp,
  bw          = bw_gwr,
  kernel      = "bisquare",
  adaptive    = TRUE,
  longlat     = FALSE
)
yp_te <- inv_scale(as.numeric(gwr_pred$SDF$prediction), gwr_pan_mean, gwr_pan_std)
yt_te <- inv_scale(test_raw$mean_pan_scaled,            gwr_pan_mean, gwr_pan_std)
m     <- calc_metrics(yt_te, yp_te)
log_metrics("GWR R (Test)", m$r2, m$rmse, m$mae,
            elapsed = proc.time()["elapsed"] - t2)
log_msg("  Thesis GWR baseline:          R2=0.361  RMSE=0.078  MAE=0.030  (LOO-CV, not holdout)")

# ===========================================================================
# GTWR
# ===========================================================================
log_msg(paste0("\n", strrep("=", 60)))
log_msg("GTWR (R/GWmodel) — daily 1 degree grid, UTC timestamps, n=6000 sample")
log_msg(strrep("=", 60))
log_msg("  Key change vs thesis: UTC timestamps (timezone bug fixed)")

gtwr_raw <- read.csv(file.path(data_dir, "gtwr_sample.csv"))
train_g  <- gtwr_raw[gtwr_raw$split == "train", ]
test_g   <- gtwr_raw[gtwr_raw$split == "test",  ]
log_msg(sprintf("  Train: %d rows    Test: %d rows", nrow(train_g), nrow(test_g)))

train_coords_g <- latlon_to_metres(train_g$lat, train_g$lon)
test_coords_g  <- latlon_to_metres(test_g$lat,  test_g$lon)

train_sp_g <- SpatialPointsDataFrame(train_coords_g, train_g)
test_sp_g  <- SpatialPointsDataFrame(test_coords_g,  test_g)

# Temporal coordinates: normalised DOY (same scale as spatial in metres)
doy_max <- max(gtwr_raw$doy)
t_train <- (train_g$doy / doy_max) * 1e6
t_test  <- (test_g$doy  / doy_max) * 1e6

gtwr_formula <- mean_pan_scaled ~ lightning_3d_count + fire_3d_count + mean_co

log_msg("  Selecting GTWR bandwidth (adaptive bisquare, CV)...")
log_msg("  This is the slow step — estimating 30-120 min.")
t3 <- proc.time()["elapsed"]
st_bw <- bw.gtwr(
  gtwr_formula,
  data     = train_sp_g,
  obs.tv   = t_train,
  approach = "CV",
  kernel   = "bisquare",
  adaptive = TRUE,
  longlat  = FALSE,
  lamda    = 0.05         # space-time ratio; 0.05 = thesis default
)
log_msg(sprintf("  Optimal st.bw: %.2f  (%.1fs)", st_bw,
                proc.time()["elapsed"] - t3))

log_msg("  Fitting GTWR on train set, predicting at test locations...")
t4 <- proc.time()["elapsed"]
gtwr_fit <- gtwr(
  gtwr_formula,
  data              = train_sp_g,
  regression.points = test_sp_g,    # predict at test locations
  obs.tv            = t_train,
  reg.tv            = t_test,
  st.bw             = st_bw,
  kernel            = "bisquare",
  adaptive          = TRUE,
  longlat           = FALSE,
  lamda             = 0.05
)

# gtwr() fitted values are at regression.points when supplied
yp_te_g <- inv_scale(as.numeric(gtwr_fit$SDF$yhat), gtwr_pan_mean, gtwr_pan_std)
yt_te_g <- inv_scale(test_g$mean_pan_scaled,         gtwr_pan_mean, gtwr_pan_std)
m_te    <- calc_metrics(yt_te_g, yp_te_g)
log_metrics("GTWR R (Test)", m_te$r2, m_te$rmse, m_te$mae,
            elapsed = proc.time()["elapsed"] - t4)

# Also fit on train only for train metrics
log_msg("  Fitting GTWR on train set only for train metrics...")
t5 <- proc.time()["elapsed"]
gtwr_train_only <- gtwr(
  gtwr_formula,
  data     = train_sp_g,
  obs.tv   = t_train,
  st.bw    = st_bw,
  kernel   = "bisquare",
  adaptive = TRUE,
  longlat  = FALSE,
  lamda    = 0.05
)
yp_tr_g <- inv_scale(as.numeric(gtwr_train_only$SDF$yhat), gtwr_pan_mean, gtwr_pan_std)
yt_tr_g <- inv_scale(train_g$mean_pan_scaled,              gtwr_pan_mean, gtwr_pan_std)
m_tr    <- calc_metrics(yt_tr_g, yp_tr_g)
log_metrics("GTWR R (Train)", m_tr$r2, m_tr$rmse, m_tr$mae,
            elapsed = proc.time()["elapsed"] - t5)

log_msg("  Thesis GTWR baseline:         R2=0.227  RMSE=0.184  MAE=0.071  (LOO-CV, 6% data, timezone bug)")
log_msg(sprintf("  Total R run elapsed:          %.1f min",
                (proc.time()["elapsed"] - t0) / 60))
log_msg("\nAll R baselines complete.")
