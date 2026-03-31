args <- commandArgs(trailingOnly=TRUE)

get_arg <- function(name, default) {
  idx <- which(args == name)
  if (length(idx) == 0) return(default)
  if (idx == length(args)) return(default)
  return(args[idx + 1])
}

DATASET <- get_arg('--dataset', 'laser')
SEED <- as.integer(get_arg('--seed', '0'))
SAMPLES <- as.integer(get_arg('--samples', '1000'))
REPEATS <- as.integer(get_arg('--repeats', '100'))
WARMUP <- as.integer(get_arg('--warmup', '5'))
TRIALS <- as.integer(get_arg('--trials', '5'))
OUT <- get_arg('--out', '')

set.seed(SEED)
Sys.setenv(OMP_NUM_THREADS = 1)
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)

# assume run from repo root
base <- file.path('experiments', 'chenchao', 'data', DATASET)

# choose train file
tra_pca <- file.path(base, paste0(DATASET, '-pca'), '5-fold', paste0(DATASET, '-pca-5-1tra.dat'))
tra_raw <- file.path(base, paste0(DATASET, '-5-1tra.dat'))
if (file.exists(tra_pca)) {
  data_path <- tra_pca
} else {
  data_path <- tra_raw
}

kmeans_dir <- file.path(base, 'kmeans', paste0('seed', SEED))
theta_path <- file.path(kmeans_dir, 'theta.dat')
rules_path <- file.path(kmeans_dir, 'rules.dat')
num_mfs_path <- file.path(kmeans_dir, 'num_mfs.json')

if (!file.exists(theta_path) || !file.exists(rules_path)) {
  stop('Missing theta.dat or rules.dat in kmeans directory: ', kmeans_dir)
}

# source FuzzyR functions from reference folder
r_dir <- file.path('reference', 'FuzzyR', 'R')
for (f in list.files(r_dir, pattern = '\\.R$', full.names = TRUE)) {
  source(f)
}

# load data (.dat)
load_dat <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  lines <- lines[!grepl('^@|^%', lines)]
  if (length(lines) == 0) stop('No data rows in ', path)
  sep <- ifelse(grepl(',', lines[1]), ',', '\\s+')
  df <- read.table(text = lines, sep = sep, header = FALSE)
  as.matrix(df)
}

mat <- load_dat(data_path)
X <- mat[, 1:(ncol(mat) - 1), drop = FALSE]

# sample rows
n <- min(SAMPLES, nrow(X))
idx <- sample(seq_len(nrow(X)), size = n, replace = FALSE)
X <- X[idx, , drop = FALSE]

# parse num_mfs.json without jsonlite
parse_num_mfs <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = '')
  txt <- gsub('[^0-9,]', '', txt)
  parts <- unlist(strsplit(txt, ','))
  as.integer(parts[parts != ''])
}

num_mfs <- if (file.exists(num_mfs_path)) parse_num_mfs(num_mfs_path) else NULL

# load theta
read_theta <- function(path) {
  lines <- readLines(path, warn = FALSE)
  out <- list()
  for (ln in lines) {
    ln <- trimws(ln)
    if (nchar(ln) == 0) next
    out[[length(out) + 1]] <- as.numeric(strsplit(ln, '\\s+')[[1]])
  }
  out
}

theta_list <- read_theta(theta_path)

# decode trapmf (same as Python)
decode_trapmf <- function(theta, m, j) {
  n <- length(theta)
  if (j == 1) return(c(-Inf, -Inf, theta[1], theta[2]))
  if (j == m) return(c(theta[n - 1], theta[n], Inf, Inf))
  start <- (j - 1) * 4 - 1  # R index
  return(theta[start:(start + 3)])
}

# build FIS
fis <- newfis('fis', fisType='mamdani', mfType='t1', andMethod='prod', orMethod='max', impMethod='min', aggMethod='max', defuzzMethod='centroid')

# add input vars
for (i in seq_len(ncol(X))) {
  rng <- range(X[, i])
  fis <- addvar(fis, 'input', paste0('x', i), rng)
}

# output range from theta output line
theta_out <- theta_list[[length(theta_list)]]
output_range <- c(min(theta_out), max(theta_out))
fis <- addvar(fis, 'output', 'y', output_range)

# add input mfs
for (i in seq_len(ncol(X))) {
  theta <- theta_list[[i]]
  m <- if (!is.null(num_mfs)) num_mfs[i] else ((length(theta) / 4) + 1)
  for (j in seq_len(m)) {
    params <- decode_trapmf(theta, m, j)
    fis <- addmf(fis, 'input', i, paste0('mf', j), 'trapmf', params)
  }
}

# add output mfs
m_out <- if (!is.null(num_mfs)) num_mfs[length(num_mfs)] else ((length(theta_out) / 4) + 1)
for (j in seq_len(m_out)) {
  params <- decode_trapmf(theta_out, m_out, j)
  fis <- addmf(fis, 'output', 1, paste0('mf', j), 'trapmf', params)
}

# load rules (first num_inputs + 1 cols)
rules_raw <- read.table(rules_path, header = FALSE)
num_inputs <- ncol(X)
rules_use <- as.matrix(rules_raw[, 1:(num_inputs + 1)])
weights <- rep(1, nrow(rules_use))
and_or <- rep(1, nrow(rules_use))
rule_list <- cbind(rules_use, weights, and_or)

fis <- addrule(fis, rule_list)

# warmup
if (WARMUP > 0) {
  for (i in seq_len(WARMUP)) {
    evalfis(X, fis, point_n = 201)
  }
}

# timing (multiple trials)
totals <- c()
means <- c()
for (t in seq_len(TRIALS)) {
  start <- proc.time()
  for (i in seq_len(REPEATS)) {
    evalfis(X, fis, point_n = 201)
  }
  elapsed <- (proc.time() - start)['elapsed']
  mean_ms <- (elapsed / REPEATS) * 1000.0
  totals <- c(totals, as.numeric(elapsed))
  means <- c(means, as.numeric(mean_ms))
  cat(sprintf('[trial %d] total_s=%.6f mean_ms=%.6f\\n', t, as.numeric(elapsed), as.numeric(mean_ms)))
}

mean_total <- mean(totals)
mean_ms_mean <- mean(means)
mean_ms_median <- median(means)
mean_ms_std <- if (length(means) > 1) sd(means) else NA

cat('======================================================================\n')
cat('R FIS benchmark\n')
cat('dataset :', DATASET, '\n')
cat('seed    :', SEED, '\n')
cat('samples :', n, '\n')
cat('repeats :', REPEATS, '\n')
cat('trials  :', TRIALS, '\n')
cat('mean_total_s :', mean_total, '\n')
cat('mean_ms_mean :', mean_ms_mean, '\n')
cat('mean_ms_median :', mean_ms_median, '\n')
cat('mean_ms_std :', mean_ms_std, '\n')
cat('======================================================================\n')

# save csv
if (OUT == '') {
  out_dir <- file.path('experiments', 'benchmarks', 'results')
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(out_dir, 'benchmark_fis_r_trials.csv')
} else {
  out_path <- OUT
}

write_header <- !file.exists(out_path)
if (write_header) {
  header <- data.frame(language=character(), dataset=character(), seed=integer(), samples=integer(),
                       repeats=integer(), trial=integer(), total_seconds=double(), mean_ms=double())
  write.table(header, out_path, sep=',', row.names=FALSE, col.names=TRUE, append=FALSE)
} else {
  # keep appending
}

for (t in seq_len(TRIALS)) {
  row <- data.frame(language='r', dataset=DATASET, seed=SEED, samples=n, repeats=REPEATS,
                    trial=t, total_seconds=as.numeric(totals[t]), mean_ms=as.numeric(means[t]))
  write.table(row, out_path, sep=',', row.names=FALSE, col.names=FALSE, append=TRUE)
}

cat('saved:', out_path, '\n')
