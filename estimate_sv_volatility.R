# -*- coding: utf-8 -*-
# (R 스크립트지만 파일 헤더는 설명용 텍스트)
#
# [SV 모형 2단계] fx_prices.csv -> sv_volatility.json
# ------------------------------------------------------------
# 업로드하신 논문(천도현·김지훈·김병천, 2017)이 사용한 확률적 변동성(SV) 모형과
# 동일한 방법론(Kim, Shephard, Chib 1998)을 R의 stochvol 패키지로 추정한다.
# MCMC(Metropolis-Hastings + Gibbs)를 직접 구현하는 대신, 이미 검증된
# stochvol 패키지의 svsample() 함수가 그 과정을 대신 수행한다.
#
# 사전 준비 (터미널에서 R 실행 후):
#   install.packages("stochvol")
#   install.packages("jsonlite")
#
# 실행:
#   Rscript estimate_sv_volatility.R

library(stochvol)
library(jsonlite)

cat("fx_prices.csv 읽는 중...\n")
prices <- read.csv("fx_prices.csv", stringsAsFactors = FALSE)
prices$date <- as.Date(prices$date)

pairs <- setdiff(colnames(prices), "date")
cat("대상 통화쌍:", paste(pairs, collapse = ", "), "\n")

result <- list()

for (pair in pairs) {
  cat("\n===", pair, "SV 모형 추정 중 (몇 분 걸릴 수 있음) ===\n")

  series <- prices[[pair]]
  dates <- prices$date

  # 결측치(주말/공휴일 등) 제거
  valid <- !is.na(series)
  series <- series[valid]
  dates <- dates[valid]

  # 로그 수익률(%) 계산 - 논문의 식 (3.1)과 동일한 방식
  log_returns <- 100 * diff(log(series))
  return_dates <- dates[-1]

  # 평균 조정 (논문의 "평균 조정된 자산 수익률" 방식)
  y <- log_returns - mean(log_returns)

  # SV 모형 추정 (MCMC). 공모전 시연용으로 draws를 논문의 300,000회보다
  # 훨씬 줄였다 (그래도 수렴에는 충분한 수준). 필요시 늘릴 수 있음.
  fit <- svsample(y, draws = 10000, burnin = 2000, quiet = TRUE)

  # 각 시점의 로그분산(h_t) 추정치 평균 -> 조건부 표준편차(sigma_t)로 변환
  h_t <- colMeans(fit$latent[[1]])
  sigma_t <- exp(h_t / 2)  # 논문 식 (2.1): e^(h_t/2)가 변동성

  pair_result <- data.frame(
    date = format(return_dates, "%Y-%m-%d"),
    sigma = sigma_t
  )

  result[[pair]] <- pair_result
  cat(pair, "완료. 평균 sigma:", round(mean(sigma_t), 4), "\n")
}

# JSON으로 저장: { "USD/KRW": [{date, sigma}, ...], ... }
write_json(result, "sv_volatility.json", auto_unbox = TRUE, pretty = TRUE)
cat("\n저장 완료 -> sv_volatility.json\n")
cat("다음 단계: python compute_impact_weights_sv.py 실행\n")
