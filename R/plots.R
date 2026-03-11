library(tidyverse)


plot_model <- function(data, filename) {
  path <- paste0("results/Int ", data, " -1.0/")
  knc <- read_csv(paste0(path, "KNeig_training.csv"), show_col_types = FALSE) |>
    mutate(model = "KNC", .before = 1)
  rfc <- read_csv(paste0(path, "Rando_training.csv"), show_col_types = FALSE) |>
    mutate(model = "RFC", .before = 1)
  lda <- read_csv(paste0(path, "Linea_training.csv"), show_col_types = FALSE) |>
    mutate(model = "LDA", .before = 1)
  dtc <- read_csv(paste0(path, "Decis_training.csv"), show_col_types = FALSE) |>
    mutate(model = "DTC", .before = 1)
  all_models <- bind_rows(dtc, knc, lda, rfc)

  all_models |>
    group_by(model) |>
    summarise(
      mean_ba = mean(BAccuracy) * 100,
      sd_ba = sd(BAccuracy) * 100,
      ci_ba = papaja::ci(BAccuracy) * 100
    )

  all_models_long <- all_models |>
    select(
      model,
      "Balanced Accuracy" = BAccuracy,
      Kappa,
      AUPRC,
      "TPR@10%FPR" = `TPR@1%FPR`
    ) |>
    pivot_longer(
      cols = c(`Balanced Accuracy`, Kappa, AUPRC, `TPR@10%FPR`),
      names_to = "measure",
      values_to = "value"
    ) |>
    mutate(
      measure = fct_relevel(
        measure,
        "Balanced Accuracy",
        "Kappa",
        "AUPRC",
        "TPR@10%FPR"
      ),
      model = fct_relevel(model, "DTC", "KNC", "RFC", "LDA")
    )

  all_models_long |>
    ggplot(aes(x = model, y = value)) +
    stat_summary(fun.data = "mean_sdl") +
    facet_wrap(vars(measure), ncol = 1) +
    coord_flip() +
    scale_y_continuous(
      labels = scales::percent,
      limits = c(0, 1),
      oob = scales::oob_keep
    ) +
    labs(x = "", y = "") +
    theme_bw() +
    theme(text = element_text(size = 15),
  axis.title = element_blank())
  ggsave(paste0("figures/", filename, ".png"), height = 7, width = 6)
}

plot_model("All Data", "models_all")
plot_model("Task ACC", "models_acc")
plot_model("Task BVP", "models_bvp")
plot_model("Task EDA", "models_eda")
plot_model("Task HRV", "models_hrv")
plot_model("Task Remove ACC", "models_noacc")
