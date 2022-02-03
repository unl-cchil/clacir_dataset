library(tidyverse)
library(lubridate)
library(hms)

# Experiment 1
all_data1 <- read_csv("~/active_sync/research/hai_cognitive/data/exp_1/clean_data.csv") %>% 
  arrange(participant) %>% 
  mutate(date = as.Date(ymd_hm(date))) %>% 
  select(participant, date)
tags1 <- read_csv("~/active_sync/research/hai_cognitive/data/exp_1/hr_log/hr_data_times.csv") %>% 
  select(-contains("hr_")) %>% 
  mutate(across(contains("t_"), ~ as_hms(parse_date_time(.x, orders = "HMS")))) %>% 
  left_join(all_data1, by = c("participant")) %>% 
  select(participant, date, condition, everything()) %>% 
  mutate(condition = fct_recode(condition, "Control" = "Con"))

write_csv(tags1, "~/active_sync/projects/hai_stress/data/raw_data/hr_data_times1.csv")

# Experiment 2
cond_data2 <- read_csv("~/active_sync/research/hai_cognitive/data/exp_2/hai_conditions.csv") %>%
  arrange(participant) %>%
  select(participant, condition) %>% 
  mutate(condition = fct_recode(condition, "HAI" = "hai", "Control" = "control"))
tags2 <- read_csv("~/active_sync/research/hai_cognitive/data/exp_2/hr_log/hr_data_times.csv") %>% 
  select(-contains("hr_")) %>% 
  mutate(date = as.Date(t_1),
         across(contains("t_"), ~ as_hms(parse_date_time(.x, orders = "ymd HMS")))) %>% 
  left_join(cond_data2, by = c("participant")) %>% 
  select(participant, date, condition, everything())

write_csv(tags2, "~/active_sync/projects/hai_stress/data/raw_data/hr_data_times2.csv")


