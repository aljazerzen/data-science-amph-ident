library(data.table)
library(ggplot2)
library(magrittr)

matches <- fread("matches.csv")

matches[, a_subset := factor(a_subset)]
matches[, b_subset := factor(b_subset)]
matches[, a_label := factor(a_label)]
matches[, b_label := factor(b_label)]
matches[, approach := factor(approach)]

summary(matches)

matches[, is_same := a_label == b_label]

matches[a_subset == b_subset] %>%
  ggplot(aes(x = dist, y = after_stat(density), color = is_same)) +
  geom_density() +
  facet_wrap(vars(a_subset, approach), scales = "free")

ggsave("figures/matches1.pdf")

