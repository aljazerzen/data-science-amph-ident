library(data.table)
library(ggplot2)

matches <- fread("matches.csv")

matches[, approach := factor(approach)]

ggplot(matches, aes(x = distance, y = after_stat(density), color = is_same)) +
  geom_density()

ggsave("figures/matches.pdf")