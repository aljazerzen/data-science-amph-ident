library(data.table)
library(ggplot2)

ranks <- fread("ranks-ident-2.csv")

ranks[, subset := factor(subset)]
ranks[, approach := factor(approach)]

summary(ranks)

highlight <- ranks[subset == "ident" & rank <= 5]

ggplot(ranks, aes(x = rank, y = frequency, color = approach, linetype = subset)) +
  geom_line() + ylim(0, 1) +
  annotate("point", x = highlight$rank, y = highlight$frequency) +
  labs(title = "Results")

ggsave("figures/ranks.pdf")