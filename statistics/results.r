library(data.table)
library(ggplot2)

ranks <- fread("ranks.csv")

ranks[, subset := factor(subset)]

summary(ranks)

plot(ranks$count)

first_test <- ranks[subset == "test" & rank <= 5]

rank_plot <- ggplot(ranks, aes(x = rank, y = frequency, color = subset)) +
  geom_line() + ylim(0, 1) +
  annotate("point", x = first_test$rank, y = first_test$frequency) +
  labs(title = "Results")

ggsave("figures/ranks.pdf", rank_plot)