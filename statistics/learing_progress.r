
library(data.table)
library(ggplot2)
library(magrittr)

setwd("../net4")

loss <- fread("loss.csv")
loss[, time := 1:.N]
loss_resets <- shift(loss$epoch, fill = -1) > loss$epoch

ggplot(loss, aes(x = time, y = loss_mean)) +
  geom_line() +
  geom_ribbon(
    aes(ymin = loss_mean - loss_std, ymax = loss_mean + loss_std),
    alpha = 0.5
  ) +
  geom_hline(yintercept = 0) +
  geom_vline(xintercept = which(loss_resets), alpha = 0.2) +
  ylim(-2, 5) + xlim(0, max(loss$time))

ggsave("loss.pdf", width = 15, height = 6)

# ----------

rank_freqs <- fread("running_ranks_freq.csv")
rank_freqs[, subset := factor(subset)]
rank_freqs[, time := 1:.N]

rank_resets <- shift(rank_freqs$epoch, fill = -1) > rank_freqs$epoch
rank_resets <- rank_freqs[rank_resets]
rank_resets[, label := ""]
annotations <- fread("annotations.csv")
for (i in seq_len(nrow(annotations))) {
  annotation <- annotations[i]
  rank_resets[annotation$index]$label <- annotation$text
}

rank_freqs <- melt(rank_freqs,
  id.vars = c("epoch", "total_cases", "subset", "time"),
  variable.name = "rank",
  value.name = "count"
)
rank_freqs[, frequency := count / total_cases]

rank_freqs %>%
  ggplot(aes(x = time * 5, y = frequency, color = subset, linetype = rank)) +
  geom_line() + ylim(0, 1) + ylab("accuracy") + xlab("epoch") + # xlim(0, 930) +
  geom_vline(xintercept = rank_resets$time * 5, alpha = 0.2) +
  annotate("text", x = rank_resets$time * 5, y = 0.1,
    label = rank_resets$label, angle = 20)

ggsave("learning-progress-ranks.pdf", width = 15, height = 6)
