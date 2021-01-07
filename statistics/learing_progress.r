
library(data.table)
library(ggplot2)

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
  ylim(-2, 5) + xlim(1000, max(loss$time))

# ----------

rank_freqs <- fread("running_rank_freq.csv")
rank_freqs[, subset := factor(subset)]
rank_freqs[, time := 1:.N]

rank_resets <- shift(rank_freqs$epoch, fill = -1) > rank_freqs$epoch
rank_resets <- rank_freqs[rank_resets]
rank_resets[, label := ""]

rank_freqs <- melt(rank_freqs,
  id.vars = c("epoch", "total_cases", "subset", "time"),
  variable.name = "rank",
  value.name = "count"
)
rank_freqs[, frequency := count / total_cases]

rank_resets[3]$label <- "dropout = 0.5"
rank_resets[5]$label <- "dropout = 0.05"
rank_resets[7]$label <- "increase aug."
rank_resets[8]$label <- "increase aug."
rank_resets[9]$label <- "decrease aug."
rank_resets[11]$label <- "increase scale aug. (0.2)"
rank_resets[12]$label <- "increase translation aug. (0.2)"
rank_resets[13]$label <- "decrease aug.\ndropout = 0.1"
rank_resets[14]$label <- "dropout = 0.5"
rank_resets[15]$label <- "dropout = 0"

rank_plot <- ggplot(rank_freqs,
  aes(x = time, y = frequency, color = subset, linetype = rank)
) +
  geom_line() + ylim(0, 1) + 
  geom_vline(xintercept = rank_resets$time, alpha = 0.2) +
  annotate("text", x = rank_resets$time, y = 0.1,
    label = rank_resets$label, angle = 45)

rank_plot
ggsave("figures/learning_progress_ranks.pdf", rank_plot, width = 21)