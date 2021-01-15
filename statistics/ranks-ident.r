library(data.table)
library(ggplot2)
library(magrittr)

setwd("..")

ranks2 <- fread("net2/ranks-ident.csv")
ranks2[, net := "net2"]

ranks3 <- fread("net3/ranks-ident.csv")
ranks3[, net := "net3"]

ranks4 <- fread("net4/ranks-ident.csv")
ranks4[, net := "net4"]

ranks <- rbind(ranks2, ranks3, ranks4)

ranks[, subset := factor(subset)]
ranks[, approach := factor(approach)]
ranks[, net := factor(net)]

ranks[, approach := sub("after-", "", approach)]
ranks[, approach := sub("fc1-", "", approach)]

ranks[, subset := ordered(subset, levels = c("train", "test", "ident"))]

summary(ranks)

net2_subsets <- ranks[net == "net2" & approach == "retrained-msr"]
highlight <- net2_subsets[subset == "ident" & rank <= 5]
net2_subsets %>%
  ggplot(aes(x = rank, y = frequency, color = subset)) +
  geom_line(size = 0.8) + ylim(0, 1) + xlim(1, 100) +
  ylab("Accuarcy") + xlab("Rank") +
  scale_color_brewer(palette = "Dark2") +
  annotate("point", x = highlight$rank, y = highlight$frequency)
ggsave("../report/figures/ranks-subsets.pdf", width = 4, height = 3)

ranks[subset == "ident" & approach == "retrained-msr"][
  (rank <= 5 | rank == 10 | rank == 100)]

ranks[subset == "ident" & grepl("msr", approach)] %>%
  ggplot(aes(x = rank, y = frequency, color = net, linetype = approach)) +
  geom_line(size = 0.8, alpha = 0.8) + ylim(0, 1) + xlim(1, 100) +
  ylab("Accuarcy") + xlab("Rank") +
  theme(legend.position = "right")

ggsave("../report/figures/ranks-ident-layer-selection.pdf", width = 4, height = 2.3)

# ---------


ranks[subset == "ident"]["retrained-msr" == approach][
  (rank <= 5 | rank == 10 | rank == 100), ] %>%
  dcast(rank ~ net, value.var = c("frequency"))

ranks[subset == "ident"][grepl("retrained", approach)] %>%
  ggplot(aes(x = rank, y = frequency, color = net, linetype = approach)) +
  geom_line(size = 0.8, alpha = 0.8) + ylim(0, 1) + xlim(1, 100) +
  ylab("Accuarcy") + xlab("Rank") +
  theme(legend.position = "right")

ggsave("../report/figures/ranks-ident-nets.pdf", width = 4, height = 2.3)

