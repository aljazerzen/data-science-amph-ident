library(data.table)
library(ggplot2)
library(magrittr)

setwd("..")

ranks1 <- fread("net1/ranks.csv")
ranks1[, net := "net1"]

ranks2 <- fread("net2/ranks.csv")
ranks2[, net := "net2"]

ranks3 <- fread("net3/ranks.csv")
ranks3[, net := "net3"]

ranks <- rbind(ranks1, ranks2, ranks3)
ranks[, subset := factor(subset)]
ranks[, net := factor(net)]
summary(ranks)

ranks[
  subset == "test" & (rank <= 3 | rank == 5 | rank == 10), .(net, rank, frequency)
] %>% dcast(rank ~ net)

highlight <- ranks[subset == "test" & rank <= 5 & net == "net1"]

ggplot(ranks, aes(x = rank, y = frequency, color = net, linetype = subset)) +
  geom_line() + ylim(0, 1) + ylab("Accuracy") + xlab("Rank") + xlim(1, 15) +
  annotate("point", x = highlight$rank, y = highlight$frequency)

ggsave("../report/figures/ranks-classification.pdf", width = 3.5, height = 2.3)
