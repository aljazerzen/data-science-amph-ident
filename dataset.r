library(data.table)
library(ggplot2)

names <- fread("Amphibian dataset/names.csv")

ggplot(names, aes(x = count)) + geom_histogram(bins = 10, color = "black")

mean(names$count)

names[, .(freq = nrow(.SD)), by = .(count)]
