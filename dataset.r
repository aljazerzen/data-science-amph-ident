library(data.table)
library(ggplot2)

names <- fread("Amphibian dataset/names.csv")
files <- fread("Amphibian dataset/files.csv")

ggplot(names, aes(x = count)) + geom_histogram(bins = 10, color = "black")

mean(names$count)

names[startsWith(name, "s")][, .(new_classes = .N, new_images = sum(count))]

names[, .(freq = nrow(.SD)), by = .(count)]

fwrite(
  files[name %in% names[count == 1]$name, .(filepath)],
  "images-single.txt"
)

count_all <- sum(names$count)

two_images <- names[count == 2]
ident <- two_images[sample(seq_len(nrow(two_images)), count_all * 0.2 / 2)]

ident

fwrite(
  files[name %in% ident$name, .(filepath)],
  "images-ident.txt"
)
