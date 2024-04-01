# install.packages("tidyverse")
# install.packages("gridExtra")

# Load required libraries
library(ggplot2)
library(gridExtra)

# Read the CSV
kmeans <- read.csv("csv/weak_scaling_r/kmeans.csv")
rf <- read.csv("csv/weak_scaling_r/rf.csv")
dbscan <- read.csv("csv/weak_scaling_r/dbscan.csv")
gray_scott <- read.csv("csv/weak_scaling_r/gray_scott.csv")

df = kmeans
df <- df[order(df$nprocs), ]

# Entering dat
df$nprocs <- paste(" ", df$nprocs, " ", sep = "")
nprocs <- df[["nprocs"]]
runtime <- df[["runtime_mean"]]
memory <- df[["mem_mean"]]
runtime_std <- df[["runtime_std"]]
memory_std <- df[["mem_std"]]
impl <- df[["impl"]]

pdf(file="/home/llogan/Documents/Projects/mega_mmap/analysis/output.pdf",
    width=8, height=4)

# Plotting Charts and adding a secondary axis
ggp <- ggplot(df)  +
  aes(fill=impl, x=nprocs, y=runtime) +
  geom_bar(stat="identity", position="dodge")+
  # geom_line(aes(x=nprocs, y=memory),stat="identity",color="red",size=2)+
  labs(title= "Runtime ",
       x="# Procs",y="Runtime (s)") +
  # scale_x_continuous(breaks=nprocs, trans = "log10") #+
  # scale_y_continuous(sec.axis=sec_axis(~.*1,name="Memory (%)"))
ggp

