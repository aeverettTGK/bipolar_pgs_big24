#!/usr/bin/env Rscript

library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

if (length(args)!=3) {
  stop("Arguments needed: PGS ID file, metadata file, and output file")
}

important <- read.csv(args[1])
metadata <- read.csv(args[2])

colnames(important) <- c("Variable","Coeff")

important <- important %>%
    rename("PGS.ID" = "Variable")

filtered_md <- metadata %>%
    inner_join(important, by = "PGS.ID") %>%
    select(PGS.ID, Coeff, Mapped.Traits)

write.csv(filtered_md, args[3], row.names = FALSE)
