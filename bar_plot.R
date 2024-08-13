#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)

data <- read.csv("/Users/aeverettgk/Desktop/RESEARCH/UCLA-BIG/prs_project/main_final/analyses/traits_labeled.csv")

agg_data <- data %>%
    group_by(PGS.ID, Type, Mapped.Traits) %>%
    summarize(AvgCoeff = mean(Coeff), .groups="drop")

excised_data <- agg_data %>%
    arrange(desc(AvgCoeff)) %>%
    slice(c(1:25, (n() - 24):n()))

top_bottom <- agg_data %>%
    arrange(desc(AvgCoeff)) %>%
    slice(c(1:10, (n() - 9):n())) %>%
    mutate(Label = if_else(row_number() <= 10, as.character(11 - row_number()), as.character(-(row_number() - n() + 10))))


top10 <- top_bottom %>% slice(1:10)
bottom10 <- top_bottom %>% slice(11:20)

colors = c("Biological process" = "#FF0000",
           "Body measurement" = "#FFA500",
           "Cancer" = "#DB2D43",
           "Cardiovascular disease" = "#67CA50",
           "Cardiovascular measurement" = "#269DCE",
           "Digestive system disorder" = "#c603fc",
           "Hematological measurement" = "#FFC0CB",
           "Immune system disorder" = "#5D2080",
           "Inflammatory measurement" = "#808080",
           "Lipid measurement" = "#A80054",
           "Liver enzyme measurement" = "#FFFFFF",
           "Metabolic disorder" = "#00FFFF",
           "Neurological disorder" = "#A3DEF8",
           "Neurological measurement" = "#00FF00",
           "Other disease" = "#E76838",
           "Other trait" = "#008080",
           "Other measurement" = "#FBF9AF")

bars <- ggplot(excised_data, aes(x = reorder(PGS.ID, -AvgCoeff), y = AvgCoeff, fill = Type)) +
    geom_bar(stat = "identity", color = "black") +
    geom_text(data = top10, aes(label = Label), vjust = 1.5, size = 3, fontface = "bold") +
    geom_text(data = bottom10, aes(label = Label), vjust = -1.5, size = 3, fontface = "bold") +
    geom_hline(yintercept = 0, color = "black") +
    scale_fill_manual(values = colors, guide = guide_legend(title = "Categories")) +
    labs(x = "Trait", y = "Avg. Coeff") +
    coord_cartesian(ylim = c(min(agg_data$AvgCoeff) - abs(max(agg_data$AvgCoeff) - min(agg_data$AvgCoeff)) * 0.1,
                             max(agg_data$AvgCoeff) + abs(max(agg_data$AvgCoeff) - min(agg_data$AvgCoeff)) * 0.1)) +
    theme(panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white"),
          axis.text.x = element_blank(),
          axis.line.y = element_line(color = "black", size = 0.5),
          legend.background = element_rect(fill = "white"))

ggsave("histogram.png", plot = bars, width = 12, height = 8, dpi = 300)
