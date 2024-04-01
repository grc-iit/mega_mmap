# Load the ggplot2 package
library(ggplot2)

# Sample data
categories <- c('Category A', 'Category B', 'Category C')
values <- c(20, 35, 30)
errors <- c(2, 3, 2)  # Example error values

# Create a data frame
df <- data.frame(categories = categories, values = values, errors = errors)

# Create the bar graph with error bars
bar_plot <- ggplot(df, aes(x = categories, y = values)) +
  geom_bar(stat = "identity", fill = "blue") +
  geom_errorbar(aes(ymin = values - errors, ymax = values + errors), width = 0.2, color = "red") +
  labs(y = "Values", title = "Bar Graph with Error Bars") +
  theme_minimal()

# Print the bar graph
print(bar_plot)