import matplotlib.pyplot as plt
import os
my_path = os.path.dirname(os.path.abspath(__file__))
plt.style.use('ggplot')

# Data
baseline = 26.23
labels = ['Scalar', 'Autovec', 'Manual vectorization']
values = [1, baseline/4.06, baseline/4.07]

# Colors for each bar
colors = ['#fdaa48', '#6890F0', '#A890F0']

# Create the bar graph
bars = plt.bar(labels, values, color=colors)

# Add title and labels
plt.title('AVX512, N=512, 1000 RUNS each')
plt.xlabel('Kernel type')
plt.ylabel('Ratio')

# Set y-axis limit to zoom in
plt.ylim(0, 8)  # Adjust the upper limit to zoom in closer

# Add the value of each bar on top of the bar
for bar in bars:
    # Get the height of each bar (the value)
    yval = bar.get_height()
    # Position the text above the bar
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval:.2f}', 
             ha='center', va='bottom', fontsize=12)

# Save the plot
plt.savefig(my_path+'/bar_graph_avx512.svg', format='svg')

# Show the plot
plt.show()
