import matplotlib.pyplot as plt

# Example precision and recall values for three layers
precision = [[0.8, 0.7, 0.6], [0.7, 0.6, 0.5], [0.9, 0.8, 0.7]]
recall = [[0.4, 0.5, 0.6], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8]]

# Plot each layer's precision-recall curve
for i in range(len(precision)):
    plt.plot(recall[i], precision[i], label=f'Layer {i+1}')

# Add labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Show the legend and plot
plt.legend()
plt.show()