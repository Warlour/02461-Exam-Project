import matplotlib.pyplot as plt

# Plotting the training and testing losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Testing Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_testing_loss_plot.png')  # Save the plot as a PNG file
plt.show()