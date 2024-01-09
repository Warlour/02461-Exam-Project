import tkinter as tk
from tkinter import messagebox
from mainclass import ModelHandler

class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Model Parameters")
        
        # Create labels and entry fields for each parameter
        self.labels = ["Model", "Batch Size", "Start Learning Rate", "Epochs", "Gamma", "Weight Decay", "Min Learning Rate", "Momentum"]
        self.entries = []
        
        for i, label in enumerate(self.labels):
            tk.Label(self.window, text=label).grid(row=i, column=0)
            entry = tk.Entry(self.window)
            entry.grid(row=i, column=1)
            self.entries.append(entry)
        
        # Create train button
        train_button = tk.Button(self.window, text="Train", command=self.train_model)
        train_button.grid(row=len(self.labels), column=0, columnspan=2)
        
        self.window.mainloop()
    
    def train_model(self):
        # Get parameter values from entry fields
        model = self.entries[0].get()
        batch_size = int(self.entries[1].get())
        start_lr = float(self.entries[2].get())
        epochs = int(self.entries[3].get())
        gamma = float(self.entries[4].get())
        weight_decay = float(self.entries[5].get())
        min_lr = float(self.entries[6].get())
        momentum = float(self.entries[7].get())
        
        # Create an instance of ModelHandler and train the model
        model_handler = ModelHandler(model=model, batch_size=batch_size, start_lr=start_lr, epochs=epochs, gamma=gamma,
                                     weight_decay=weight_decay, min_lr=min_lr, momentum=momentum)
        model_handler.train(stoppage=True)
        
        # Show a message box with the training results
        messagebox.showinfo("Training Results", f"Accuracy: {model_handler.accuracy}")
        
        # Save the trained model and training results
        model_handler.save_model("models", save_lowest=True)
        model_handler.save_excel("training_results")
        
        # Plot the training and validation loss
        model_handler.plot_trainvsvalidationloss(save_plot=True, display_plot=False)

# Create an instance of the GUI class
gui = GUI()
