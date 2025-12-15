import re
import matplotlib.pyplot as plt
import numpy as np


# Function to extract losses from each line
def extract_losses_from_file(log_file):
    iterations = []
    loss_ce = []
    loss_fm = []
    loss_S = []
    loss_D = []
    loss_st = []

    # Regular expression pattern to extract the required values from each line
    pattern = r"iter=(\d+), loss_ce=([0-9.]+), loss_fm=([0-9.]+), loss_S=([0-9.]+), loss_D=([0-9.]+), loss_st=([0-9.]+)"

    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(pattern, line.strip())
            if match:
                # Extracting values and appending to lists
                iterations.append(int(match.group(1)))
                loss_ce.append(float(match.group(2)))
                loss_fm.append(float(match.group(3)))
                loss_S.append(float(match.group(4)))
                loss_D.append(float(match.group(5)))
                loss_st.append(float(match.group(6)))

    return iterations, loss_ce, loss_fm, loss_S, loss_D, loss_st


# Replace 'log_file_path.txt' with the path to your actual log file
log_file_path = 'log_file_path.txt'
iterations, loss_ce, loss_fm, loss_S, loss_D, loss_st = extract_losses_from_file(log_file_path)

# Convert iterations to epochs
batch_size = 8
total_samples = 11436
steps_per_epoch = total_samples // batch_size
epochs = np.array(iterations) / steps_per_epoch

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_ce, label="loss_ce", marker='o')
plt.plot(epochs, loss_fm, label="loss_fm", marker='o')
plt.plot(epochs, loss_S, label="loss_S", marker='o')
plt.plot(epochs, loss_D, label="loss_D", marker='o')
plt.plot(epochs, loss_st, label="loss_st", marker='o')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses over Epochs")
plt.legend()
plt.grid(True)
plt.show()
