import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output = pd.read_csv('trainloop.csv', sep=',', header=None)
epoch = output[1]
loss = output[6]
acc = output[8]

fig, axs = plt.subplots(2)
fig.suptitle('Loss and accuracy per epoch')
axs[0].plot(epoch, loss, label='loss')
axs[0].set_ylabel('loss')
axs[0].grid()
axs[1].plot(epoch, acc, label='accuracy')
axs[1].set_ylabel('accuracy')
axs[1].set_xlabel('epoch')
axs[1].grid()
plt.show()
