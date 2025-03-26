import numpy as np
import matplotlib.pyplot as plt

#lista apo xromata me 4 katigories 
colors = ['red', 'blue', 'green', 'orange']
labels = ['C1', 'C2', 'C3', 'C4']

# 1) Plot gia train_data.txt
#Diabazoume olo to arxio
train_data = np.loadtxt('train_data.txt', skiprows=1)
# Stiles x1, x2, class ]
x1_train = train_data[:, 0]
x2_train = train_data[:, 1]
c_train = train_data[:, 2].astype(int)  #akereo gia maska

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for cls in range(4):
    #maska gia cls
    mask = (c_train == cls)
    # s=megeyhso ton koukidon tou scatter
    plt.scatter(
        x1_train[mask],
        x2_train[mask],
        color=colors[cls],
        label=f'{labels[cls]} (train)',
        alpha=0.5,
        s=10  # pio mikres koukides 
    )

plt.title('Train data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

#Plot gia test_data.txt
test_data = np.loadtxt('test_data.txt', skiprows=1)
#Stiles=[ x1, x2, class ]
x1_test = test_data[:, 0]
x2_test = test_data[:, 1]
c_test = test_data[:, 2].astype(int)

plt.subplot(1, 2, 2)
for cls in range(4):
    mask = (c_test == cls)
    plt.scatter(
        x1_test[mask],
        x2_test[mask],
        color=colors[cls],
        label=f'{labels[cls]} (test)',
        alpha=0.5,
        s=10  # gai tis mikres koukides
    )

plt.title('Test data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

#  plot gia classified_test_points.txt
classified_data = np.loadtxt('classified_test_points.txt')
x1_c = classified_data[:, 0]
x2_c = classified_data[:, 1]
true_c = classified_data[:, 2].astype(int)
pred_c = classified_data[:, 3].astype(int)
correct = classified_data[:, 4].astype(int)

plt.figure(figsize=(12, 5))

# Aristera me basi tin katigoria
plt.subplot(1, 2, 1)
for cls in range(4):
    mask = (true_c == cls)
    plt.scatter(
        x1_c[mask],
        x2_c[mask],
        color=colors[cls],
        label=f'True: {labels[cls]}',
        alpha=0.5,
        s=10  # mikroteres koukides
    )
plt.title('Test data (True classes)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

#Deksia gia marker gia s/k proble4is
plt.subplot(1, 2, 2)
for cls in range(4):
    #sostes proble4is
    mask_correct = (pred_c == cls) & (correct == 1)
    #lathos proble4is
    mask_incorrect = (pred_c == cls) & (correct == 0)
    
    plt.scatter(
        x1_c[mask_correct],
        x2_c[mask_correct],
        color=colors[cls],
        marker='o',
        alpha=0.7,
        label=f'Pred: {labels[cls]} (correct)',
        s=10  # mikroteres koukides
    )
    plt.scatter(
        x1_c[mask_incorrect],
        x2_c[mask_incorrect],
        color=colors[cls],
        marker='x',
        alpha=0.7,
        label=f'Pred: {labels[cls]} (wrong)',
        s=30  #gia na ksexorizoun ta x
    )

plt.title('Test data (Predicted classes)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

# 4) Plot gia to error (loss) apo to loss_per_epoch.txt
loss_data = np.loadtxt('loss_per_epoch.txt')
epochs = loss_data[:, 0]
loss = loss_data[:, 1]

plt.figure(figsize=(8, 4))
plt.plot(epochs, loss, 'b-o', markersize=5, alpha=0.7, label='Training Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)
plt.show()