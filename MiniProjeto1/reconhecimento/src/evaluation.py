import mnist_loader
import network
import csv
import sys

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

HIDDEN_LAYERS = [1, 2, 3, 4]
HIDDEN_NEURONS = [30, 60, 120, 240]
LEARNING_RATE = [0.001, 0.01, 0.1, 1.0]
EPOCHS = 30
BATCH_SIZE = [4, 8, 16]

for hn in HIDDEN_NEURONS:
    net_input = []
    net_input.append(784)
    net_settings = []

    for hl in HIDDEN_LAYERS:
        net_input.append(hn)
        net_input.append(10)

        net = network.Network(net_input)

        for lr in LEARNING_RATE:
            for bs in BATCH_SIZE:
                acc = net.SGD(training_data, EPOCHS, bs, lr, test_data)

                sett = {'Hidden Layers': hl,
                        'Hidden Neurons': hn,
                        'Learning Rate': lr,
                        'Batch Size': bs,
                        'Accuracy': acc}

                net_settings.append(sett)
                print(sett)
            
        net_input = net_input[:-1]

with open('results.csv', 'wb+') as dictwriter:
    fieldnames = sett.keys()
    writer = csv.DictWriter(dictwriter, fieldnames=fieldnames)

    writer.writeheader()

    for row in net_settings:
        writer.writerow(row)
