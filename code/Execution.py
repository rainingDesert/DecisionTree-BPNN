import DecisionTree as tree
import NeuralNetwork as NN

import random
import gc
import csv
import copy

class Run:
    # initial function
    def __init__(self):
        self.data = None
        self.data_label = None

    # randomly split
    def randomSplit(self, ratio, data, data_label):
        rows_id = list(range(len(data_label)))
        random.shuffle(rows_id)

        train_data = [data[0]]
        test_data = [data[0]]

        train_data.append([data[1][row_id] for row_id in rows_id[:int(len(rows_id) * ratio)]])
        test_data.append([data[1][row_id] for row_id in rows_id[int(len(rows_id) * ratio):]])

        train_data_label = [data_label[row_id] for row_id in rows_id[:int(len(rows_id) * ratio)]]
        test_data_label = [data_label[row_id] for row_id in rows_id[int(len(rows_id) * ratio):]]

        return train_data, test_data, train_data_label, test_data_label

    # load data from file
    def loadFile(self, file_name, head):
        # initial data
        if self.data != None:
            del self.data
            del self.data_label
            gc.collect()
        self.data = [[], []]
        self.data_label = []

        # load data
        with open(file_name, "r") as f:
            context = csv.reader(f, delimiter = ",")
            if(head == None):
                self.data[0] = next(context)[:-1]
            else:
                self.data[0] = head
            for row in context:
                if(len(row) == 0):
                    continue
                self.data[1].append(row[:-1])
                self.data_label.append(row[-1])

    # run with decision tree
    # function: 0 -> information gain, 1 -> gini index
    def simpleRunDT(self, train_ratio = 0.75, function = 1):
        # split train data and test data
        train_data, test_data, train_data_label, test_data_label = self.randomSplit(train_ratio, self.data, self.data_label)

        # train
        dt = tree.DT()
        if function == 0:
            dt.train(data = train_data, data_label = train_data_label, fun = dt.infoGain)
        else:
            dt.train(data = train_data, data_label = train_data_label, fun = dt.giniIndex)

        # test
        predictions = dt.test(test_data)
        accuracy = sum([1 for label, pred in zip(test_data_label, predictions) if label == pred]) / len(predictions)

        print("accuracy is: %f" %(accuracy))
        return accuracy

    # run with Neural Network
    # function: 0 -> sigmoid, 1 -> tanh, 2 -> ReLu
    def simpleRunNN(self, epoches, batch, pace, train_ratio = 0.75):
        # translate label into number
        data = [copy.copy(self.data[0])]
        data.append([[float(ele) for ele in sample] for sample in self.data[1]])

        # normalize data
        right = []
        left = []
        for attr_id in range(len(data[1][0])):
            attr_list = [row[attr_id] for row in data[1]]
            right.append(max(attr_list))
            left.append(min(attr_list))
        
        for r_id in range(len(data[1])):
            for c_id in range(len(data[1][r_id])):
                data[1][r_id][c_id] = (data[1][r_id][c_id] - left[c_id]) / (right[c_id] - left[c_id])

        clas_list = list(set(self.data_label))
        new_label = []
        for label in self.data_label:
            new_label.append([0 for i in range(len(clas_list))])
            new_label[-1][clas_list.index(label)] = 1

        # split train data and test data
        train_data, test_data, train_data_label, test_data_label = self.randomSplit(train_ratio, data, new_label)

        # train
        nn = NN.BPNN()
        act = NN.Activition()
        
        # one input, one hidden, one softmax
        nn.deliverLayer(len(train_data[0]))
        nn.activeLayer(8, act.sigmoid)
        nn.activeLayer(len(clas_list), act.sigmoid)
        nn.softmaxLayer(len(clas_list))
        self.train(train_data[1], train_data_label, test_data, test_data_label, clas_list, epoches, batch, pace, nn)

        # test
        predictions = nn.predict(test_data[1])
        accuracy = sum([1 for label, pred in zip(test_data_label, predictions) if clas_list[label.index(1)] == clas_list[pred.index(1)]]) / len(predictions)
        # for label, pred in zip(test_data_label, predictions):
        #     print(str(clas_list[label.index(1)]) + " " + str(clas_list[pred.index(1)]))

        print("gengeral accuracy is: %f" %(accuracy))
        return accuracy

    def train(self, data, data_label, test_data, test_data_label, clas_list, epoches, batches, pace, nn):
        # start train
        cur_pos = 0
        data = copy.copy(data)
        data_label = copy.copy(data_label)
        for epoch in range(epoches):
            # get current batch
            if cur_pos + batches < len(data):
                train_data = copy.copy(data[cur_pos : cur_pos + batches])
                train_data_label = copy.copy(data_label[cur_pos : cur_pos + batches])
                cur_pos += batches
            else:
                train_data = copy.copy(data[cur_pos :])
                train_data_label = copy.copy(data_label[cur_pos :])

                # next round
                data, data_label = nn.ranShuf(data, data_label)
                cur_pos = batches - (len(data) - cur_pos)
                train_data += copy.copy(data[: cur_pos])
                train_data_label += copy.copy(data_label[: cur_pos])

            loss = nn.doTrain(train_data, train_data_label, pace)

            if(epoch % 100 == 0):
                # test
                predictions = nn.predict(test_data[1])
                accuracy = sum([1 for label, pred in zip(test_data_label, predictions) if clas_list[label.index(1)] == clas_list[pred.index(1)]]) / len(predictions)
                print("epoch: %d, loss: %s, accuracy: %f" %(epoch, sum([abs(ls/batches) for ls in loss]), accuracy))


            del train_data
            del train_data_label
            gc.collect()

if __name__ == "__main__":
    # file_name = "./data/iris.data.discrete.txt"
    file_name = "./data/iris.data.txt"
    head = list(range(4))
    epoches = 10000
    batch = 4
    pace = 0.1

    r = Run()
    r.loadFile(file_name, head)
    # r.simpleRunDT()
    r.simpleRunNN(epoches = epoches, batch = batch, pace = pace)
