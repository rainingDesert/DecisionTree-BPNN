import math
import random
import copy
import gc

class Activition:
    # sigmoid
    def sigmoid(self, data, deri = False):
        if not deri:
            return math.exp(data) / (1 + math.exp(data))
        else:
            return math.exp(data) / (1 + math.exp(data)) ** 2

    # tanh
    def tanh(self, data, deri = False):
        if not deri:
            return (math.exp(data) - math.exp(-data)) / (math.exp(data) + math.exp(-data))
        else:
            return 1 - self.tanh(data) ** 2

    # ReLu
    def ReLu(self, data, deri = False):
        if not deri:
            return max(0, data)
        else:
            return max(0, data // abs(data))

class BPNN:
    def __init__(self):
        self.layers = []
        self.weight = []
        self.bias = []
    
    # class for deliver layer
    class DeLayer:
        def __init__(self, cell_num, layer_id):
            self.cell_num = cell_num
            self.layer_id = layer_id
        def passCal(self, data_list):
            return copy.deepcopy(data_list), copy.deepcopy(data_list)
    
    # class for active layer
    class AcLayer:
        def __init__(self, cell_num, fun, layer_id):
            self.cell_num = cell_num
            self.fun = fun
            self.layer_id = layer_id
        
        # feed forward
        def passCal(self, data_list, weights, bias):
            in_value = []
            output = []

            # add up for each cell of next layer
            for in_ws, b in zip(weights, bias):
                in_value.append(sum([ele * w for w, ele in zip(in_ws, data_list)]))
                in_value[-1] += b
                output.append(self.fun(in_value[-1]))

            return in_value, output

        # backpropagation
        def backCal(self, theta_list, in_list, weights):
            back_out = []

            if weights == None:
                back_out = [self.fun(in_list[cell], deri = True) * theta_list[cell] for cell in range(self.cell_num)]
                return back_out

            # next theta
            for cell in range(self.cell_num):
                # calculate sum
                theta_sum = 0
                for theta, ws in zip(theta_list, weights):
                    theta_sum += ws[cell] * theta
                back_out.append(theta_sum * self.fun(in_list[cell], deri = True))

            return back_out

    # class for softmax layer
    class SoftLayer:
        def __init__(self, cell_num, layer_id):
            self.cell_num = cell_num
            self.layer_id = layer_id

        # do softmax
        def softmax(self, data_list, deri = False):
            if not deri:
                exp_list = [math.exp(data) for data in data_list]
                return [exp / sum(exp_list) for exp in exp_list]
            else:
                return [data * (1 - data) for data in self.softmax(data_list)]

        # pass calculation
        def passCal(self, data_list, weights, bias):
            in_value = []
            output = None

            # add up each cell of next layer
            for in_ws, b in zip(weights, bias):
                in_value.append(sum([ele * w for w, ele in zip(in_ws, data_list)]))
                in_value[-1] += b
            
            # softmax
            output = self.softmax(in_value)

            return in_value, output

        # get theta for backpropagation
        def backCal(self, theta_list, in_list, weights):
            back_out = []
            soft_deri = self.softmax(in_list, deri = True)

            if weights == None:
                return [soft_deri[cell] * theta_list[cell] for cell in range(self.cell_num)]

            # for each input cell of weight
            for cell in range(self.cell_num):
                # calculate sum
                theta_sum = 0
                for theta, ws in zip(theta_list, weights):
                    theta_sum += ws[cell] * theta
                back_out.append(theta_sum * soft_deri[cell])

            return back_out

    # randomly shuffle data set
    def ranShuf(self, data, data_label):
        rand_ids = list(range(len(data)))
        random.shuffle(rand_ids)

        new_data = [data[row_id] for row_id in rand_ids]
        new_data_label = [data_label[row_id] for row_id in rand_ids]

        return new_data, new_data_label

    # add deliver layer
    def deliverLayer(self, cell_num):
        if len(self.layers) == 0:
            self.layers.append(BPNN.DeLayer(cell_num, len(self.layers)))

    # add active layer
    def activeLayer(self, cell_num, fun):
        self.layers.append(BPNN.AcLayer(cell_num, fun, len(self.layers)))

    # add softmax layer
    def softmaxLayer(self, cell_num):
        self.layers.append(BPNN.SoftLayer(cell_num, len(self.layers)))

    # forward transfer
    def forward(self, train_data):
        # each layer
        outputs = [copy.deepcopy(train_data)]
        in_values = [copy.deepcopy(train_data)]
        input_data = copy.deepcopy(train_data)
        for layer in self.layers[1:]:

            # calculate output of each cell
            outputs.append([])
            in_values.append([])

            for sample in input_data:
                if(self.weight[layer.layer_id - 1] == None):
                    in_value, output = layer.passCal(sample)
                else:
                    in_value, output = layer.passCal(sample, self.weight[layer.layer_id - 1], self.bias[layer.layer_id - 1])
                in_values[-1].append(in_value)
                outputs[-1].append(output)
            
            # renew input data
            del input_data
            input_data = copy.deepcopy(outputs[-1])

        gc.collect()
        return in_values, outputs

    # back propagate
    def backpropagation(self, train_data_label, in_values, outputs, pace):
        loss = [0 for i in range(len(train_data_label[0]))]

        # initial delta
        delta = []
        thetas = []
        for ws in self.weight:
            if(ws == None):
                delta.append(None)
                thetas.append(None)
            else:
                delta.append([[0 for in_cell in out_cell] for out_cell in ws])
                thetas.append([0 for out_cell in ws])

        # pass loss and get propagate delta
        for label_id in range(len(train_data_label)):
            label_output = train_data_label[label_id]
            sample_output = outputs[-1][label_id]

            # initial theta
            init_theta = [0 for i in range(len(label_output))]
            for output_id in range(len(label_output)):
                init_theta[output_id] = label_output[output_id] - sample_output[output_id]

            loss = [loss[i] + init_theta[i] for i in range(len(label_output))]

            # calculate theta
            new_theta = [self.layers[-1].backCal(init_theta, in_values[self.layers[-1].layer_id][label_id], None)]
            for layer in self.layers[:0:-1][1:]:    # except input layer and first hidden layer
                new_theta.append(layer.backCal(new_theta[-1], in_values[layer.layer_id][label_id], self.weight[layer.layer_id]))
            new_theta.reverse()

            # get delta
            for delta_id, delta_ele in enumerate(delta):
                # for each weight
                if(delta_ele == None):
                    continue
                for out_cell in range(len(delta_ele)):
                    for in_cell in range(len(delta_ele[out_cell])):
                        delta[delta_id][out_cell][in_cell] += new_theta[delta_id][out_cell] * outputs[delta_id][label_id][in_cell]

            # store theta
            for theta_id, theta in enumerate(thetas):
                # for each theta
                if(theta == None):
                    continue
                for out_cell in range(len(theta)):
                    thetas[theta_id][out_cell] += new_theta[theta_id][out_cell]

        # update weights
        for ws_id in range(len(self.weight)):
            if(self.weight[ws_id] == None):
                continue
            # for each weight
            for out_cell in range(len(self.weight[ws_id])):
                for in_cell in range(len(self.weight[ws_id][out_cell])):
                    self.weight[ws_id][out_cell][in_cell] += delta[ws_id][out_cell][in_cell] / len(outputs[0]) * pace

        # update bias
        for bs_id in range(len(self.bias)):
            if(self.bias[bs_id] == None):
                continue
            # for each bias
            for out_cell in range(len(self.bias[bs_id])):
                self.bias[bs_id][out_cell] += thetas[bs_id][out_cell] / len(outputs[0]) * pace

        return loss

    # train network for one round
    def doTrain(self, data_list, data_label, pace, init_w = False):
        # check initilization
        if len(self.layers) == 0:
            print("you should initilize layer")
            return None

        # the first layer must be deliver layer
        if not isinstance(self.layers[0], BPNN.DeLayer):
            self.layers.insert(0, BPNN.DeLayer(len(data_list[0]), -1))
            for layer in self.layers:
                layer.layer_id += 1
        
        # check initial
        if(init_w or len(self.weight) == 0):
            for in_layer, out_layer in zip(self.layers[:-1], self.layers[1:]):
                if not isinstance(out_layer, BPNN.DeLayer):
                    self.weight.append([[2 * random.random() - 1 for in_cell in range(in_layer.cell_num)] for out_cell in range(out_layer.cell_num)])
                    self.bias.append([2 * random.random() - 1 for out_cell in range(out_layer.cell_num)])
                else:
                    self.weight.append(None)
                    self.bias.append(None)

        # forward
        in_values, outputs = self.forward(data_list)

        # back propogation
        loss = self.backpropagation(data_label, in_values, outputs, pace)

        del outputs
        del in_values
        return loss

    # simple train network
    # data: [[value](...)]
    # data_label: [label(...)]
    def train(self, data, data_label, epoches, batches, pace, init_w = False):
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
                data, data_label = self.ranShuf(data, data_label)
                cur_pos = batches - (len(data) - cur_pos)
                train_data += copy.copy(data[: cur_pos])
                train_data_label += copy.copy(data_label[: cur_pos])

            loss = self.doTrain(train_data, train_data_label, pace)

            if(epoch % 500 == 0):
                print("path: %d, loss: %s" %(epoch, str([abs(ls/batches) for ls in loss])))

            del train_data
            del train_data_label
            gc.collect()
    
    # predict result
    def predict(self, test_data):
        in_values, outputs = self.forward(test_data)
        results = []
        for output in outputs[-1]:
            results.append([0 for ele in output])
            results[-1][output.index(max(output))] = 1
        return results