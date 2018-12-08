import math

import copy
import gc

class Inter_Node:
    # attr: string
    # parent: inter_node
    # children: dictionary {value:node}
    def __init__(self, attr = None, parent = None, children = None):
        self.isLeaf = False
        self.attr = attr
        self.parent = parent
        self.children = children

class Leaf:
    # class
    def __init__(self, cla):
        self.isLeaf = True
        self.cla = cla

# decision tree
class DT:
    # initial
    def __init__(self, data = None, data_label = None, fun = None, mode = None):
        if(data != None):
            self.train(data, data_label, fun, mode)
        else:
            self.root = None

    # delete tree
    def __delTree(self, node):
        # reach leaf
        if(node.isLeaf):
            del node
            return
        
        # go to next level
        for child in node.children:
            self.__delTree(child)
            del node
            
    # get best attribute to split
    def __getAttr(self, data_index, data, data_label, fun):
        return fun(data_index, data, data_label)

    # split data set
    def __splitData(self, attr, data_index, data, value_dict):
        # get id of attribute
        attrId = data[0].index(attr)

        # get new attribute id list
        new_attr_ids = copy.copy(data_index[0])
        new_attr_ids.pop(data_index[0].index(attrId))

        # get new data index according to current attribute
        new_data_index = {value:[copy.copy(new_attr_ids), []] for value in value_dict[attr]}
        for data_id in data_index[1]:
            value = data[1][data_id][attrId]
            new_data_index[value][1].append(data_id)
        
        return new_data_index

    # get major class
    def __majorCla(self, data_index, data_label):
        # get current class number
        clas = [data_label[row] for row in data_index[1]]
        clas_values = list(set(clas))
        clas_counts = [clas.count(clas_v) for clas_v in clas_values]

        return clas_values[clas_counts.index(max(clas_counts))]

    # check whether should end
    def __checkEnd(self, pre_data_index, data_index, data_label):
        # if no example, get the major class of parent
        if(len(data_index[1]) == 0):
           return self.__majorCla(pre_data_index, data_label)

        # if no attribute exists, find the class with largest number
        if(len(data_index[0]) == 0):
            return self.__majorCla(data_index, data_label)

        # get current class number
        clas = [data_label[row] for row in data_index[1]]
        clas_values = list(set(clas))

        # if pure data
        if(len(clas_values) == 1):
            return clas_values[0]
        
        return None

    # information entropy
    def __ent(self, data_index, data_label):
        # get class
        clas = [data_label[row_id] for row_id in data_index[1]]
        clas_count_p = [clas.count(cla) / len(clas) for cla in list(set(clas))]

        # calculate entropy
        return -sum([count_p + math.log2(count_p) for count_p in clas_count_p])

    # gini
    def __gini(self, data_index, data_label):
        # get class
        clas = [data_label[row_id] for row_id in data_index[1]]
        clas_count_p = [clas.count(cla) / len(clas) for cla in list(set(clas))]

        # calculate gini
        return 1 - sum([count_p * 2 for count_p in clas_count_p])

    # train decision tree
    # data: [[attr(...)], [values(...)]]
    # fun: return attribute
    def train(self, data, data_label, fun, mode = 0):
        # initial decision tree
        if self.root != None:
            self.__delTree(self.root)
            self.root = None
            gc.collect()
        
        # get value list
        value_dict = {attr:list(set([ele[attrId] for ele in data[1]])) for attrId, attr in enumerate(data[0])}

        # train tree
        data_index = [list(range(len(data[0]))), list(range(len(data[1])))]
        self.root = Inter_Node()
        queue = [[data_index, self.root]]

        # build
        while(len(queue) != 0):
            # get current node
            queue_element = queue.pop(0)
            node_temp = queue_element[1]

            # get attribute
            attr = self.__getAttr(queue_element[0], data, data_label, fun)
            node_temp.attr = attr

            # get children
            split_dict = self.__splitData(attr, queue_element[0], data, value_dict)
            node_temp.children = {}

            # check whether children end
            for value in split_dict:
                cla = self.__checkEnd(queue_element[0], split_dict[value], data_label)
                if cla != None:
                    node_temp.children[value] = Leaf(cla)
                else:
                    # get parent
                    node_temp.children[value] = Inter_Node(parent = node_temp)
                    queue += [[split_dict[value], node_temp.children[value]]]

    # test result with current tree
    # data: [[attr(...)], [values(...)]]
    def test(self, data):
        prediction = []
        for ele in data[1]:
            curNode = self.root
            # search to leaf
            while(not curNode.isLeaf):
                attrId = data[0].index(curNode.attr)
                curNode = curNode.children[ele[attrId]]
            
            # get value of leaf
            prediction.append(curNode.cla)

        return prediction

    # information gain
    def infoGain(self, data_index, data, data_label):
        # current entropy
        curEnt = self.__ent(data_index, data_label)
        chosen = [-1, -10000]

        for attrId in data_index[0]:
            # entropy after split
            values = list(set([data[1][row_id][attrId] for row_id in data_index[1]]))
            split_dict = self.__splitData(data[0][attrId], data_index, data, {data[0][attrId]:values})
            gain = curEnt - sum([(len(split_dict[val]) / len(data_index[1])) * self.__ent(split_dict[val], data_label) for val in split_dict])

            # maximum
            if chosen[1] <= gain:
                chosen = [attrId, gain]
        
        return data[0][chosen[0]]

    # Gini index
    def giniIndex(self, data_index, data, data_label):
        chosen =[-1, 10000]

        for attrId in data_index[0]:
            # gini after split
            values = list(set([data[1][row_id][attrId] for row_id in data_index[1]]))
            split_dict = self.__splitData(data[0][attrId], data_index, data, {data[0][attrId]:values})
            gini_index = sum([len(split_dict[val]) / len(data_index[1]) * self.__gini(split_dict[val], data_label) for val in split_dict])

            # minmum
            if chosen[1] >= gini_index:
                chosen = [attrId, gini_index]

        return data[0][chosen[0]]
