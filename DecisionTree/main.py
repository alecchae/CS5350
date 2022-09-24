import math
import pandas as pd

def ID3(S, Attributes, labels, split):
    if (S["label"] == S["label"][0]).all():
        leafnode = Node(True, None)
        leafnode.attribute(S["label"][0])
        return leafnode

    else:
        S_v = S.copy()
        Rootnode = Node(False, None)
        A = Bestsplit(S, Attributes, split, labels)
        for v in Attributes[A]:
            node = Node(False, Rootnode)
            Rootnode.branches(v, node)
            S_v = S_v.loc[S_v[A] == v]
            if S_v.empty:
                leafnode = Node(True, Rootnode)
                leafnode.label(get_commonvalue(S_v))
                Rootnode.branches[v] = leafnode
            else:
                _Attributes = Attributes.copy()
                del _Attributes[A]
                Rootnode.branches[v] = ID3(S_v, _Attributes, labels)

        return Rootnode


def get_commonvalue(S):
    # gets the most frequent label in the table
    return S["label"].mode()[0]


def giniindex(S, Attributes, labels):
    Total = len(S.index)
    unacc_count = len(S[S["label"] == "unacc"])
    acc_count = len(S[S["label"] == "acc"])
    good_count = len(S[S["label"] == "good"])
    vgood_count = len(S[S["label"] == "vgood"])
    Total_gain = 1 - (math.pow(unacc_count / Total, 2) + math.pow(acc_count / Total)
                      + math.pow(good_count / Total) + math.pow(vgood_count / Total))
    max_gain = -1
    max_attribute = list(Attributes.keys())[0]
    for A, v in Attributes.items():
        temp_totalgain = 0
        for _v in v:
            temp_gain = 1
            for label in labels:
                temp_gain = temp_gain - math.pow(len(S[(S[A] == _v) & (S["label"] == label)]) / len(S[S[A] == _v]), 2)
            temp_gain = temp_gain * len(S[S[A] == _v]) / Total
            temp_totalgain = temp_totalgain + temp_gain
        if max_gain < Total_gain - temp_totalgain:
            max_gain = Total_gain - temp_totalgain
            max_attribute = A

    return max_attribute


def majorityerror(S, Attributes, labels):
    Total = len(S.index)
    unacc_count = len(S[S["label"] == "unacc"])
    acc_count = len(S[S["label"] == "acc"])
    good_count = len(S[S["label"] == "good"])
    vgood_count = len(S[S["label"] == "vgood"])
    Total_gain = min(unacc_count, acc_count, good_count, vgood_count) / Total

    max_gain = -1
    max_attribute = list(Attributes.keys())[0]
    for A, v in Attributes.items():
        temp_totalgain = 0
        for _v in v:
            temp_gain = 1
            for label in labels:
                temp_totalgain = temp_totalgain + (len(S[(S[A] == _v)]) / Total) \
                                 * (getmin(S, A, _v, labels) / len(S[(S[A] == _v)]))
        if max_gain < Total_gain - temp_totalgain:
            max_gain = Total_gain - temp_totalgain
            max_attribute = A

    return max_attribute


def getmin(S, A, _v, labels):
    min = len(S[(S[A] == _v) & (S["label"] == labels[0])])
    for label in labels:
        tempmin = len(S[(S[A] == _v) & (S["label"] == label)])
        if min > tempmin:
            min = tempmin

    return min


def informationgain(S, Attributes, labels):
    Total = len(S.index)
    unacc_count = len(S[S["label"] == "unacc"])
    acc_count = len(S[S["label"] == "acc"])
    good_count = len(S[S["label"] == "good"])
    vgood_count = len(S[S["label"] == "vgood"])
    Total_gain = -(unacc_count / Total) * math.log2(unacc_count / Total) \
                 - (acc_count / Total) * math.log2(acc_count / Total) \
                 - (good_count / Total) * math.log2(good_count / Total) \
                 - (vgood_count / Total) * math.log2(vgood_count / Total)
    max_gain = -1
    max_attribute = list(Attributes.keys())[0]
    for A, v in Attributes.items():
        temp_totalgain = 0
        for _v in v:
            temp_gain = 0
            for label in labels:
                if len(S[(S[A] == _v) & (S["label"] == label)]) / len(S[(S[A] == _v)]) != 0:
                    temp_gain = temp_gain - (len(S[(S[A] == _v) & (S["label"] == label)]) / len(S[(S[A] == _v)]) *
                                             math.log2(len(S[(S[A] == _v) & (S["label"] == label)]) / len(S[(S[A] == _v)])))

            temp_gain = temp_gain * len(S[S[A] == _v]) / Total
            temp_totalgain = temp_totalgain + temp_gain
        if max_gain < Total_gain - temp_totalgain:
            max_gain = Total_gain - temp_totalgain
            max_attribute = A
    return max_attribute


def Bestsplit(S, Attributes, split, labels):
    if split == "majority error":
        return majorityerror(S, Attributes, labels)
    if split == "gini index":
        return giniindex(S, Attributes, labels)
    # default information gain
    else:
        return informationgain(S, Attributes, labels)


class Node:
    def __init__(self, is_leaf, parent):
        self.label = None
        self.attribute = None
        self.is_leaf = is_leaf
        self.branches = {}
        self.parent = parent

    def attribute(self, attribute):
        self.attribute = attribute

    def branches(self, label, node):
        self.branches[label] = node

    def label(self, label):
        self.label = label


class Tree:
    def __init__(self, split='majority error', max_depth=10):
        self.root = None
        self.split = split
        self.max_depth = max_depth


if __name__ == '__main__':
    # data = sys.argv[1]
    # if data == "car":
    data = pd.read_csv("car/train.csv", header=None)
    data.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
    attributes = {"buying": ["vhigh", "high", "med", "low"],
                  "maint": ["vhigh", "high", "med", "low"],
                  "doors": ["2", "3", "4", "5more"],
                  "persons": ["2", "4", "more"],
                  "lug_boot": ["small", "med", "big"],
                  "safety": ["low", "med", "high"]}
    labels = ["unacc", "acc", "good", "vgood"]

    splits = ["information gain", "majority error", "gini index"]

    print(ID3(data, attributes, labels, splits[0]))
