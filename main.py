# -*- coding: utf-8 -*-
import re
import math
import copy
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

from lstm import LSTM

# =============================================================================
# Number of days of every month
# =============================================================================
month = dict()
month[1] = 31
month[2] = 30
month[3] = 31
month[4] = 30
month[5] = 31
month[6] = 30
month[7] = 31
month[8] = 31
month[9] = 30
month[10] = 31
month[11] = 30
month[12] = 31

input_path = './input_5flavors_cpu_7days.txt.'
test_path = './TestData_2015.2.20_2015.2.27.txt'
train_path = 'TrainData_2015.1.1_2015.2.19.txt'

total_flavor = 15

global sample_ps
sample_vm = list()
global dim_to_be_optimized
global history_begin
global predict_begin
global predict_end
global flavor_num

# =============================================================================
# physical server class definition
# =============================================================================
class PhysicalServer:
    
    def __init__(self, cpu, mem, sto):
        self.cpu = cpu
        self.rest_cpu = cpu
        self.mem = mem
        self.rest_mem = mem
        self.sto = sto
        self.rest_sto = sto
        self.vm = []
    
    def addVm(self, vm):
        self.vm.append(vm)
        self.rest_cpu -= vm.cpu
        self.rest_mem -= vm.mem
        self.rest_sto -= vm.sto
        
    def rmVm(self, num):
        self.rest_cpu += self.vm[num].cpu
        self.rest_mem += self.vm[num].mem
        self.rest_sto += self.vm[num].sto
        del self.vm[num]
        
    def state(self):
        print('Total CPU: ' + str(self.cpu) + '\n' +
              'Used CPU: ' + str(self.cpu - self.rest_cpu) + '\n' +
              'Rest CPU: ' + str(self.rest_cpu) +'\n')
        print('Total memory: ' + str(self.mem) + '\n' +
              'Used memory: ' + str(self.mem - self.rest_mem) + '\n' +
              'Rest memory: ' + str(self.rest_mem) + '\n')
        print('Total storage: ' + str(self.sto) + '\n' +
              'Used storage: ' + str(self.sto - self.rest_sto) + '\n' +
              'Rest storage: ' + str(self.rest_sto) + '\n')
        print('Total virtual machine: ' + str(len(self.vm)) + '\n' +
              'List: ')
        for i in range(len(self.vm)):
            print('  VM ' + str(i) + ': ')
            self.vm[i].state()
        print('\n')
        
   
# =============================================================================
# virtual machine class definition
# =============================================================================
class VirtualMachine:
    
    def __init__(self, num, cpu, mem):
        self.num = num
        self.cpu = cpu
        self.mem = mem
    
    def state(self):
        print('Flavor' + str(self.num) + ': \n'
              '    CPU: ' + str(self.cpu) + '\n' +
              '    Memory: ' + str(self.mem) + '\n')
        
        
# =============================================================================
# Convert time into value
# =============================================================================
def time2val(time):
    
    #yyyy = time[0:4]
    mm = time[5:7]
    dd = time[8:10]
    hh = time[11:13]
    
    # Convertion
    #yyyy *= 365 * 24
    mm = int(mm)
    dd = int(dd)
    hh = int(hh)
    
    # To value
    value = 0
    mm -= 1
    for i in range(0, mm):
        value += month[i+1] * 24
    value += (dd-1) * 24 + hh
    
    return int(value / 24)
        

# =============================================================================
# Read data from given txt
# =============================================================================
def readData():
    
    global sample_ps
    global sample_vm
    global dim_to_be_optimized
    global history_begin
    global predict_begin
    global predict_end
    global flavor_num
    
    # Read input file
    nowBlock = 0
    flavor_num = 0
    flavorList = []
    f = open(input_path, 'r+', encoding='utf-8')
    for line in f:
        if line is not '\n':
            if nowBlock == 0:
                Space_1 = line.find(' ')
                Space_2 = line.find(' ', Space_1+1)
                CPU = int(line[0:Space_1])
                MEM = int(line[Space_1:Space_2])
                STO = int(line[Space_2:])
                sample_ps = PhysicalServer(CPU, MEM, STO)
                sample_ps.state()
                nowBlock += 1
            else:
                if nowBlock == 1:
                    flavor_num = int(line)
                    for i in range(flavor_num):
                        line = f.readline()
                        Space_1 = line.find(' ')
                        Space_2 = line.find(' ', Space_1+1)
                        Space_3 = line.find('\n', Space_2+1)
                        NUM = int(line[6:Space_1])
                        CPU = int(line[Space_1:Space_2])
                        MEM = int(line[Space_2:Space_3])
                        tempVM = VirtualMachine(NUM, CPU, MEM)
                        sample_vm.append(tempVM)
                        flavorList.append(NUM)
                        tempVM.state()
                    nowBlock += 1
                else:
                    if nowBlock == 2:
                        dim_to_be_optimized = line.replace('\n', '')
                        print('The dimension to be optimized is: ' + dim_to_be_optimized)
                        nowBlock += 1
                    else:
                        if nowBlock == 3:
                            predict_begin = line.replace('\n', '')
                            predict_end = f.readline().replace('\n', '')
                            print('Predict time begin at: ' + predict_begin)
                            print('Predict time end at: ' + predict_end)
                            print('\n')
            
    
    # Read the beginning time
    line = open(train_path, encoding='utf-8').readline()
    Space_1 = line.find('\t')
    Space_2 = line.find('\t', Space_1+1)
    history_begin = line[Space_2+1:].replace('\n', '')
    
    historyData = [[0]for i in range(total_flavor)]
    for i in range(total_flavor):
        for j in range(time2val(history_begin), time2val(predict_begin) - 1):
            historyData[i].append(0)
            
    futureData = [[0]for i in range(total_flavor)]
    for i in range(total_flavor):
        for j in range(time2val(predict_begin), time2val(predict_end) - 1):
            futureData[i].append(0)
            
    # Read history data
    for line in open(train_path, encoding='utf-8'):
        Space_1 = line.find('\t')
        Space_2 = line.find('\t', Space_1+1)
        tempFlavor = int(line[Space_1+7:Space_2])
        tempTime = line[Space_2+1:].replace('\n', '')
        if tempTime is not None:
            value = time2val(tempTime)
            if tempFlavor <= total_flavor:
                historyData[tempFlavor-1][value] += 1
            else:
                None
#                print('Flavor data error.\n')
#                print('Now flavor: ' + str(tempFlavor))
        else:
            print('Time data error.\n')
            
                
    # Print history data
    print('History data: ')
    print('Total diffs: ' + str(len(historyData[0])))
    for i in range(total_flavor):
        print('Flavor' + str(i+1) + ': (Total: ' + str(sum(historyData[i])) + ')\n' + str(historyData[i]) + '\n')
        
    # Read test data
    for line in open(test_path, encoding='utf-8'):
        Space_1 = line.find('\t')
        Space_2 = line.find('\t', Space_1+1)
        tempFlavor = int(line[Space_1+7:Space_2])
        tempTime = line[Space_2+1:].replace('\n', '')
        if tempTime is not None:
            value = time2val(tempTime) - time2val(predict_begin) - 1
            if tempFlavor <= total_flavor:
                futureData[tempFlavor-1][value] += 1
            else:
                None
#                print('Flavor data error.\n')
#                print('Now flavor: ' + str(tempFlavor))
        else:
            print('Time data error.\n')
            
                
    # Print history data
    print('Future data: ')
    print('Total diffs: ' + str(len(futureData[0])))
    for i in range(total_flavor):
        print('Flavor' + str(i+1) + ': (Total: ' + str(sum(futureData[i])) + ')\n' + str(futureData[i]) + '\n')
#    plt.plot(historyData[2])
    return historyData, futureData

# =============================================================================
# 数据加和
# =============================================================================
def dataAddUp(dataset, n=7):
    # 以七天为单位加和
    dataset_copy = copy.deepcopy(dataset)
    for i in range(total_flavor):
        for j in range(n-1, len(dataset[i])):
            historyData[i][j] = sum(dataset_copy[i][j-n+1:j])
    return dataset
            
# =============================================================================
# 差分数据
# =============================================================================
def dataDifference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)
    return diff

# =============================================================================
# Sigmoid变换
# =============================================================================
def sigmoid(value):
    
    return (1.0 / (1 + math.exp(-value)))

def listSigmoid(dataset):
    sig = list()
    for i in range(len(dataset)):
        if dataset[i] is not 0:
            value = sigmoid(dataset[i])
            sig.append(value)
        else:
            sig.append(0)
    return sig
 
# =============================================================================
# Sigmoid反变换
# =============================================================================
def asigmoid(value):
    return -math.log((1.0 / value) - 1)

def listAsigmoid(dataset):
    sig = list()
    for i in range(len(dataset)):
        if dataset[i] is not 0:
            value = asigmoid(dataset[i])
            sig.append(value)
        else:
            sig.append(0)
    return sig

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':

    historyData, futureData = readData()
    data = [[]for i in range(total_flavor)]
    for i in range(total_flavor):
        for j in range(len(historyData[i])):
            data[i].append(historyData[i][j])
        for j in range(len(futureData[i])):
            data[i].append(futureData[i][j])
            
    lstm = LSTM(data)
    lstm.data_preprocess(50, 6, 0, 50)