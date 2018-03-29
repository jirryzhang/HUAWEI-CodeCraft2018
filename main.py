 # -*- coding: utf-8 -*-
import re
import math
import copy
import numpy as np
from matplotlib import pyplot as plt

from tf_lstm import LSTM
from keras_lstm import mLSTM

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
standardlization = 1

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
    now_block = 0
    flavor_num = 0
    flavor_list = []
    f = open(input_path, 'r+', encoding='utf-8')
    for line in f:
        if line is not '\n':
            if now_block == 0:
                space_1 = line.find(' ')
                space_2 = line.find(' ', space_1+1)
                CPU = int(line[0:space_1])
                MEM = int(line[space_1:space_2])
                STO = int(line[space_2:])
                sample_ps = PhysicalServer(CPU, MEM, STO)
                sample_ps.state()
                now_block += 1
            else:
                if now_block == 1:
                    flavor_num = int(line)
                    for i in range(flavor_num):
                        line = f.readline()
                        space_1 = line.find(' ')
                        space_2 = line.find(' ', space_1+1)
                        space_3 = line.find('\n', space_2+1)
                        NUM = int(line[6:space_1])
                        CPU = int(line[space_1:space_2])
                        MEM = int(line[space_2:space_3])
                        tempVM = VirtualMachine(NUM, CPU, MEM)
                        sample_vm.append(tempVM)
                        flavor_list.append(NUM)
                        tempVM.state()
                    now_block += 1
                else:
                    if now_block == 2:
                        dim_to_be_optimized = line.replace('\n', '')
                        print('The dimension to be optimized is: ' + dim_to_be_optimized)
                        now_block += 1
                    else:
                        if now_block == 3:
                            predict_begin = line.replace('\n', '')
                            predict_end = f.readline().replace('\n', '')
                            print('Predict time begin at: ' + predict_begin)
                            print('Predict time end at: ' + predict_end)
                            print('\n')
            
    
    # Read the beginning time
    line = open(train_path, encoding='utf-8').readline()
    space_1 = line.find('\t')
    space_2 = line.find('\t', space_1+1)
    history_begin = line[space_2+1:].replace('\n', '')
    
    history_data = [[.0]for i in range(total_flavor)]
    for i in range(total_flavor):
        for j in range(time2val(history_begin), time2val(predict_begin) - 1):
            history_data[i].append(0)
            
    future_data = [[.0]for i in range(total_flavor)]
    for i in range(total_flavor):
        for j in range(time2val(predict_begin), time2val(predict_end) - 1):
            future_data[i].append(0)
            
    # Read history data
    for line in open(train_path, encoding='utf-8'):
        space_1 = line.find('\t')
        space_2 = line.find('\t', space_1+1)
        temp_flavor = int(line[space_1+7:space_2])
        temp_time = line[space_2+1:].replace('\n', '')
        if temp_time is not None:
            value = time2val(temp_time)
            if temp_flavor <= total_flavor:
                history_data[temp_flavor-1][value] += 1 * standardlization
            else:
                pass
#                print('Flavor data error.\n')
#                print('Now flavor: ' + str(temp_flavor))
        else:
            print('Time data error.\n')
            
                
    # Print history data
    print('History data: ')
    print('Total diffs: ' + str(len(history_data[0])))
    for i in range(total_flavor):
        print('Flavor' + str(i+1) + ': (Total: ' + str(sum(history_data[i])) + ')\n' + str(history_data[i]) + '\n')
        
    # Read test data
    for line in open(test_path, encoding='utf-8'):
        space_1 = line.find('\t')
        space_2 = line.find('\t', space_1+1)
        temp_flavor = int(line[space_1+7:space_2])
        temp_time = line[space_2+1:].replace('\n', '')
        if temp_time is not None:
            value = time2val(temp_time) - time2val(predict_begin) - 1
            if temp_flavor <= total_flavor:
                future_data[temp_flavor-1][value] += 1 * standardlization
            else:
                pass
#                print('Flavor data error.\n')
#                print('Now flavor: ' + str(temp_flavor))
        else:
            print('Time data error.\n')
            
                
    # Print history data
    print('Future data: ')
    print('Total diffs: ' + str(len(future_data[0])))
    for i in range(total_flavor):
        print('Flavor' + str(i+1) + ': (Total: ' + str(sum(future_data[i])) + ')\n' + str(future_data[i]) + '\n')
#    plt.plot(history_data[2])
        
    return history_data, future_data

# =============================================================================
# 数据加和
# =============================================================================
def data_addup(dataset, n=7):
    dataset_copy = copy.deepcopy(dataset)
    for i in range(total_flavor):
        for j in range(n-1, len(dataset[i])):
            dataset[i][j] = sum(dataset_copy[i][j-n+1:j+1])
    return dataset
            
# =============================================================================
# 差分数据
# =============================================================================
def data_difference(dataset, interval=1):
    diff = [[]for i in range(total_flavor)]
    for i in range(total_flavor):
        for j in range(interval, len(dataset[i])):
            value = dataset[i][j] - dataset[i][j-interval]
            diff[i].append(value)
    return diff

# =============================================================================
# Sigmoid变换
# =============================================================================
def sigmoid(value):
    
    return (1.0 / (1 + math.exp(-value)))

def list_sigmoid(dataset):
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

def list_asigmoid(dataset):
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

    history_data, future_data = readData()
    data = [[]for i in range(total_flavor)]
    for i in range(total_flavor):
        for j in range(len(history_data[i])):
            data[i].append(history_data[i][j])
        for j in range(len(future_data[i])):
            data[i].append(future_data[i][j])

    data = data_addup(data)
    # data = data_difference(data)

    lstm = mLSTM(data[1])
    lstm.lstm_model()
