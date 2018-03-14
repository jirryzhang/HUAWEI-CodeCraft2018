# -*- coding: utf-8 -*-
import re
import math
import copy
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

# =============================================================================
# Number of days of every month
# =============================================================================
Month = dict()
Month[1] = 31
Month[2] = 30
Month[3] = 31
Month[4] = 30
Month[5] = 31
Month[6] = 30
Month[7] = 31
Month[8] = 31
Month[9] = 30
Month[10] = 31
Month[11] = 30
Month[12] = 31

INPUT = './input_5flavors_cpu_7days.txt.'
TEST = './TestData_2015.2.20_2015.2.27.txt'
TRAIN = 'TrainData_2015.1.1_2015.2.19.txt'

H = 24
N = 7
TOTAL_FLAVOR = 15

global SamplePS
global SampleVM
global DimToBeOptimized
global HistoryTime_Begin
global PredictTime_Begin
global PredictTime_End
global FlavorNum

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
        value += Month[i+1] * 24
    value += (dd-1) * 24 + hh
    
    return int(value / H)
        

# =============================================================================
# Read data from given txt
# =============================================================================
def readData():
    
    global SamplePS
    global SampleVM
    global DimToBeOptimized
    global HistoryTime_Begin
    global PredictTime_Begin
    global PredictTime_End
    global FlavorNum
    
    # Read input file
    nowBlock = 0
    FlavorNum = 0
    flavorList = []
    f = open(INPUT, 'r+', encoding='utf-8')
    for line in f:
        if line is not '\n':
            if nowBlock == 0:
                Space_1 = line.find(' ')
                Space_2 = line.find(' ', Space_1+1)
                CPU = int(line[0:Space_1])
                MEM = int(line[Space_1:Space_2])
                STO = int(line[Space_2:])
                SamplePS = PhysicalServer(CPU, MEM, STO)
                SamplePS.state()
                nowBlock += 1
            else:
                if nowBlock == 1:
                    FlavorNum = int(line)
                    for i in range(FlavorNum):
                        line = f.readline()
                        Space_1 = line.find(' ')
                        Space_2 = line.find(' ', Space_1+1)
                        Space_3 = line.find('\n', Space_2+1)
                        NUM = int(line[6:Space_1])
                        CPU = int(line[Space_1:Space_2])
                        MEM = int(line[Space_2:Space_3])
                        tempVM = VirtualMachine(NUM, CPU, MEM)
                        SampleVM.append(tempVM)
                        flavorList.append(NUM)
                        tempVM.state()
                    nowBlock += 1
                else:
                    if nowBlock == 2:
                        DimToBeOptimized = line.replace('\n', '')
                        print('The dimension to be optimized is: ' + DimToBeOptimized)
                        nowBlock += 1
                    else:
                        if nowBlock == 3:
                            PredictTime_Begin = line.replace('\n', '')
                            PredictTime_End = f.readline().replace('\n', '')
                            print('Predict time begin at: ' + PredictTime_Begin)
                            print('Predict time end at: ' + PredictTime_End)
                            print('\n')
            
    
    # Read the beginning time
    line = open(TRAIN, encoding='utf-8').readline()
    Space_1 = line.find('\t')
    Space_2 = line.find('\t', Space_1+1)
    HistoryTime_Begin = line[Space_2+1:].replace('\n', '')
    
    historyData = [[0]for i in range(TOTAL_FLAVOR)]
    for i in range(TOTAL_FLAVOR):
        for j in range(time2val(HistoryTime_Begin), time2val(PredictTime_Begin)):
            historyData[i].append(0)
            
    testData = [[0]for i in range(TOTAL_FLAVOR)]
    for i in range(TOTAL_FLAVOR):
        for j in range(time2val(PredictTime_Begin), time2val(PredictTime_End)+1):
            testData[i].append(0)
            
    # Read history data
    for line in open(TRAIN, encoding='utf-8'):
        Space_1 = line.find('\t')
        Space_2 = line.find('\t', Space_1+1)
        tempFlavor = int(line[Space_1+7:Space_2])
        tempTime = line[Space_2+1:].replace('\n', '')
        if tempTime is not None:
            value = time2val(tempTime)
            if tempFlavor <= TOTAL_FLAVOR:
                historyData[tempFlavor-1][value] += 1
            else:
                None
#                print('Flavor data error.\n')
#                print('Now flavor: ' + str(tempFlavor))
        else:
            print('Time data error.\n')
            
                
    # Print history data
    print('History data: ')
    print('Total diffs: ' + str(value+1))
    for i in range(TOTAL_FLAVOR):
        print('Flavor' + str(i+1) + ': (Total: ' + str(sum(historyData[i])) + ')\n' + str(historyData[i]))
        
    # Read test data
    for line in open(TEST, encoding='utf-8'):
        Space_1 = line.find('\t')
        Space_2 = line.find('\t', Space_1+1)
        tempFlavor = int(line[Space_1+7:Space_2])
        tempTime = line[Space_2+1:].replace('\n', '')
        if tempTime is not None:
            value = time2val(tempTime) - time2val(PredictTime_Begin)
            if tempFlavor <= TOTAL_FLAVOR:
                testData[tempFlavor-1][value] += 1
            else:
                None
#                print('Flavor data error.\n')
#                print('Now flavor: ' + str(tempFlavor))
        else:
            print('Time data error.\n')
            
                
    # Print history data
    print('Test data: ')
    print('Total diffs: ' + str(value+1))
    for i in range(15):
        print('Flavor' + str(i+1) + ': (Total: ' + str(sum(testData[i])) + ')\n' + str(testData[i]))
    print('\n')
#    plt.plot(historyData[2])
    return historyData, testData


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    
    trainData, testData = readData()
    mixedData = copy.deepcopy(trainData)
    for i in range(TOTAL_FLAVOR):
        for j in range(len(testData[i])):
            mixedData[i].append(testData[i][j])
    
    finalData = copy.deepcopy(mixedData)
    time_split = time2val(PredictTime_Begin)
    
    x = [[]for i in range(TOTAL_FLAVOR)]
    y = [[]for i in range(TOTAL_FLAVOR)]
    
    for i in range(TOTAL_FLAVOR):
        for j in range(N, len(mixedData[i])-1):
            finalData[i][j+1] = sum(mixedData[i][j-N:j-1])
            x[i].append(finalData[i][j-N:j])
            y[i].append(finalData[i][j+1])
            
    print('Final Data:')
    for i in range(TOTAL_FLAVOR):
        print('Flavor' + str(i+1) + ':\n' + str(finalData[i]))
        
    print('X:')
    print(x[0])

# =============================================================================
#     LSE拟合
# =============================================================================
    clf = []
    for i in range(TOTAL_FLAVOR):
        clf.append(linear_model.LinearRegression())
        clf[i].fit(x[i][:time_split], y[i][:time_split])
    
# =============================================================================
#     Neural network拟合
# =============================================================================
    clf = []
    for i in range(TOTAL_FLAVOR):
        clf.append(MLPRegressor(solver='sgd',
                           alpha=1e-5,
                           hidden_layer_sizes=(200, 200),
                           random_state=0))
        clf[i].fit(x[i][:time_split], y[i][:time_split])
    
# =============================================================================
#     预测
# =============================================================================
    
    y_predict = []
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    
    for i in range(FlavorNum):
   
        y_predict.append(clf[i].predict(x[i][time_split+1:]))
    
        if y_predict[i][0] < 0:
            y_predict[i][0] = 0
        else:
            y_predict[i][0] = round(y_predict[i][0])
        
        print('Flavor' + str(i+1) + ':')
        print('Prediction: ' + str(y_predict[i][0]) + '\nActual: ' + str(y[i][-1]) + '\n')
        
        sum_1 += math.pow((y_predict[i][0] - y[i][-1]), 2)
        sum_2 += math.pow((y_predict[i][0]), 2)
        sum_3 += math.pow(y[i][-1], 2)
    
    score_1 = (1 - math.sqrt(sum_1 / FlavorNum) / (math.sqrt(sum_2 / FlavorNum) + math.sqrt(sum_3 / FlavorNum)))
    print(score_1)
