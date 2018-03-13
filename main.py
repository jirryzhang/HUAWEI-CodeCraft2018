# -*- coding: utf-8 -*-
import re
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

H = 12
N = 8

# =============================================================================
# physical server class definition
# =============================================================================
class physical_server:
    
    def __init__(self, cpu, mem, sto):
        self.cpu = cpu
        self.rest_cpu = cpu
        self.mem = mem
        self.rest_mem = mem
        self.sto = sto
        self.rest_sto = sto
        self.vm = []
    
    def add_vm(self, vm):
        self.vm.append(vm)
        self.rest_cpu -= vm.cpu
        self.rest_mem -= vm.mem
        self.rest_sto -= vm.sto
        
    def rm_vm(self, num):
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
        
   
# =============================================================================
# virtual machine class definition
# =============================================================================
class virtual_machine:
    
    def __init__(self, cpu, mem, sto):
        self.cpu = cpu
        self.mem = mem
        self.sto = sto
    
    def state(self):
        print('    CPU: ' + str(self.cpu) + '\n' +
              '    Memory: ' + str(self.mem) + '\n' +
              '    Storage: ' + str(self.sto))
        
        
# =============================================================================
# Convert time into value
# =============================================================================
def time2value(time):
    
    #yyyy = time[0:4]
    mm = time[5:7]
    dd = time[8:10]
    hh = time[11:13]
    #print(yyyy)
    #print(mm)
    #print(dd)
    #print(hh)
    
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
# Read data from txt and get the time vector
# =============================================================================
def read_data(path):
    
    f = open(path, encoding='utf-8')
    
    mList = [[0]for i in range(15)];
    
    for line in f:
        
        time = re.search('(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)', line).group()
        flavor = int(re.search('\d', re.search('flavor\d*', line).group()).group())
        
        if time is not None:
            value = time2value(time)
            for i in range(len(mList[flavor]), value+1):
                for j in range(15):
                    mList[j].append(0)
            mList[flavor][value] += 1
                
#    print(mList)
#    plt.plot(mList[1])
#    for i in range(15):
#        if len(mList[i]) > 1:
#            plt.plot(mList[1])
#            print(mList[i])
    
    return mList[1]


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    
    ps = physical_server(56, 128, 1200)
    vm = virtual_machine(28, 64, 600)
    ps.add_vm(vm)
    ps.rm_vm(0)
    #ps.state()
    time_array = read_data('./put_together.txt')
    x = []
    y = []
    for i in range(N, len(time_array)):
        x.append(time_array[i-N : i-1])
        y.append(time_array[i])
    
    total = len(x)
    part = int(total * 7 / 10)

# =============================================================================
#     LSE拟合
# =============================================================================
    clf = linear_model.LinearRegression()
    
# =============================================================================
#     Neural network拟合
# =============================================================================
#    clf = MLPRegressor(solver='sgd',
#                       alpha=1e-5,
#                       hidden_layer_sizes=(50, 50),
#                       random_state=1)
    
# =============================================================================
#     预测
# =============================================================================
    clf.fit(x[:part], y[:part])
    predictX = clf.predict(x[part:])
    
    for i in range(len(predictX)):
        predictX[i] = round(predictX[i])
    testY = y[part:]
    
    print(sum(testY))
    print(sum(predictX))
    plt.plot(predictX)
    plt.plot(testY)
#    print(clf.coef_)
    
        
    
        