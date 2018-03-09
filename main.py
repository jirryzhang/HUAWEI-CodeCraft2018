# -*- coding: utf-8 -*-

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
              'Rest storage: ' + str(self.rest_sto))
        print('Total virtual machine: ' + str(len(self.vm)) + '\n' +
              'List: ')
        for i in range(len(self.vm)):
            print('  VM ' + str(i) + ': ')
            self.vm[i].state()
        
    
class virtual_machine:
    
    def __init__(self, cpu, mem, sto):
        self.cpu = cpu
        self.mem = mem
        self.sto = sto
    
    def state(self):
        print('    CPU: ' + str(self.cpu) + '\n' +
              '    Memory: ' + str(self.mem) + '\n' +
              '    Storage: ' + str(self.sto))

if __name__ == '__main__':
    
    ps = physical_server(56, 128, 1200)
    vm = virtual_machine(28, 64, 600)
    ps.add_vm(vm)
    #ps.rm_vm(0)
    ps.state()
        
        