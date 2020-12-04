import numpy as np
from brian2 import *

#Generates array of ascending values up to 1000 with every 11th element a 0
def make_targetting_array():
    targetting_count_array=[]
    j=1
    for i in range(1100):
        if i%11!=0:
            targetting_count_array.append(j)
            j+=1
        else:
            targetting_count_array.append(0)
    targetting_count_array=np.array(targetting_count_array)
    return targetting_count_array
  

def randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons):
    check=np.random.normal(scale=1000)
    target=np.round(check/10)
    temp_tar=int(real_i+target)
    if ((i+target)<0) or ((temp_tar)>=len(targetting_count_array)):
        checker=999999999
        return checker,check,12345678
    checker=targetting_count_array[temp_tar]
    if checker>number_of_neurons:
        checker=999999999
        return checker,check,12345678
    return checker,check,temp_tar
    
def randomize_and_edge_checkint(real_i,i,targetting_count_array,n_ca1_neurons):
    check=np.random.normal(scale=100)
    target=np.round(check/10)
    temp_tar=int(real_i+target)
    if ((real_i+target)<0) or ((temp_tar)>=len(targetting_count_array)):
        checker=999999999
        return checker,check,12345678
    checker=targetting_count_array[temp_tar]
    if checker>n_ca1_neurons:
        checker=999999999
        return checker,check,12345678
    return checker,check,temp_tar

#Defines the connectivity schema for excitatory neurons
def excite_to_excite_wire_it_up(number_of_neurons,exkij,exkijsd,inkj,inkijsd,speed):
    connectivity=[]
    connectivityin=[]
    delayex=[]
    delayin=[]
    
    #Makes an array whereby every 11th element is a 0
    targetting_count_array=make_targetting_array()
    
    for i in range(1,number_of_neurons+1):
        in_count=0
        
        #Generates random number from a gaussian distribution of mean and SD provided by input to function
        numb_in_connect=np.int(np.round(np.random.normal(loc=inkj,scale=inkijsd)))
        if exkij!=0:
            numb_connect=np.int(np.round(np.random.normal(loc=exkij,scale=exkijsd)))
        else:
            numb_connect=0
            
        #Finds which index in targetting_count_array corresponds to current value of i
        real_i=np.where(targetting_count_array==i)[0][0]
        
        
        for j in range(1,numb_connect):  
            
            
            checker,check,temp_tar=randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons)
            if checker==999999999:
                continue
            while temp_tar%11==0 or checker==i:
                if in_count<numb_in_connect and checker!=i and temp_tar!=0:
                    connectivityin.append([i,np.int(temp_tar/11)])
                    delayin.append(np.abs(np.round(check*um/speed,decimals=4)))
                    in_count+=1
                checker,check,temp_tar=randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons)
                if checker==999999999:
                    continue
            if checker==999999999:
                continue
            connectivity.append([i,checker])
            delayex.append(np.abs(np.round(check*um/speed,decimals=4)))
        while in_count<numb_in_connect:
            checker,check,temp_tar=randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons)
            if checker==999999999:
                continue
            if temp_tar%11==0 and temp_tar!=0:
                connectivityin.append([i,np.int(temp_tar/11)])
                delayin.append(np.abs(np.round(check*um/speed,decimals=4)))
                in_count+=1
    connectivity=np.array(connectivity)
    connectivityin=np.array(connectivityin)
    return connectivity,connectivityin,delayex,delayin

def ca1in_wire_it_up(number_of_ca1_neurons,exkij,exkijsd,inkj,inkijsd,speed):
    speed2=100*um/ms
    connectivity=[]
    connectivityin=[]
    delayex=[]
    delayin=[]
    targetting_count_array=make_targetting_array()
    for i in range(1,int(number_of_ca1_neurons/10)+1):
        in_count=0
        numb_in_in_connect=np.int(np.round(np.random.normal(loc=inkj,scale=inkijsd)))
        numb_2_ex_connect=np.int(np.round(np.random.normal(loc=exkij,scale=exkijsd)))
        real_i=np.where(targetting_count_array==i)[0][0]
        for j in range(1,numb_2_ex_connect):  
            checker,check,temp_tar=randomize_and_edge_checkint(i*11,i,targetting_count_array,
                number_of_ca1_neurons)
            if checker==999999999:
                continue
            while temp_tar%11==0 or checker==i:
                if in_count<numb_in_in_connect and checker!=i and temp_tar!=0:
                    connectivityin.append([i,np.int(temp_tar/11)])
                    delayin.append(np.abs(np.round(check*um/speed2,decimals=4)))
                    in_count+=1
                checker,check,temp_tar=randomize_and_edge_checkint(i*11,i,
                    targetting_count_array,number_of_ca1_neurons)
                if checker==999999999:
                    continue
            if checker==999999999:
                continue
            connectivity.append([i,checker])
            delayex.append(np.abs(np.round(check*um/speed2,decimals=4)))
        while in_count<numb_in_in_connect:
            checker,check,temp_tar=randomize_and_edge_checkint(i*11,i,targetting_count_array,
                number_of_ca1_neurons)
            if checker==999999999:
                continue
            if temp_tar%11==0 and temp_tar!=0:
                connectivityin.append([i,np.int(temp_tar/11)])
                delayin.append(np.abs(np.round(check*um/speed2,decimals=4)))
                in_count+=1
    connectivity=np.array(connectivity)
    connectivityin=np.array(connectivityin)
    return connectivity,connectivityin,delayex,delayin

#wires up CA3 in2ex
def in2ex_wire_it_up(number_of_ca3_neurons,ca3_inpykij,ca3_inpykij_sd,speed):
    targetting_count_array=make_targetting_array()
    connectivity=[]
    delay=[]
    for i in range(1,101):
        numb_connect=np.int(np.round(np.random.normal(loc=ca3_inpykij,scale=ca3_inpykij_sd)))
        cell_pos=i*11
        for j in range(1,numb_connect):  
            step=np.random.uniform(-1.5*100,1.5*100)
            cell_step=step/10
            target_cell=np.int(np.round(cell_pos+cell_step))

            if target_cell<0 or (target_cell>=len(targetting_count_array)):
                continue
            real_target=targetting_count_array[target_cell]
            while real_target==0:
                step=np.random.uniform(-1.5*100,1.5*100)
                cell_step=step/10
                target_cell=np.int(np.round(cell_pos+cell_step))

                if target_cell<0 or (target_cell>=len(targetting_count_array)):
                    continue
                real_target=targetting_count_array[target_cell] 
            connectivity.append([i,real_target])
            delay.append((np.abs(np.round(step*um/speed,decimals=4))))
    connectivity=np.array(connectivity)
    return connectivity,delay

def randomize_and_edge_checkSC(real_i,i,targetting_count_array,number_of_neurons):
    check=np.random.normal(scale=1200)
    target=np.round(check/10)
    temp_tar=int(real_i+target)
    if ((i+target)<0) or ((temp_tar)>=len(targetting_count_array)):
        checker=999999999
        return checker,check,12345678
    checker=targetting_count_array[temp_tar]
    if checker>number_of_neurons:
        checker=999999999
        return checker,check,12345678
    return checker,check,temp_tar

def schafer_collaterals(number_of_neurons,exkij,exkijsd,inkj,inkijsd,speed):
    connectivity=[]
    connectivityin=[]
    delayex=[]
    delayin=[]
    targetting_count_array=make_targetting_array()
    for i in range(1,number_of_neurons+1):
        if exkij!=0:
            numb_connect=np.int(np.round(np.random.normal(loc=exkij,scale=exkijsd)))
        else:
            numb_connect=0
        real_i=np.where(targetting_count_array==i)[0][0]
        for j in range(1,numb_connect):  
            checker,check,temp_tar=randomize_and_edge_checkSC(real_i,i,targetting_count_array,number_of_neurons)
            if checker==999999999:
                continue
            if temp_tar%11==0 and temp_tar!=0:
                dupe_rate=13
                for k in range(dupe_rate):
                    connectivityin.append([i,np.int(temp_tar/11)])
                    delayin.append(np.abs(np.round(check*um/speed,decimals=4)))
            else:
                dupe_rate=np.abs(np.round(13+np.random.normal(scale=13)))
                for k in range(int(dupe_rate)):
                    connectivity.append([i,checker])
                    delayex.append(np.abs(np.round(check*um/speed,decimals=4)))
    connectivity=np.array(connectivity)
    connectivityin=np.array(connectivityin)
    delayex=delayex+1000*um/speed
    delayin=delayin+1000*um/speed
    return connectivity,connectivityin,delayex,delayin  