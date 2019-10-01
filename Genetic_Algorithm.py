

#!/usr/bin/env python
# coding: utf8#!/usr/bin/env python


import random as rand 
import NNGeneration as nn
import csv
import os 
import errno
import io 




def GA(X_train, Y_train, X_test, Y_test, num_gen, layers, neurons, LR, activations, optimizers, batch_sizes, ModPerGen, mutation_rate, fittest_rate, Crossover_rate, island, Global_Population):

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" + str(island)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    params = [layers, neurons, LR, activations, optimizers, batch_sizes]
    #Population_final = []
    for i in range(1, num_gen):
        
        path = "./Models_new_out_FLV/Island" + str(island) + "/Generation " 
        
        
        if i == 1: 
            make_dir(path + str(i))
            Population, Global_Population = initial_population(1, X_train, Y_train, X_test, Y_test, ModPerGen, layers, neurons, LR, activations, optimizers, batch_sizes, path, list(Global_Population))
        
        
        
        Population.sort(key=lambda x: x[7])  # sort with respect to MSE
        with io.open(path + str(i) + "/Report" + str(i) + ".csv", 'wb') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(Population) 
            
           
        
        # take the fittest models from previous Population
        fittest_cutoff = int(fittest_rate*ModPerGen)
        fittest_models =  int(fittest_cutoff*2/3)
        #fittest_cutoff + rand.random(fittest_cutoff,ModPerGen)
        Population_tmp = list(Population[:fittest_models])
        
        #Population_final.extend(Population[:fittest_cutoff])
        #print(Population_tmp)
        # take other 10 % randomly and mutate them
        
        make_dir(path + str(i+1))
        
        # mutation of the other fittest Models 
        print("------------ Mutation of fittest now -------------------")
        for k in range(fittest_cutoff - fittest_models):
            rand_idx =rand.randint(0,fittest_cutoff-1) #fittest_models + rand.randint(1,fittest_cutoff-fittest_models-1)
            rand_Model = list(Population[rand_idx])
            mutated, Global_Population = mutate(rand_Model, i+1, X_train, Y_train, 
                                         X_test, Y_test, params,list(Population_tmp), path, list(Global_Population))
            Population_tmp.append(mutated)

            
        # mutation of others
        print("------------ Mutation of others now -------------------")
        mutation_cutoff = int(mutation_rate*ModPerGen)
        for k in range(mutation_cutoff):
            rand_idx = fittest_cutoff + rand.randint(1,ModPerGen-fittest_cutoff-1)
            rand_Model = list(Population[rand_idx])
            mutated, Global_Population = mutate(rand_Model, i+1, X_train, Y_train, 
                                         X_test, Y_test, params, list(Population_tmp), path, list(Global_Population))
            Population_tmp.append(mutated)
            #opulation_tmp = Population_final
            #print(Population_tmp[len(Population_tmp)-1])
        #print(Population_tmp)
        # Crossover with the fittest models
        #Pop_Crossover = []
        print("------------ Crossover now -------------------")
        for j in range(ModPerGen-fittest_cutoff-mutation_cutoff):
            
            Best_Models = list(Population[:fittest_cutoff])
            crossed_model, Global_Population = crossover(list(Best_Models), i+1, X_train, Y_train, 
                                         X_test, Y_test, params, list(Population_tmp), path, list(Global_Population))
            Population_tmp.append(crossed_model)
            #Population_tmp = Population_final
            
            #print(Population_tmp[len(Population_tmp)-1])
            #print(Population_tmp)
        print("------------ Crossover End ------------------")
        #Population_tmp.extend(Pop_Crossover)
        #print(Population_tmp)
        Population = Population_tmp
        
        
        
    Population.sort(key=lambda x: x[7])  # sort with respect to MSE
    with io.open(path + str(num_gen) + "/Report"+ str(num_gen) +".csv", 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(Population) 
        
    return Population, Global_Population
    
    
    
def initial_population(Gen, X_train, Y_train, X_test, Y_test, ModPerGen, layers, neurons, LR, activations, optimizers, batch_sizes, pt, Global_Population):
    Population = []
    Global_pop = Global_Population
    idx = [0,3,4,5]
    params = [layers, neurons, LR, activations, optimizers, batch_sizes]
    for i in range(100):
        print("Initial Model " + str(i))
        num_layers = layers[rand.randint(0, len(layers)-1)]
        num_neurons = neurons[rand.randint(0, len(neurons)-1)]
        lr = LR[rand.randint(0, len(LR)-1)]
        activation = activations[rand.randint(0, len(activations)-1)]
        optimizer = optimizers[rand.randint(0, len(optimizers)-1)]
        batch_size = batch_sizes[rand.randint(0, len(batch_sizes)-1)]
        tmp = [num_layers, num_neurons, lr, activation, optimizer, batch_size]
        
        if list(tmp[i] for i in idx) in (Global_pop[i][:] for i in range(len(Global_pop))):
            Pop, Global_pop = mutate(tmp, Gen, X_train, Y_train, 
                                         X_test, Y_test, params, list(Global_pop), pt, list(Global_pop))
            Population.append(Pop)
        
        else:   
            MSE, MAE, path = nn.generate_nn(Gen, X_train, Y_train, X_test, Y_test,  
                                               num_layers, num_neurons, lr, len(X_train[0]),  
                                               activation, optimizer, batch_size, pt)
            
            Population.append([num_layers, num_neurons, lr, activation, optimizer, batch_size, 
                                path, MSE, MAE, "I"])
            Global_pop.append(list(tmp[i] for i in idx))
            
            
        
    return Population, Global_pop

def mutate(Pop, Gen, X_train, Y_train, X_test, Y_test, params, Pop_tmp, pt, Global_Population):
    new_Model = Pop
    Global_pop = Global_Population
    idx = [0,3,4,5]
    [layer, neuron, lr,  activation, optimizer, batchsize] = [new_Model[0], new_Model[1], new_Model[2], new_Model[3], new_Model[4], new_Model[5]]
    tmp = [layer, neuron, lr,  activation, optimizer, batchsize]
    #print(tmp)
    #print(Pop_tmp)
    #while list(tmp[i] for i in idx) in (Global_pop[i][:] for i in range(len(Global_pop))):
    #    print("already in population")
    rand_param = rand.randint(0, len(params)-1)
    rand_hyperparam = params[rand_param][rand.randint(1,len(params[rand_param])-1)]
    new_Model[rand_param] = rand_hyperparam
    [layer, neuron, lr,  activation, optimizer, batchsize] = [new_Model[0], new_Model[1], new_Model[2], new_Model[3], new_Model[4], new_Model[5]]
    tmp = [layer, neuron, lr,  activation, optimizer, batchsize]

    
    Global_pop.append(list(tmp[i] for i in idx))
    MSE, MAE, path = nn.generate_nn(Gen, X_train, Y_train, X_test, Y_test,  
                                           layer, neuron, lr, len(X_train[0]),  
                                           activation, optimizer, batchsize, pt)
        

    return [layer, neuron, lr, activation, optimizer, batchsize, path, MSE, MAE, "M"], Global_pop
    

def crossover(Best_Models, Gen, X_train, Y_train, X_test, Y_test, params, Pop_tmp, pt, Global_Population):
    idx = [0,3,4,5]
    Global_pop = Global_Population
    Father_Model = list(Best_Models[rand.randint(0,len(Best_Models)-1)])
    Father_Performance = Father_Model[7]
    while True:
        Mother_Model = list(Best_Models[rand.randint(0,len(Best_Models)-1)])
        if(Mother_Model != Father_Model):
            break

    Mother_Peformance = Mother_Model[7]
    Child_Model = list(Father_Model)
    Threshold = Mother_Peformance/(Mother_Peformance + Father_Performance)
    
    for i in range(len(Father_Model)-1):
        if i == 1:
            Child_Model[i] = int((1-Threshold) * Mother_Model[i] + Threshold * Father_Model[i])
        elif i == 2: 
            Child_Model[i] = round((1-Threshold) * Mother_Model[i] + Threshold * Father_Model[i], 4)
        else:
            if float(rand.randint(1,100))/100 >= Threshold:
                Child_Model[i] = Mother_Model[i]

    [layer, neuron, lr, activation, optimizer, batchsize] = [Child_Model[0], Child_Model[1], Child_Model[2], Child_Model[3], Child_Model[4], Child_Model[5]]
    tmp = [layer, neuron, lr, activation, optimizer, batchsize]
    
    if list(tmp[i] for i in idx) in (Global_Population[i][:] for i in range(len(Global_Population))): 
        print('Mutation')
        return mutate(list(tmp), Gen, X_train, Y_train, X_test, Y_test, params,list(Pop_tmp), pt, Global_Population)
    
    else:
        print('crossover')
        MSE, MAE, path = nn.generate_nn(Gen, X_train, Y_train, X_test, Y_test,  
                                               layer, neuron, lr, len(X_train[0]),  
                                               activation, optimizer, batchsize, pt)
        Global_pop.append(list(tmp[i] for i in idx))  
        return [layer, neuron, lr, activation, optimizer, batchsize, path, MSE, MAE, "C"], Global_pop




def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)  
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: 
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise
