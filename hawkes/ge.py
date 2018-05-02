from deap import base
from deap import creator
from deap import tools
import random
from evaluate import max_likelihood
IND_SIZE = 5
import pandas as pd
from math import log

#df = pd.read_excel('C:/Users/l_cry/Desktop/pa1.xls')
#df = pd.read_excel('C:/Users/l_cry/Desktop/hs300.xls')
df = pd.read_excel('../hs300.xls')

df.rename(columns={'交易日期_TrdDt':'TrdDt','收盘价(元/点)_ClPr':'ClPr','成交量_TrdVol':'TrdVol',
            '昨收盘(元/点)_PrevClPr':'PrevClPr','成交金额(元)_TrdSum':'TrdSum','流通市值_TMV':'TMV'},inplace=True)

#df['AdjClpr2']= df.AdjClpr2.fillna(method='ffill')

df['ch_pct']=df.ClPr/df.PrevClPr
df['ch_pct_log']=df.ch_pct.dropna().apply(lambda x:-log(x))
df['ch_pct_log_abs'] = df.ch_pct_log.dropna().apply(lambda x :abs(x))
df.TMV = df.TMV.fillna(method='ffill')
df['DTrdTurnR'] = df.TrdSum/df.TMV



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("tao", random.uniform,0.001,0.5)
toolbox.register('fi',random.uniform,0.001,0.5)
toolbox.register('gama',random.uniform,0.001,0.5)
toolbox.register('delta',random.uniform,0.001,2)
toolbox.register('kesi',random.uniform,0.001,0.5)
toolbox.register('beta',random.uniform,0.001,0.5)
toolbox.register('alpha',random.uniform,0.001,0.5)
toolbox.register('v_delta',random.uniform,0.01,0.5)
toolbox.register('v_beta',random.uniform,0.001,0.5)
toolbox.register('v_alpha',random.uniform,0.001,0.5)
toolbox.register('v_gama',random.uniform,0.001,0.5)
IND_SIZE=1
toolbox.register("individual", tools.initCycle, creator.Individual,\
                 (toolbox.tao,toolbox.fi,toolbox.gama,\
                  toolbox.delta,toolbox.kesi,\
                  toolbox.beta,toolbox.alpha,\
                  toolbox.v_delta,toolbox.v_beta,\
                  toolbox.v_alpha,toolbox.v_gama), n=IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
evaluate = lambda x: max_likelihood(df.TrdDt.values,df.ch_pct_log_abs.values,df.DTrdTurnR.values,{'tao':x[0],'fi':x[1],'gama':x[2],'delta':x[3],'kesi':x[4],\
             'beta':x[5],'alpha':x[6],'u':0.031,'ge':20\
            ,'v_delta':x[7],'vu':0.25,'v_beta':x[8],'v_alpha':x[9],'v_gama':x[10]})
toolbox.register("evaluate", evaluate)

def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator
toolbox.decorate("mate", checkBounds(0.001, 2))
toolbox.decorate("mutate", checkBounds(0.001, 2))

def main():
    pop = toolbox.population(n=99)
    ind = creator.Individual([0.0056640625, 0.022437500000000003, 0.05075, 0.5453247070312499, 0.025875000000000002, 0.008796875, 0.001, 0.404716796875, 0.25075, 0.0149921875, 0.04234375])
    pop.append(ind)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 300

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind , fit in zip(pop, fitnesses):
        ind.fitness.values = fit,

    for g in range(NGEN):
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit,

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    return pop

if __name__ == "__main__":
    main()