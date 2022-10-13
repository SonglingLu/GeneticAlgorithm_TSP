import random

# initial population creation function
def CreateInitialPopulation(size):
    population = []
    for _ in range(size):
        population.append(random.sample(list(range(num_loc)), k = num_loc))
    return population

# euclidean distance calculation function
def distance(m, n):
    result = ((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2 + (m[2] - n[2]) ** 2) ** 0.5
    return result

# path distance calculation
def pathDistance(path, num_loc, locations):
    cost = 0
    for i in range(num_loc):
        cost += distance(locations[path[i]], locations[path[(i + 1) % num_loc]])
    return cost

# fitness calcultaion
def fitness(x):
    return 1/x

# population and distance ranking function
def rank(distances, population):
    ordered = sorted(zip(distances, population))
    rankedDistances = [x for x, _ in ordered]
    rankedPopulation = [x for _, x in ordered]
    return rankedDistances, rankedPopulation

# mating pool selection function
def CreateMatingPool(population, pathfitness, num_loc, elites):
    pool = []
    sumFitness = sum(pathfitness)
    
    # keep the elites
    for i in range(elites):
        pool.append(population[i])
    
    # select the rest
    for _ in range(len(population) - elites):
        r = random.uniform(0.0, sumFitness)
        for j in range(len(pathfitness)):
            if r <= pathfitness[j]:
                pool.append(population[j])
                break
            else:
                r -= pathfitness[j]
                
    return pool

# breeding function
def breed(pool, population_size, elites):
    nextGen = []
    # keep the elites
    for i in range(elites):
        nextGen.append(pool[i])
    
    # shuffle the mating pool
    random.shuffle(pool)
    # crossover to produce children
    for i in range(population_size - elites):
        p1 = i
        p2 = - (i + 1)
        child = crossover(pool[p1], pool[p2])
        nextGen.append(child)
    
    return nextGen

# crossover function
def crossover(p1, p2):
    start = random.randrange(num_loc)
    end = random.randrange(num_loc)
    if start > end:
        start, end = end, start

    p1_c = p1[start:end + 1]
    p2_c = [x for x in p2 if x not in p1_c]
    child = p2_c[:start] + p1_c + p2_c[start:]
    
    # perform a mutation
    if random.uniform(0, 1) < 0.1:
        child = mutate(child)
    return child

# mutate function
def mutate(child):
    start = random.randrange(num_loc)
    end = random.randrange(num_loc)
    if start > end:
        start, end = end, start
        
    # perform a inversion mutation
    mutated = child[:start] + list(reversed(child[start:end + 1])) + child[end + 1:]
    return mutated






# read in from the input.txt file
with open('input.txt', 'r') as f:
    num_loc = int(f.readline())
    locations = []
    for line in f:
        coordinates = [int(i) for i in line.split()]
        locations.append(coordinates)


population_size = 200
elites = 100
epoch = 1000

#initialize the population
population = CreateInitialPopulation(population_size)

#calculate the distances
distances = []
for i in range(population_size):
    distances.append(pathDistance(population[i], num_loc, locations))

# ranke the initial population
distances, population = rank(distances, population)

#record the shortest distance at each epoch
shortestDist = []
shortestDist.append(distances[0])

# repeat crossover and mutation
for i in range(epoch):
    pathfitness = [fitness(x) for x in distances]

    # mating
    pool = CreateMatingPool(population, pathfitness, num_loc, elites)
    nextGen = breed(pool, population_size, elites)
    
    population = nextGen
    distances = []
    for i in range(population_size):
        distances.append(pathDistance(population[i], num_loc, locations))

    # rank the pathes
    distances, population = rank(distances, population)
    shortestDist.append(distances[0])

#print(shortestDist)
#print(population[0])

# write the path to output
with open('output.txt', 'w') as f:
    for i in range(len(population[0])):
        str_coords = [str(x) for x in locations[population[0][i]]]
        line = ' '.join(str_coords) + '\n'
        f.write(line)

    str_coords = [str(x) for x in locations[population[0][0]]]
    line = ' '.join(str_coords)
    f.write(line)


import matplotlib.pyplot as plt
plt.plot(list(range(epoch + 1)), shortestDist)
plt.show()