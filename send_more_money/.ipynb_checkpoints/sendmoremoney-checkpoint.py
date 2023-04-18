
import numpy as np


# In[296]:


first_factor = 'SEND'
second_factor = 'MORE'
result_factor = 'MONEY'
phrase = list( set(first_factor).union(set(second_factor)).union(set(result_factor)) )
phrase


# In[297]:


## Define alguns parametros importantes para o algoritmo genetico

pop_size = 500
pop_random_perc = .1
generation = 1500
n_parents_survive = 10


# In[298]:


pop_size * pop_random_perc


# In[299]:


## Este template servirá entendermos quais números em um gene referenciam cada letra.
template = list(set(list(phrase)))
template


# In[300]:


## Obtem qual o tamanho necessário para o gene:
gene_lentgh = len(template)


# In[301]:


# Este template define qual o universo possível de valores para o gene.
template_gene = list(range(10))
template_gene


# In[302]:


def generate_individuals(template_gene, gene_lenth, pop_size):
    return np.random.choice(template_gene, size=(pop_size, gene_lentgh))


# In[303]:


def add_to_array_of_map(mp, key, array):
    new_array_map = mp.copy()
    for enum, arr in enumerate(array):
        new_array_map[enum][key] = arr
    return new_array_map 


# In[304]:


def fitness_function(population, template):
    global first_factor, second_factor, result_factor
    array_result = []
    
    for individual in population:
        dict_gene = dict(zip(template, individual['gene']))
        
        valor_calculado = sum_array([first_factor, second_factor], dict_gene)
        solucao_proposta = encoder_codification(result_factor, dict_gene)
        array_result.append(abs(valor_calculado - solucao_proposta))
    return array_result


# In[305]:


def roulette(array_fitness, n_parents):
    scores = array_fitness.copy()
    inverted_list = [max(scores) - score for score in scores]
    # Encontre a soma das pontuações invertidas
    total_inverted_score = sum(inverted_list)
    # Calcule as proporções da roleta
    proportions = [s / total_inverted_score for s in inverted_list]
    #[print("{}: {:f}".format(enum, a)) for enum, a in enumerate(proportions)]
    selected_index = np.random.choice(len(scores), size=n_parents, p=proportions)
    return selected_index


# In[306]:


def select_random_couples(array_parents):
    return np.random.choice(array_parents, size=2)


# In[307]:


def pmx_crossover(population, array_parents, pop_size, num_genes, pop_random_perc, template_gene):
    children = []
    
    pop_crossover = int(pop_size * (1-pop_random_perc))
    pop_random = pop_size - pop_crossover
    for individual in range(pop_crossover):
        ## Seleciona 2 pais para o cruzamento
        parentes = select_random_couples(array_parents)
        ## Define uma mascara de quais genes serão pegos entre o pai 1 e o pai 2
        corte_left = np.random.randint(num_genes/2)
        cort_right = np.random.randint(num_genes/2, num_genes)
        
        cross_mask = np.asarray( ([0]*corte_left) + ([1]*(cort_right-corte_left)) + ([0]*(num_genes-cort_right)) )
        ## Construi o filho baseado na mascara acima.
        child = [population[parentes[father]]['gene'][enum]  for enum, father in enumerate(cross_mask)]
        children.append(np.asarray(child))
        
    ## Gera filhos aleatórios para evitar convergência entre os individuos
    generate_random_individuals = generate_individuals(template_gene, num_genes, pop_random)
    
    return np.concatenate( (children, generate_random_individuals) , axis=0)


# In[308]:


def simple_cossover(population, array_parents, pop_size, num_genes, pop_random_perc, template_gene):
    children = []
    
    pop_crossover = int(pop_size * (1-pop_random_perc))
    pop_random = pop_size - pop_crossover
    for individual in range(pop_crossover):
        ## Seleciona 2 pais para o cruzamento
        parentes = select_random_couples(array_parents)
        ## Define uma mascara de quais genes serão pegos entre o pai 1 e o pai 2
        corte = np.random.randint(num_genes)
        cross_mask = np.asarray(( ([0]*corte) + ([1]*(num_genes-corte)) ))
        ## Construi o filho baseado na mascara acima.
        child = [population[parentes[father]]['gene'][enum]  for enum, father in enumerate(cross_mask)]
        children.append(np.asarray(child))
        
    ## Gera filhos aleatórios para evitar convergência entre os individuos
    generate_random_individuals = generate_individuals(template_gene, num_genes, pop_random)
    
    return np.concatenate( (children, generate_random_individuals) , axis=0)


# In[309]:


def random_crossover(population, array_parents, pop_size, num_genes, pop_random_perc, template_gene):
    
    children = []
    pop_crossover = int(pop_size * (1-pop_random_perc))
    pop_random = pop_size - pop_crossover
    for individual in range(pop_crossover):
        ## Seleciona 2 pais para o cruzamento
        parentes = select_random_couples(array_parents)
        ## Define uma mascara de quais genes serão pegos entre o pai 1 e o pai 2
        cross_mask = np.random.choice([0,1], size=num_genes)
        ## Construi o filho baseado na mascara acima.
        child = [population[parentes[father]]['gene'][enum]  for enum, father in enumerate(cross_mask)]
        children.append(np.asarray(child))
      
    ## Gera filhos aleatórios para evitar convergência entre os individuos
    generate_random_individuals = generate_individuals(template_gene, num_genes, pop_random)
    return np.concatenate( (children, generate_random_individuals) , axis=0)


# In[310]:


def find_duplicates(gene):
    duplicates = []
    for index in range(len(gene)):
        if gene[index] in gene[:index]:
            duplicates.append(index)
    return duplicates


# In[311]:


def mutation(children_array, template_gene, max_convergence=.5):
    gene_missings = 0
    children_mutated = children_array.copy()
    
    for enum, child_gene in enumerate(children_array):
        missing_genes = list(set(template_gene).difference(child_gene))
        if missing_genes != []:
            index_with_duplicated_genes = find_duplicates(child_gene)
            for index in index_with_duplicated_genes:
                select_gene_on_missin_gene = np.random.choice(missing_genes)
                missing_genes.remove(select_gene_on_missin_gene)
                children_mutated[enum][index] = select_gene_on_missin_gene
        
    return children_mutated


# In[312]:


def encoder_codification(word, dict_gene, int_encoder=True):
    codification = [dict_gene[letter] for letter in str(word)]
    if int_encoder:
        return int(''.join(map(str, codification)))
    return codification


# In[313]:


## Soma uma lista de listas, ou seja: Imagine que temos a seguinte lista para a soma ['SEND', 'MORE'], o objetivo é somar e estes dois fatores 
def sum_array(array, dict_gene, int_encoder=True):
    global result_factor
    result = 0
    for factor in array:
        codification = [dict_gene[letter] for letter in str(factor)]
        codification_bs10 = int(''.join(map(str, codification)))
        result +=codification_bs10
        
    if not int_encoder:
        list_result_str = list(str(result).zfill(len(result_factor)))
        list_result_int = [int(x) for x in list_result_str]
        return list_result_int
    return result


# In[314]:


def check_if_finish(array_fitness):
    return np.asarray(array_fitness).min() == 0


# In[315]:


def decoder(population, template, first_factor, second_factor):
    
    result = []
    for individual in population:
        dict_gene = dict(zip(template, individual['gene']))
        invert_dict_gene = dict(zip(dict_gene.values() , dict_gene.keys()))
        valor_calculado = sum_array([first_factor, second_factor], dict_gene, int_encoder=False)
        
        result_aux = []
        for g in valor_calculado:
            if g in invert_dict_gene.keys():
                result_aux.append(invert_dict_gene[g])
            else:
                result_aux.append('_')
        result.append(result_aux)
    return result


# In[316]:


def degree_of_convergence(population):
    genes = np.asarray([individual['gene'] for individual in population])
    total_genes = genes.shape[0]
    unique_genes = np.unique(genes, axis=0).shape[0]
    convergence = 1-(unique_genes / total_genes)
    return convergence


# In[317]:


## Inicia a população com genes aleatórios.
initial_population = generate_individuals(template_gene, gene_lentgh, pop_size)


# In[321]:


new_population = initial_population.copy()
best_result_for_population = []
for gen in range(generation):
    ## Adiciona os genes iniciais em uma estrutura de map e salva no array population.
    population = [{'gene': gene} for gene in new_population]
    ## Obtém os resultados de cada gene.
    array_fitness = fitness_function(population, template)
    ## Junta as informações no array de map population.
    population = add_to_array_of_map(population, 'score', array_fitness)  
  
    indexes = roulette(array_fitness, n_parents_survive)
    children = simple_cossover(population, indexes, pop_size, gene_lentgh, pop_random_perc, template_gene)
    children = mutation(children, template_gene)
    new_population = children.copy()
    decoder_population_genes = decoder(population, template, first_factor, second_factor)
    population = add_to_array_of_map(population, 'decoder', decoder_population_genes)
    
    best_result = np.asarray(array_fitness).min()
    argmin = np.asarray(array_fitness).argmin()
    best_result_for_population.append(population[argmin])
    decoder_best_result = population[argmin]['decoder']
    convergence = degree_of_convergence(population)
    
    print('Epoch: {} - Best result: {} - Decoder: {} - Convergence: {:.2f}'.format(gen, best_result, decoder_best_result, convergence))
    if best_result == 0:
        break
