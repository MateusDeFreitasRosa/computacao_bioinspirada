import numpy as np

phrase = 'SENDMOREMONEY'

first_factor = 'SEND'
second_factor = 'MORE'
result_factor = 'MONEY'

## Define alguns parametros importantes para o algoritmo genetico
pop_size = 500
generation = 100
n_parents_survive = 25

## Este template servirá entendermos quais números em um gene referenciam cada letra.
template = list(set(list(phrase)))

## Obtem qual o tamanho necessário para o gene:
gene_lentgh = len(template)

# Este template define qual o universo possível de valores para o gene.
template_gene = list(range(10))


def generate_individuals(template_gene, gene_lenth, pop_size):
    return np.random.choice(template_gene, size=(pop_size, gene_lentgh))

def fitness_function(population, template):
    global first_factor, second_factor, result_factor
    array_result = []
    
    for individual in population:
        dict_gene = dict(zip(template, individual['gene']))
        
        valor_calculado = sum_array([first_factor, second_factor], dict_gene)
        solucao_proposta = encoder_codification(result_factor, dict_gene)
        array_result.append(abs(valor_calculado - solucao_proposta))
    return array_result

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

def select_random_couples(array_parents):
    return np.random.choice(array_parents, size=2)

def simple_crossover(population, array_parents, pop_size, num_genes):
    
    children = []
    for individual in range(pop_size):
        ## Seleciona 2 pais para o cruzamento
        parentes = select_random_couples(array_parents)
        ## Define uma mascara de quais genes serão pegos entre o pai 1 e o pai 2
        cross_mask = np.random.choice([0,1], size=num_genes)
        ## Construi o filho baseado na mascara acima.
        child = [population[parentes[father]]['gene'][enum]  for enum, father in enumerate(cross_mask)]
        children.append(np.asarray(child))
    return children

def find_duplicates(gene):
    duplicates = []
    for index in range(len(gene)):
        if gene[index] in gene[:index]:
            duplicates.append(index)
    return duplicates

def mutation(children_array, template_gene):
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

def encoder_codification(word, dict_gene, int_encoder=True):
    codification = [dict_gene[letter] for letter in str(word)]
    if int_encoder:
        return int(''.join(map(str, codification)))
    return codification

## Soma uma lista de listas, ou seja: Imagine que temos a seguinte lista para a soma ['SEND', 'MORE'], o objetivo é somar e estes dois fatores 
def sum_array(array, dict_gene, int_encoder=True):
    result = 0
    for factor in array:
        codification = [dict_gene[letter] for letter in str(factor)]
        codification_bs10 = int(''.join(map(str, codification)))
        result +=codification_bs10
        
    if not int_encoder:
        list_result_str = list(str(result).zfill(5))
        list_result_int = [int(x) for x in list_result_str]
        return list_result_int
    return result

def check_if_finish(array_fitness):
    return np.asarray(array_fitness).min() == 0

def decoder(population, template):
    result = []
    for individual in population:
        dict_gene = dict(zip(template, individual['gene']))
        invert_dict_gene = dict(zip(dict_gene.values() , dict_gene.keys()))
        valor_calculado = sum_array(['SEND', 'MORE'], dict_gene, int_encoder=False)
        
        result_aux = []
        for g in valor_calculado:
            if g in invert_dict_gene.keys():
                result_aux.append(invert_dict_gene[g])
            else:
                result_aux.append('_')
        result.append(result_aux)
    return result

def degree_of_convergence(population):
    genes = np.asarray([individual['gene'] for individual in population])
    total_genes = genes.shape[0]
    unique_genes = np.unique(genes, axis=0).shape[0]
    convergence = 1-(unique_genes / total_genes)
    return convergence

## Inicia a população com genes aleatórios.
initial_population = generate_individuals(template_gene, gene_lentgh, pop_size)

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
    children = simple_crossover(population, indexes, pop_size, gene_lentgh)
    children = mutation(children, template_gene)
    new_population = children.copy()
    decoder_population_genes = decoder(population, template)
    population = add_to_array_of_map(population, 'decoder', decoder_population_genes)
    
    best_result = np.asarray(array_fitness).min()
    argmin = np.asarray(array_fitness).argmin()
    best_result_for_population.append(population[argmin])
    decoder_best_result = population[argmin]['decoder']
    convergence = degree_of_convergence(population)
    
    print('Epoch: {} - Best result: {} - Decoder: {} - Convergence: {:.2f}'.format(gen, best_result, decoder_best_result, convergence))
    if best_result == 0:
        break