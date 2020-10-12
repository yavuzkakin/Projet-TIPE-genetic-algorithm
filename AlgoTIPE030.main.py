import numpy as np
from random import *
from math import * 
from operator import itemgetter
import matplotlib.pyplot as plt
import time
  


####################################################################################################################
V_000 = [2,20]
V_001 = [2,1]
V_002 = [3,20]
V_003 = [2,2]
V_004 = [3,2]
V_005 = [4,1]


V_006 = [2,2]
V_007 = [2,2]
V_008 = [3,4]



liste_de_vol001 = [V_000,V_001,V_002]
liste_de_vol002 = [V_000,V_001,V_002,V_003]
liste_de_vol003 = [V_000,V_001,V_002,V_003,V_004]
liste_de_vol004 = [V_000,V_001,V_002,V_003,V_004,V_005]
liste_de_vol005 = [V_000,V_001,V_002,V_003,V_004,V_005,V_006]
liste_de_vol006 = [V_000,V_001,V_002,V_003,V_004,V_005,V_006,V_007]
liste_de_vol007 = [V_000,V_001,V_002,V_003,V_004,V_005,V_006,V_007,V_008]
liste_de_vol008 = [[25, 2], [41, 4], [25, 2], [89, 12], [37, 7], [39, 8], [100, 9], [74, 12], [73, 7], [96, 4], [14, 1], [10, 5], [32, 6], [72, 7], [34, 12], [99, 12], [93, 10], [96, 12], [7, 11], [20, 3], [48, 6], [95, 12], [37, 8], [91, 12], [16, 11], [75, 8], [48, 3], [60, 10], [82, 8], [63, 9], [57, 10], [80, 12], [61, 9], [100, 4], [8, 2], [10, 9], [78, 12], [11, 11], [4, 4], [53, 4], [96, 3], [43, 8], [52, 5], [24, 6], [99, 11], [20, 3], [94, 2], [70, 7], [47, 3], [21, 1], [39, 8], [81, 1], [72, 11], [30, 12], [25, 7], [60, 10], [54, 7], [16, 10], [98, 10], [52, 10], [41, 1], [74, 4], [47, 11], [44, 10], [10, 9], [13, 2], [82, 2], [18, 2], [18, 1], [80, 3], [41, 7], [98, 7], [9, 11], [28, 3], [63, 3], [37, 7], [68, 12], [57, 7], [38, 12], [80, 12], [80, 4], [48, 6], [94, 12], [17, 12], [69, 12], [76, 11], [40, 2], [83, 1], [33, 1], [82, 4], [2, 12], [48, 7], [21, 1], [35, 5], [1, 4], [57, 7], [56, 9], [48, 8], [43, 5], [93, 2]]
liste_de_vol009 = [[3, 2], [10, 3], [3, 2], [7, 1], [9, 3], [8, 1], [7, 1], [14, 1], [5, 3], [12, 1], [14, 1], [5, 2], [15, 1], [10, 1], [2, 3]]
V00=[1,4]
V01=[1,65]
V02=[1,2]
V03=[2,2]
V04=[2,4]
liste_de_vol010=[V00,V01,V02,V03,V04]

liste_de_vol012 = [[25, 0], [41, 0], [25, 0], [89, 50], [37, 50], [39, 0], [100, 50], [74, 500], [89, 1], [89, 1], [89, 1], [89, 5], [89, 6], [89, 7], [89, 1], [89, 12], [93, 10], [96, 12], [7, 11], [20, 3], [48, 6], [95, 12], [37, 8], [91, 12], [16, 11], [75, 8], [48, 3], [60, 10], [82, 8], [63, 9], [57, 10], [80, 12], [61, 9], [100, 4], [8, 2], [10, 9], [78, 12], [11, 11], [4, 4], [53, 4], [96, 3], [43, 8], [52, 5], [24, 6], [99, 11], [20, 3], [94, 2], [70, 7], [47, 3], [21, 1], [39, 8], [81, 1], [72, 11], [30, 12], [25, 7], [60, 10], [54, 7], [16, 10], [98, 10], [52, 10], [41, 50], [74, 4], [47, 11], [44, 10], [10, 9], [13, 2], [82, 2], [18, 2], [18, 1], [80, 3], [41, 7], [98, 7], [9, 11], [28, 3], [63, 3], [37, 7], [68, 12], [57, 7], [38, 12], [80, 12], [80, 4], [48, 6], [94, 12], [17, 12], [69, 12], [76, 11], [40, 2], [83, 1], [33, 1], [82, 4], [2, 12], [48, 7], [21, 1], [35, 5], [1, 4], [57, 7], [56, 9], [48, 8], [43, 5], [93, 2]] 

        
# permutation =  [liste représentant le (tour attibué aléatoirement au vol i) - 1 de liste_de_vol pour i dans 0,len(liste_de_vol) - 1 ] 
# régulation = [permutation, coût de la permutation]

####################################################################################################################

def perm_to_tour(permutation):
    return [permutation[i] + 1 for i in range(len(permutation))]
    
    
def cout_permutation(liste_de_vol, permutation):
    S = 0
    for i in range(len(liste_de_vol)):
        S += abs(liste_de_vol[i][0]-(permutation[i]+1))*(liste_de_vol[i][-1])
        # abs(tour souhaité par i  - tour donné pour i par la permutation)* coût du déplacement
    return S 
    
def tirage_couple_sans_remise(n):
    liste = list(range(n))
    a = randint(0, n-1)
    a = liste[a]
    del liste[a]
    b = randint(0, n-2)
    b = liste[b]
    return (a, b)
    
def generate_n_random_regulations(n, liste_de_vol):
    liste_permutations = []
    for i in range(n):
        liste_permutations.append(list(np.random.permutation(len(liste_de_vol))))
    liste_regulations = [(liste_permutations[i],cout_permutation(liste_de_vol,liste_permutations[i])) for i in range(n)]
    liste_regulations = sorted(liste_regulations,key = itemgetter(1)) 
    return liste_regulations
    
"""Prends une liste de régulations et renvoie la liste des probabilités d'être choisi au tour suivant pour chaque régulation"""

def prob(liste_regulations,p_s):
    #p_s = pression séléctive
    S = 0
    liste_prob = []
    for element in liste_regulations:
        S += (1/element[1])**p_s
    for element in liste_regulations:
        liste_prob.append((1/element[1])**p_s/S)
    return liste_prob
    

def selection(liste_regulations, n, elitism, p_s):
    indices = np.arange(len(liste_regulations))# Nécessaire pour utiliser random.choice
    list_prob = prob(liste_regulations, p_s)
    selection_indices = np.random.choice(indices, (n//2) , replace = False, p = list_prob)
    # on sélectionne n//2 individus que l'on va faire reproduire 
    selection_indices_lili = selection_indices.tolist()
    selection_indices_lili = sorted(selection_indices_lili)
    #On range les indices par ordre croissant afin de pouvoir les sélectionner en connaissant leur coût.
    selection_list = [liste_regulations[i] for i in selection_indices_lili]
    selection_list = sorted(selection_list, key = itemgetter(1))
    return selection_list
    

def reproduction (parent1, parent2, liste_de_vol):
    p1 = parent1[:][0][:]
    p2 = parent2[:][0][:]
    fils1 = [-1]*(len(p1))
    fils2 = [-1]*(len(p2))
    genes_choosed = []
    numberofgenes = len(liste_de_vol)//2 #ARBITRAIRE
    for i in range(numberofgenes):  # on choisit les genes aléatoirement
        a = randint(0,len(p1)-1)
        genes_choosed.append(a) 
    for k in range(len(fils1)):
        if k in genes_choosed:
            fils1[k] = p1[k]
            p2.remove(p1[k])  
    i = 0
    for k in range(len(fils1)):
        if fils1[k] == -1:
            fils1[k] = p2[i]
            i += 1
    p1 = parent1[:][0][:]
    p2 = parent2[:][0][:]
    for k in range(len(fils2)):
        if k in genes_choosed:
            fils2[k] = p2[k]
            p1.remove(p2[k])
    i = 0
    for k in range(len(fils2)):
        if fils2[k] == -1:   # Pour ne pas interferer avec la valeur de 0
            fils2[k] = p1[i]
            i += 1
    fils1_c = (fils1, cout_permutation(liste_de_vol, fils1))
    fils2_c = (fils2, cout_permutation(liste_de_vol, fils2))
    return [fils1_c,fils2_c]
    
def nouvelle_gen(une_selection, liste_de_vol, n):     #On reproduit une liste de regulations
    nvelle_liste = une_selection[:]
    while len(nvelle_liste) < n:
        (A, B) = tirage_couple_sans_remise(len(une_selection))
        parentA = une_selection[A]
        parentB = une_selection[B]
        nvelle_liste += reproduction(parentA ,parentB ,liste_de_vol)
    nvelle_liste = sorted(nvelle_liste,key = itemgetter(1)) 
    return nvelle_liste
    
def mutation(regulation, liste_de_vol):   #Rien qu'une transposition (Régulation= permutation+ cout)
    liste_reg = list(regulation)
    a = randint(0, len(regulation[0]) - 1) 
    b = randint(0, len(regulation[0]) - 1)
    (liste_reg[0][a], liste_reg[0][b]) = (liste_reg[0][b], liste_reg[0][a])
    liste_reg[1] = cout_permutation(liste_de_vol, liste_reg[0])
    regulation = tuple(liste_reg)
    return regulation
    
def nouvelle_gen_mutee(nvelle_liste, liste_de_vol,n,nb_mut,elitism): #nvelle_liste de nouvelle_gen
    nvelle_liste_mutee = nvelle_liste[:]
    for i in range(nb_mut): #arbitraire = taux mutation
        a = randint(0, len(nvelle_liste)-1) # 0 ou 1  
        nvelle_liste_mutee[a] = mutation(nvelle_liste_mutee[a], liste_de_vol)
    for i in range(len(nvelle_liste_mutee)):  #Obligation de rajouter cette boucle sinon on obtient pour une raison quelconque des mauvais coûts de régulation
        lili_nvelle_liste_mutee_i = list(nvelle_liste_mutee[i])
        lili_nvelle_liste_mutee_i[-1] = cout_permutation(liste_de_vol, lili_nvelle_liste_mutee_i[0])
        nvelle_liste_mutee[i] = tuple(lili_nvelle_liste_mutee_i)
    nvelle_liste_mutee = sorted(nvelle_liste_mutee,key = itemgetter(1)) 
    return nvelle_liste_mutee
    
def selection_naturelle_sur_n_indivudus_pendant_m_generations(n, m, liste_de_vol, elitism, nb_mut, p_s):
    liste_regulations = generate_n_random_regulations(n, liste_de_vol)
    liste_c_representants = [liste_regulations[0][-1]]
    for i in range(m):
        liste_regulations = selection(liste_regulations,n, elitism, p_s)
        elite_squad = [liste_regulations[k] for k in range(elitism)]
        liste_regulations = nouvelle_gen(liste_regulations, liste_de_vol, n)
        liste_regulations = nouvelle_gen_mutee(liste_regulations, liste_de_vol,n, nb_mut,elitism)
        for k in range(elitism):
            liste_regulations.insert(0, elite_squad.pop())
            liste_regulations.pop()
        liste_regulations = sorted(liste_regulations, key = itemgetter(1))
        liste_c_representants.append(liste_regulations[0][-1])
    return (liste_regulations,liste_c_representants)
    """
    for k in range(elitism):
        selection_indices_lili.insert(0,k) # on entre la valeur k à la position 0
        selection_indices_lili.pop() # Grâce au sorted 
    """
################################################################################################################

    
def graphique_cout_representants(n, m, liste_de_vol, elitism, nb_mut, p_s):
    plt.xlabel('i-ème génération')
    plt.ylabel('Coût de la permutation la plus performante')
    X = list(range(m+1))
    Y = selection_naturelle_sur_n_indivudus_pendant_m_generations(n, m, liste_de_vol, elitism, nb_mut, p_s)[1]
    plt.plot(X,Y,'r')
    plt.title('Cout du représentant')
    #plt.savefig(r'C:\Users\Hassan Berrada\Desktop\MP\TIPE\Graphiques TIPE\ps ='+str(int(p_s))+';elitism ='+str(int(elitism))+';taille_pop ='+str(int(n))+';nb_mut ='+str(int(nb_mut))+';nb de gen ='+str(int(m))+'.jpg')
    plt.show()
    #changenamefile
    #fi = open('C:/Users/Hassan Berrada/Desktop/tipe_stat_3/ps ='+str(int(p_s))+';elitism ='+str(int(elitism))+';taille_pop ='+str(int(n))+';nb_mut ='+str(int(nb_mut))+';nb de gen ='+str(int(m))+'.txt','w')
    #fi =  open('E:/tipestat_2/p-s = 10;elitism = n_25;taille_pop = 500;nb_mut = n_2,nb de gen = ' + str(int(m)) + '.txt','w')
    #changenamefile
    """
    for k in range(len(Y)):
        fi.write(str(Y[k])+'\n')
    fi.close()"""

def time_selection_nat(n, m, liste_de_vol, elitism, nb_mut, p_s):
    start = time.time()    
    selection_naturelle_sur_n_indivudus_pendant_m_generations(n, m, liste_de_vol, elitism, nb_mut, p_s)
    end = time.time()
    return end - start
    
#####################################################################################################
    
def graph_etude_stat_nvari(n, m, liste_de_vol, elitism, nb_mut, p_s, e):
    start = time.time()    
    Y = selection_naturelle_sur_n_indivudus_pendant_m_generations(n, m, liste_de_vol, elitism, nb_mut, p_s)[1]
    end = time.time()
    #fi = open('C:/Users/Hassan Berrada/Desktop/tipe_stat_'+str(int(e))+'/nvari/ps ='+str(int(p_s))+';elitism ='+str(int(elitism))+';taille_pop ='+str(int(n))+';nb_mut ='+str(int(nb_mut))+';nb de gen ='+str(int(m))+'.txt','w')
    for k in range(len(Y)):
        fi.write(str(Y[k])+'\n')
    fi.close()
    return end - start
    
def graph_etude_stat_elitismvari(n, m, liste_de_vol, elitism, nb_mut, p_s, e):
    start = time.time()    
    Y = selection_naturelle_sur_n_indivudus_pendant_m_generations(n, m, liste_de_vol, elitism, nb_mut, p_s)[1]
    end = time.time()
    #fi = open('C:/Users/Hassan Berrada/Desktop/tipe_stat_'+str(int(e))+'/elitismvari/ps ='+str(int(p_s))+';elitism ='+str(int(elitism))+';taille_pop ='+str(int(n))+';nb_mut ='+str(int(nb_mut))+';nb de gen ='+str(int(m))+'.txt','w')
    """for k in range(len(Y)):
        fi.write(str(Y[k])+'\n')
    fi.close()"""
    return end - start
    
def graph_etude_stat_tauxmutvari(n, m, liste_de_vol, elitism, nb_mut, p_s, e):
    start = time.time()    
    Y = selection_naturelle_sur_n_indivudus_pendant_m_generations(n, m, liste_de_vol, elitism, nb_mut, p_s)[1]
    end = time.time()
    #fi = open('C:/Users/Hassan Berrada/Desktop/tipe_stat_'+str(int(e))+'/tauxmutvari/ps ='+str(int(p_s))+';elitism ='+str(int(elitism))+';taille_pop ='+str(int(n))+';nb_mut ='+str(int(nb_mut))+';nb de gen ='+str(int(m))+'.txt','w')
    """for k in range(len(Y)):
        fi.write(str(Y[k])+'\n')
    fi.close()"""
    return end - start

def graph_etude_stat_p_svari(n, m, liste_de_vol, elitism, nb_mut, p_s, e):
    start = time.time()
    Y = selection_naturelle_sur_n_indivudus_pendant_m_generations(n, m, liste_de_vol, elitism, nb_mut, p_s)[1]
    end = time.time()
    #fi = open('C:/Users/Hassan Berrada/Desktop/tipe_stat_'+str(int(e))+'/p_svari/ps ='+str(int(p_s))+';elitism ='+str(int(elitism))+';taille_pop ='+str(int(n))+';nb_mut ='+str(int(nb_mut))+';nb de gen ='+str(int(m))+'.txt','w')
    """for k in range(len(Y)):
        fi.write(str(Y[k])+'\n')
    fi.close()"""
    return end - start
    
def etude_stat_3():
    e = 3
    m = 2000
    liste_n = [10, 20, 50, 75, 100, 200, 300, 500]
    liste_ps = [1, 2, 5, 10, 12, 20, 30]
    liste_inv_tauxmut = [100, 50, 20, 10, 6, 3, 2]
    liste_prop_elitism = [100, 50, 20, 15, 10, 6, 3]
    for n in liste_n:
        graph_etude_stat_nvari(n, m, liste_de_vol008, n//7, n//10, 10, e)
    for p_s in liste_ps:
        graph_etude_stat_p_svari(100, m, liste_de_vol008, 100//7, 10, p_s, e)
    for tm in liste_inv_tauxmut:
        graph_etude_stat_tauxmutvari(100, m, liste_de_vol008, 100//7, 100//tm, 10, e)
    for el in liste_prop_elitism:
        graph_etude_stat_elitismvari(100, m, liste_de_vol008, 100//el, 10, 10, e)
        
        


def etude_stat_4():
    e = 4
    m = 2000
    liste_n = [10, 20, 50, 75, 100, 200, 300, 500]
    liste_ps = [1, 2, 5, 10, 12, 20, 30]
    liste_inv_tauxmut = [100, 50, 20, 10, 6, 3, 2]
    liste_prop_elitism = [50, 20, 15, 10, 6, 4, 3, 2]
    for n in liste_n:
        graph_etude_stat_nvari(n, m, liste_de_vol008, n//6, n//10, 12, e)
    for p_s in liste_ps:
        graph_etude_stat_p_svari(100, m, liste_de_vol008, 100//6, 10, p_s, e)
    for tm in liste_inv_tauxmut:
        graph_etude_stat_tauxmutvari(100, m, liste_de_vol008, 100//6, 100//tm, 12, e)
    for el in liste_prop_elitism:
        graph_etude_stat_elitismvari(100, m, liste_de_vol008, 100//el, 10, 12, e)
        
def etude_stat_rapide():
    etude_stat_3()
    etude_stat_4()
        

def etude_complexite_taille_pop(n_max, liste_de_vol, e = 5):
    Y = [time_selection_nat(n, 1000, liste_de_vol, n//6, n//10, 10) for n in range(5,n_max,2)]
    fi = open('C:/Users/Hassan Berrada/Desktop/tipe_stat_'+str(int(e))+'/étude de complexité/Etude de complexité en fonction de la taille de la population_2.txt','w')
    for k in range(len(Y)):
        fi.write(str(Y[k])+'\n')
    fi.close()
    
def etude_complexite_taille_liste(liste_de_liste_de_vol, e = 5):
    Y = []
    for i in range(len(liste_de_liste_de_vol)):
        n = int(len(liste_de_liste_de_vol[i]))
        print(int(n))
        Y.append(time_selection_nat(n, 100, liste_de_liste_de_vol[i], n//6, n//10, 10) for i in range(len(liste_de_liste_de_vol)))
        print(Y)
    fi = open('C:/Users/Hassan Berrada/Desktop/tipe_stat_'+str(int(e))+'/étude de complexité/Etude de complexité en fonction de la taille de la liste_2.txt','w')
    for k in range(len(Y)):
        fi.write(str(Y[k])+'\n')
    fi.close()
   
    
""" liste de vol réparti par une gaussienne 
grand ou petit écart type de cout
Rassembler les représentants de plusieurs évolutions """

#####################################################################################################