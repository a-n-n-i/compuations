
import pickle
import AllFunctions
import itertools

parameter=[-0.06285556753370763949511129726305908862772147851980580123171816356121368068899110013161906643192060040919404038442229618162633441934071541770622657966444780606, -0.125711135067415278990222594526118177255442957039611602463436327122427361377982200263238132863841200818388080768844592363252668838681430835412453159328895612133, -0.0628555675337076394951112972630590886277214785198058012317181635612136806889911001316190664319206004091940403844222961816263344193407154177062265796644478060665, -0.38434494644064063487056189797209379978236669557702368968774104897549973387195268100209120274560562286793771229689822066387339235522849676358499860365285874814014, 0.384344946440640634870561897972093799782366695577023689687741048975499733871952681002091202745605622867937712296898220663873392355228496763584998603652858748140147, -0.01670827051616703987453704987568812148455895868972907586825405012595463382764688765282630084431802125966159703452413533402667644193065203076134264472640278308227]
initial=[-0.06285556753370763949511129726305908862772147851980580123171816356121368068899110013161906643192060040919404038442229618162633441934071541770622657966444780606657, -0.125711135067415278990222594526118177255442957039611602463436327122427361377982200263238132863841200818388080768844592363252668838681430835412453159328895612133152, -0.062855567533707639495111297263059088627721478519805801231718163561213680688991100131619066431920600409194040384422296181626334419340715417706226579664447806066576]
print(AllFunctions.length_of_curves(parameter, initial,50))


parameter=[0,0,0,0,0,0]
initial=[0,0,0]
print(AllFunctions.differential_of_curves(parameter, initial,50))
#=================================================
'''
min_params, min_value = AllFunctions.gradient_descent_convex(  
    func=AllFunctions.length_function,
    coef=[1,1,0,0,0,0,0,1,1,0,0,0],  
    initial_params=[0,0,0,0,0,0],
    initial=[0, 0, 0],    
    learning_rate= 0.003,  
    max_iter=10000,  
    tol=1e-15 ,
    precision=40 
)  
'''
#==================================================
LL=AllFunctions. automorphism_group_quotient_hyperelliptic_involution(12,0)
def combinations_k1k2(k1, k2):
    if k1 <= 0 or k2 <= 0 or k1 < k2:
        return []
    
    result = []
    elements = list(range(k1))  
    
    def backtrack(start, path):
        if len(path) == k2:
            result.append(path.copy())
            return
        
        for i in range(start, len(elements) - (k2 - len(path)) + 1):
            path.append(elements[i])
            backtrack(i + 1, path)  
            path.pop()
    
    backtrack(0, [])
    return result
all_orbit=[]
for k in range(13):
    all_cases=combinations_k1k2(12, k)
    orbit=[]
    for v in all_cases:
        ii=0
        for gg in LL:

            v1=[]
            for i in range(len(v)):
                v1.append(gg[v[i]]-1)
            if sorted(v1) not in orbit:
                ii=ii+1
        #print(ii)
        if ii==len(LL):
            orbit.append(sorted(v))
    #print(k)     
    #print(orbit)
    #print(len(orbit))
    #print("-----------------")
    all_orbit.append(orbit)

all_orbit1=[]
for L in all_orbit:
    for v in L:
        v1=[]
        for a in v:
            v1.append(a+1)
        all_orbit1.append(v1)
#print(all_orbit1)
#print(combinations_k1k2(12, 11))


def load_MM_pickle(file_name='D_12_1.pickle'):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

DD= load_MM_pickle()
#print(DD)
def load_MM2_pickle(file_name='D_12_2.pickle'):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


MM2 = load_MM2_pickle()
#print(type(MM3_new[0][0][0][0])) 
#print(MM2[2][0])
def load_MM3_pickle(file_name='D_12_3.pickle'):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

MM3 = load_MM3_pickle()
#print(type(MM3_new[0][0][0][0])) 
#print(MM3[2][0])

'''
#print(AllFunctions.if_adjacent_to_stratum(12,[1,2,3,8,9,12],DD,MM2,MM3))
CC1=[]
for LL1 in all_orbit1:
    print(LL1)
    print(AllFunctions.if_adjacent_to_stratum(12,LL1,DD,MM2,MM3))
    #if if_adjacent_to_stratum(12,LL1,DD,MM2,MM3)==1:
      #  CC1.append(LL1)
    print("---------------------------")
'''
#=======================================


def load_MM_pickle(file_name='D_9_1.pickle'):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

DD= load_MM_pickle()
#print(DD)
def load_MM2_pickle(file_name='D_9_2.pickle'):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


MM2 = load_MM2_pickle()
#print(type(MM3_new[0][0][0][0])) 
#print(MM2[2][0])
def load_MM3_pickle(file_name='D_9_3.pickle'):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

MM3 = load_MM3_pickle()
#print(type(MM3_new[0][0][0][0])) 
#print(MM3[2][0])

print(AllFunctions.if_adjacent_to_stratum(9, [1,2,3,8,9,12],DD,MM2,MM3))


