initial =[0 ,0 ,0]
parameters =[0 ,0 ,0 ,0 ,0 ,0]
precision =10
print ( length_of_curves ( parameters , initial , precision ) )

#=============================================================================
with open ('D_12_1.pickle ', 'rb') as f :
    M1 = pickle . load ( f ) #First - order derivative
with open ('D_12_2.pickle', 'rb') as f :
    M2 = pickle . load ( f ) #Second - order Hessian
with open ('D_12_3.pickle', 'rb') as f :
    M3 = pickle . load ( f ) #Third - order tensor
with open ('D_12_4.pickle', 'rb') as f :
    M4 = pickle . load ( f )
with open ('D_12_5.pickle', 'rb') as f :
    M5 = pickle . load ( f )
#=============================================================================
print(automorphism_group_quotient_hyperelliptic_involution(12,1))
#=============================================================================
G=automorphism_group_quotient_hyperelliptic_involution(12,0)
all_orbit=get_orbits_of_subset_of_C_with_some_cardinality(12,G,4) #all_orbit is the set of all subset of C with cardinality 4 up to isomorphism, all_orbit[4] has four 4-chains:[1,2,5,7], [1,2,5,8], [1,2,5,11], [1,2,7,11]
print(if_adjacent_to_stratum_3order(12,[1,2,5,7],M1,M2,M3))
print(if_adjacent_to_stratum_3order(12,[1,2,5,11],M1,M2,M3))
print(if_adjacent_to_stratum_3order(12,[1,2,7,11],M1,M2,M3))
print(if_adjacent_to_stratum_3order(12,[1,2,5,8],M1,M2,M3)) #this takes hours
#=============================================================================
A_T=np.array([[0,0,2,0,0,0],
	    [0,1,0,0,0,0],
	    [2+np.sqrt(2),1+np.sqrt(2)/2,2+2*np.sqrt(2),-2*np.sqrt(1+np.sqrt(2)),2*np.sqrt(1+np.sqrt(2)),0],
	    [-2-2*np.sqrt(2),-1,-2-2*np.sqrt(2),2*np.sqrt(2+2*np.sqrt(2)),0,2*np.sqrt(2+2*np.sqrt(2))],
	    [2,0,0,0,0,0],
	    [-2-np.sqrt(2),-1-np.sqrt(2)/2,-2-2*np.sqrt(2),2*np.sqrt(7+5*np.sqrt(2)),2*np.sqrt(1+np.sqrt(2)),2*np.sqrt(2+2*np.sqrt(2))]])
A=A_T.T
A_inv=np.linalg.inv(A)
x=symbols('x')
y=symbols('y')
v1=np.dot([-1,x,y,-1,3-x+y,-1],A_inv)
ff=[]
for i in range(12):
	ff.append(simplify(np.dot(np.dot(v1,M2[i]),v1)))
print((ff[7]-ff[0]-ff[9]+ff[3]-ff[10]))
#=============================================================================
vv=[-1,1,1,-1,3,-1]
v1=np.dot(vv,A_inv) #A_inv is as before
print(v1)
vv1=[1.5/3,1/3,-0.5/3,0.32179713/3,-0.77688699/3,0.77688699/3]
print(if_adjacent_to_stratum_3order(12,[1,4,10,11,12],M1,M2,M3,vv1))
