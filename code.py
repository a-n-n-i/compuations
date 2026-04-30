
import numpy as np  
from scipy.optimize import fsolve  
import math
from decimal import Decimal, getcontext
from gurobipy import Model, GRB, quicksum
import pickle
import mpmath as mp

# ============================================================================
# Auxiliary Geometric Functions: Hyperbolic Geometry Calculations
# These functions implement fundamental geometric operations in hyperbolic plane
# for constructing the parameterization of the Techmuller space
# ============================================================================



def distance_between_points(w,z):
    """
    Calculate hyperbolic distance between two points in the upper half-plane model.
    
    Uses the formula: log(|w̅ - z| + |w - z|) - log(|w̅ - z| - |w - z|)
    where w̅ denotes the complex conjugate of w.
    
    Args:
        w, z: Complex numbers representing points in the upper half-plane
        
    Returns:
        Hyperbolic distance between w and z
    """
    dom=abs(w.conjugate()-z)+abs(w-z)
    num=abs(w.conjugate()-z)-abs(w-z)
    return mpmath.log(dom/num)

def distance_between_semicircle_and_semicircle(c1,r1,c2,r2):
    """
    Calculate hyperbolic distance between two semicircles (representing geodesics).
    
    Used for computing distances between different edges of the Bolza surface.
    
    Args:
        c1, r1: Center and radius of first semicircle
        c2, r2: Center and radius of second semicircle
        
    Returns:
        Hyperbolic distance between the two semicircles
    """
    if c1!=c2:
        cx=(r2**2-r1**2+c1**2-c2**2)/(2*(c1-c2))
        r=mpmath.sqrt((c1-cx)**2-r1**2)
        p1=(r1**2/(cx-c1)+c1)+(r*r1/abs(c1-cx))*1j
        p2=(r2**2/(cx-c2)+c2)+(r*r2/abs(c2-cx))*1j
        #print("p1",p1)
        #print("p2",p2)
        return distance_between_points(p1,p2)
    if c1==c2:
        return distance_between_points(r1*1j,r2*1j)
#print(distance_between_semicircle_and_semicircle(0,1,0,2))


def get_geodesic_from_two_points(z1,z2):
    """
    Find the center and radius of the geodesic (semicircle) passing through two points.
    
    Used for constructing the 12-gon representation of the Bolza surface.
    
    Args:
        z1, z2: Complex numbers representing points in the upper half-plane
        
    Returns:
        Tuple (center, radius) of the geodesic semicircle
    """
    x1=z1.real
    y1=z1.imag
    x2=z2.real
    y2=z2.imag
    xc=(y1+y2)*(y1-y2)/(2*(x1-x2))+(x1+x2)/2
    r=mpmath.sqrt((x1-xc)**2+y1**2)
    return (xc,r)

def from_endpoint_to_next(c1,r1,p1,thee1,l2):
    """
    Transform from one endpoint to the next along the polygon boundary.
    
    Implements Möbius transformations corresponding to rotations and scalings
    along the edges of the 12-gon.
    
    Args:
        c1, r1: Center and radius of current geodesic
        p1: Current endpoint
        thee1: Angle parameter
        l2: Length parameter
        
    Returns:
        Next endpoint coordinates
    """
    M2=mpmath.matrix([[mpmath.cos(mpmath.pi-thee1)+1, -mpmath.exp(l2)*mpmath.sin(thee1)], [mpmath.sin(thee1), mpmath.exp(l2)*(mpmath.cos(mpmath.pi-thee1)+1)]])
    p2_2=(M2[0,0]*1j+M2[0,1])/(M2[1,0]*1j+M2[1,1])
    #print(p2_2)
    x0=p1.real
    y0=p1.imag
    cc=(x0+y0*1j+r1-c1)/((c1+r1-x0)*1j+y0)
    M1=mpmath.matrix([[(c1+r1)*cc,c1-r1],[cc,1]])
    p2=(M1[0,0]*p2_2+M1[0,1])/(M1[1,0]*p2_2+M1[1,1])   
    return p2

def get_intersection_between_two_geodesics(c1,r1,c2,r2):
    """
    Calculate intersection point of two hyperbolic geodesics (semicircles).
    
    Used for determining intersections between different strata of the Bolza surface.
    
    Args:
        c1, r1: Center and radius of first geodesic
        c2, r2: Center and radius of second geodesic
        
    Returns:
        Complex number representing intersection point
    """
    x=(1/2)*((r1**2-r2**2)/(c2-c1)+c1+c2)
    y=abs(mpmath.sqrt(r1**2-(x-c1)**2))
    return x+y*1j

def find_angle(p,p1,p2):
    """
    Calculate the angle at point p between the geodesics to p1 and p2.
    
    Used for computing interior angles of the 12-gon vertices.
    
    Args:
        p: Vertex point where angle is measured
        p1, p2: Points defining the two geodesics
        
    Returns:
        Angle in radians at point p
    """
    [c1,r1]=get_geodesic_from_two_points(p,p1)
    x0=p.real
    y0=p.imag
    cc=(x0+y0*1j+r1-c1)/((c1+r1-x0)*1j+y0)
    M1=mpmath.matrix([[(c1+r1)*cc,c1-r1],[cc,1]])
    M2=mpmath.inverse(M1)
    pp=(M2[0,0]*p+M2[0,1])/(M2[1,0]*p+M2[1,1])
    pp2=(M2[0,0]*p2+M2[0,1])/(M2[1,0]*p2+M2[1,1])
    [c2,r2]=get_geodesic_from_two_points(pp,pp2)
    the1=mpmath.atan(c2/pp.imag) 
    return mpmath.pi/2+the1


# ============================================================================
# Core Function: Compute Lengths of All Systoles on Bolza Surface
# This implements the parameterization algorithm described in the README
# ============================================================================



def length_of_curves(x,initial=[0,0,0]):
    """
    Compute lengths of all 12 systoles for the Bolza surface at given parameters.
    
    Mathematical Background:
    According to Reference [1], cutting the Bolza surface along a 4-chain yields
    a 12-gon. Parameterizing this polygon requires 3 angle parameters and 6 edge
    length parameters. Given any 6 parameters, the remaining 3 can be determined
    numerically through closure conditions.
    
    Args:
        x: 6-dimensional parameter vector [l0, l1, l2, the0, the1, the2]
           - l0, l1, l2: Three free edge length parameters
           - the0, the1, the2: Three free angle parameters
        initial: Initial guess for numerical solver
        
    Returns:
        length_systole_list_ordered: List of 12 systole lengths in order
        
    """
    mpmath.mp.dps = 300  # Set high precision for numerical stability
    
    # Base geometric parameters of Bolza surface
    a=mpmath.acosh(mpmath.csc(mpmath.pi/8)/2)  
    L=2*a
    m4=3  # Corresponds to 12-gon structure
    # Base systole lengths and angles
    length_list=[1,2,1,1,2,1,1,2,1,1,2,1]
    for i in range(len(length_list)):
        length_list[i]=length_list[i]*L
    angle_list=[3,1,3,1,3,1,3,1,3,1,3,1]
    for i in range(len(angle_list)):
        angle_list[i]=angle_list[i]*mpmath.pi/4
        
    # Edge pairing relations in the 12-gon
    Pairs=[[6,8],[0,2],[1,7],[10,4],[3,5],[11,9]]
    # Color grouping for identifying different systole collections
    color_pairs=[[0,1],[2],[3],[4,5]]
    
    # Direction markers: which parameters are free vs dependent
    direction=[1,2,4]    # Indices of free parameters
    all_direction=[]
    for i in range(2*m4):
        if i not in direction:
            all_direction.append(i)
            
    # Numerical solver tolerances and limits
    Epsilen=mpmath.mpf('1e-60')
    M1=mpmath.matrix([[1, 0], [0, 1]])
    C=M1
    for i in range(4*m4):
        M2=mpmath.matrix([[-mpmath.cos(angle_list[i])+1, -mpmath.exp(length_list[i])*mpmath.sin(angle_list[i])], [mpmath.sin(angle_list[i]), mpmath.exp(length_list[i])*(-mpmath.cos(angle_list[i])+1)]])
        C=M2*C


    A=[1,1,1,1,1,1]
    CC=[1,2,7,8,9,11]
    L3=[[2,8],[5,11],[2, 5, 11, 8]]
    LL4=[[2,8],[5,11],[2,5,11,8]]
    tolerance = mpmath.mpf('1e-200')    
    max_steps = 10000000000
    
    # Extract free parameters
    l0=x[0]
    l1=x[1]
    l2=x[2]
    the0=x[3]
    the1=x[4]
    the2=x[5]    
    # Angle grouping
    Angles=[[10,9,4,3],[11,8,5,2],[0,7,6,1]]
    
    Epsilen4=1e-120  
    def equations2(y0,y1,y2):
        """
        Define closure conditions for the 12-gon.
        
        The 12-gon must close on itself, which provides constraints on the
        dependent parameters. This implements the numerical solving part
        mentioned in the README.
        """
        
        theta1 = [ angle_list[Angles[i][0]] for i in range(m4)] 
        theta1[0]=theta1[0]+the0
        theta1[1]=theta1[1]+the1
        theta1[2]=theta1[2]+the2
        theta=[angle_list[i] for i in range(4*m4)] 
        for i in range(m4):
            theta[Angles[i][0]]=theta1[i]
            theta[Angles[i][1]]=mpmath.pi-theta1[i]
            theta[Angles[i][2]]=theta1[i]
            theta[Angles[i][3]]=mpmath.pi-theta1[i]
        z = [mpmath.mpf(0) for _ in range(m4*2)]  
        z[direction[0]]=y0
        z[direction[1]]=y1
        z[direction[2]]=y2
        z[all_direction[0]]=l0
        z[all_direction[1]]=l1
        z[all_direction[2]]=l2
        x=[mpmath.mpf(0) for _ in range(m4*4)]  
        for i in range(m4*2):
            x[Pairs[i][0]]=z[i]
            x[Pairs[i][1]]=z[i]

        # Build complete transformation matrix and check closure condition
        C=M1
        for i in range(4*m4):
        #M2=(1/(2*x[i]))**(1/2)*np.array([[1, -x[i]], [1, x[i] ]])
            M2=mpmath.matrix([[-mpmath.cos(theta[i])+1, -mpmath.exp(length_list[i]+x[i])*mpmath.sin(theta[i])], [mpmath.sin(theta[i]), mpmath.exp(length_list[i]+x[i])*(-mpmath.cos(theta[i])+1)]])       
            C=M2*C
            #print(M2*M1)
        # Closure condition: the resulting matrix should be Indentity matrix in PSL(2,R)
        eqcons2=[]
        eqcons2.append(C[0,0]/C[1,1]-1)
        eqcons2.append(C[0,1]/C[1,1])
        eqcons2.append(C[1,0]/C[1,1])
        #c=(M1[0][0]*M1[1][1]-M1[0][1]*M1[1][0])**(1/2)
        return eqcons2  
    initial_guess=initial
    solution=mpmath.findroot(equations2, initial_guess, tol=tolerance,maxsteps=max_steps)
    if isinstance(solution, mpmath.matrix):
        solution = [solution[i] for i in range(solution.rows * solution.cols)]
    else:
        solution = solution

    length_systole_list= [mpmath.mpf(0) for _ in range(int(4*m4))]
    length_edge_list=[mpmath.mpf(0) for _ in range(int(2*m4))]
    length_edge_list[all_direction[0]]=length_list[Pairs[all_direction[0]][0]]+l0
    length_edge_list[all_direction[1]]=length_list[Pairs[all_direction[1]][0]]+l1
    length_edge_list[all_direction[2]]=length_list[Pairs[all_direction[2]][0]]+l2
    length_edge_list[direction[0]]=length_list[Pairs[direction[0]][0]]+solution[0]
    length_edge_list[direction[1]]=length_list[Pairs[direction[1]][0]]+solution[1]
    length_edge_list[direction[2]]=length_list[Pairs[direction[2]][0]]+solution[2]
    for i in range(4):
        v=color_pairs[i]
        for k in v:
            length_systole_list[i]=length_systole_list[i]+length_edge_list[k]
    
    new_length_list1=[mpmath.mpf(0) for _ in range(int(4*m4))]
    for i in range(2*m4):
        new_length_list1[Pairs[i][0]]=length_edge_list[i]
        new_length_list1[Pairs[i][1]]=length_edge_list[i]
    

    Epsilen3=1e-70
    #k=0
    
    coordinate_list=[]
    coordinate_list.append(1j)
    reverse_length_list=list(reversed(new_length_list1))
   # list(reversed(new_length_list1[k,:]))
    theta1 = [ angle_list[Angles[i][0]] for i in range(m4)] 
    theta1[0]=theta1[0]+the0
    theta1[1]=theta1[1]+the1
    theta1[2]=theta1[2]+the2
    theta=[angle_list[i] for i in range(4*m4)] 
    for i in range(m4):
        theta[Angles[i][0]]=theta1[i]
        theta[Angles[i][1]]=mpmath.pi-theta1[i]
        theta[Angles[i][2]]=theta1[i]
        theta[Angles[i][3]]=mpmath.pi-theta1[i]
    #theta[1]=mpmath.pi-theta[1]
#theta[2]=mpmath.pi-theta[2]
    M2=mpmath.matrix([[mpmath.cos(mpmath.pi-theta[4*m4-1])+1, -mpmath.exp(reverse_length_list[0])*mpmath.sin(mpmath.pi-theta[4*m4-1])], [mpmath.sin(mpmath.pi-theta[4*m4-1]), mpmath.exp(reverse_length_list[0])*(mpmath.cos(mpmath.pi-theta[4*m4-1])+1)]])
    co1=(M2[0,0]*1j+M2[0,1])/(M2[1,0]*1j+M2[1,1])
    p1=mpmath.mpc(1j)
    p2=co1
    coordinate_list.append(p2)
    for j in range(len(length_list)-2):
        if j<len(length_list)-3:
            xc=get_geodesic_from_two_points(p1,p2)[0]
           
            rr=get_geodesic_from_two_points(p1,p2)[1]
            #[x1,r1]=from_endpoint_to_next_geodesic(xc,r2,p2,theta[4*m4-1-i])
            #print(from_one_coordinate_to_the_next(xc,rr,p2,l*reverse_length_list[i+1]))
            p3=from_endpoint_to_next(xc,rr,p2,theta[4*m4-j-2],reverse_length_list[j+1])
            
            #p3=from_geodesic_to_next_point(x1,r1,p2,reverse_length_list[i+1],1)
            coordinate_list.append(p3)
            p1=p2
            p2=p3
        if j==len(length_list)-3:
            xc=get_geodesic_from_two_points(p1,p2)[0]
            rr=get_geodesic_from_two_points(p1,p2)[1]
            #[x1,r1]=from_endpoint_to_next_geodesic(xc,r2,p2,theta[i])
            #print(from_one_coordinate_to_the_next(xc,rr,p2,l*reverse_length_list[i+1]))
            #p3=from_geodesic_to_next_point(x1,r1,p2,reverse_length_list[i+1],2)
            p3=from_endpoint_to_next(xc,rr,p2,theta[4*m4-j-2],reverse_length_list[j+1])
            #p3=-1e-20+mpmath.exp(reverse_length_list[j+2])*1j
            coordinate_list.append(p3)
            p1=p2
            p2=p3
    coordinate_list1=[]
    coordinate_list1.append(mpmath.mpc(1j))
    for j in range(len(length_list)-1):
        coordinate_list1.append(coordinate_list[len(length_list)-j-1])
    

    for i in range(len(L3)):
        n=int(len(L3[i])/2)
        if n==1:
            def distance_between_endpoints(x):
                [c0,r0]=get_geodesic_from_two_points(coordinate_list1[L3[i][0]],coordinate_list1[L3[i][0]+1])
                [c1,r1]=get_geodesic_from_two_points(coordinate_list1[L3[i][1]],coordinate_list1[(L3[i][1]+1)%12])
                
                p0=from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],x)

                #p0=-1e-20+mpmath.exp(x)*1j
                p1=from_endpoint_to_next(c1,r1,coordinate_list1[L3[i][1]],theta[L3[i][1]-1],new_length_list1[(L3[i][1]-1+12)%12]-x)
                '''if L3[i][0]==1:
                    p0=1e-20+mpmath.exp(x)*1j
                    p1=from_endpoint_to_next(c1,r1,coordinate_list1[L3[i][1]],theta[L3[i][1]-1],x)'''
                if p0.real>0:
                    xx=p0.real
                    p5=-xx+p0.imag*1j
                    p0=p5
                return distance_between_points(p0,p1)
            def derivative(x):
                return (distance_between_endpoints(x+Epsilen3)-distance_between_endpoints(x))/Epsilen3
            initial_guess=[a]
            solution=mpmath.findroot(derivative, initial_guess, tol=tolerance,maxsteps=max_steps)
            if isinstance(solution, mpmath.matrix):
                solution = [solution[i] for i in range(solution.rows * solution.cols)]
            [c0,r0]=get_geodesic_from_two_points(coordinate_list1[L3[i][0]],coordinate_list1[L3[i][0]+1])
            [c1,r1]=get_geodesic_from_two_points(coordinate_list1[L3[i][1]],coordinate_list1[(L3[i][1]+1)%12])
            if i==0:
                pp2=from_endpoint_to_next(c1,r1,coordinate_list1[L3[i][1]],theta[L3[i][1]-1],new_length_list1[(L3[i][1]-1+12)%12]-solution)
                pp4=from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],solution)
                #print("p2",from_endpoint_to_next(c1,r1,coordinate_list1[L3[i][1]],theta[L3[i][1]-1],new_length_list1[(L3[i][1]-1+12)%12]-solution))
                #print("p4",from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],solution))
            if i==1:
                pp1=from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],solution)
                pp3=from_endpoint_to_next(c1,r1,coordinate_list1[L3[i][1]],theta[L3[i][1]-1],new_length_list1[(L3[i][1]-1+12)%12]-solution)
                #print("p1",from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],solution))
                #print("p3",from_endpoint_to_next(c1,r1,coordinate_list1[L3[i][1]],theta[L3[i][1]-1],new_length_list1[(L3[i][1]-1+12)%12]-solution))
            length_systole_list[4+i]=distance_between_endpoints(solution)
        if n==2:
            for kk in range(4):
                L3[2][kk]=LL4[2][kk]
            def distance_between_endpoints(x,y):
                [c0,r0]=get_geodesic_from_two_points(coordinate_list1[L3[i][0]],coordinate_list1[(L3[i][0]+1)%12])
                [c1,r1]=get_geodesic_from_two_points(coordinate_list1[L3[i][1]],coordinate_list1[(L3[i][1]+1)%12])
                [c2,r2]=get_geodesic_from_two_points(coordinate_list1[L3[i][2]],coordinate_list1[(L3[i][2]+1)%12])
                [c3,r3]=get_geodesic_from_two_points(coordinate_list1[L3[i][3]],coordinate_list1[(L3[i][3]+1)%12])
                
                p0=from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],x)
                #p0=from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],x)
                p1=from_endpoint_to_next(c1,r1,coordinate_list1[L3[i][1]],theta[L3[i][1]-1],y)
                p2=from_endpoint_to_next(c2,r2,coordinate_list1[L3[i][2]],theta[L3[i][2]-1],new_length_list1[(L3[i][2]-1+12)%12]-y)
                p3=from_endpoint_to_next(c3,r3,coordinate_list1[L3[i][3]],theta[L3[i][3]-1],new_length_list1[(L3[i][3]-1+12)%12]-x)
                if p1.real>0:
                    xx=p1.real
                    p5=-xx+p1.imag*1j
                    p1=p5
                return [distance_between_points(p0,p1),distance_between_points(p2,p3)]
            def derivative(x,y):
                return [(distance_between_endpoints(x+Epsilen3,y)[0]-distance_between_endpoints(x,y)[0])/Epsilen3+(distance_between_endpoints(x+Epsilen3,y)[1]-distance_between_endpoints(x,y)[1])/Epsilen3,(distance_between_endpoints(x,y+Epsilen3)[0]-distance_between_endpoints(x,y)[0])/Epsilen3+(distance_between_endpoints(x,y+Epsilen3)[1]-distance_between_endpoints(x,y)[1])/Epsilen3]
            initial_guess=[a,a]
            solution=mpmath.findroot(derivative, initial_guess, tol=tolerance,maxsteps=max_steps)
            if isinstance(solution, mpmath.matrix):
                solution = [solution[i] for i in range(solution.rows * solution.cols)]
            length_systole_list[4+i]=distance_between_endpoints(solution[0],solution[1])[0]+distance_between_endpoints(solution[0],solution[1])[1]
    [c1,r1]=get_geodesic_from_two_points(pp1,pp3)
    [c2,r2]=get_geodesic_from_two_points(pp2,pp4)

    pp=get_intersection_between_two_geodesics(c1,r1,c2,r2)
    
    
    class Edge:
        def __init__(self, in_systole, in_face, length,  angle):
            self.In_systole = in_systole
            self.In_face = in_face
            self.length=length
            self.angle=angle
        def __eq__(self, other):
            if not isinstance(other, Edge):
                return False
    
                # Compare In_face
            if self.In_face != other.In_face:
                return False
            if self.In_systole != other.In_systole:
                return False
                # Compare Coordinate as sets (order doesn't matter)
            return True
        def __repr__(self):
            return f"Edge(In_systole={self.In_systole}, In_face={self.In_face}, length={self.length}, angle={self.angle})"
    
      
    Edge_list=[]
    Boundary_edge_list=[]
    ''''''
    #print("Boundary_edge_list ",Boundary_edge_list)

    #print(Boundary_edge_list)
#--------------------------------------------------------------------------
    Boundary_edge_list1=[]
    Boundary_edge_list1.append(Edge(9,3,new_length_list1[0],theta[0]))
    Boundary_edge_list1.append(Edge(8,3,distance_between_points(pp4,coordinate_list1[1]),find_angle(pp4,pp,coordinate_list1[1])))
    Boundary_edge_list1.append(Edge(8,2,distance_between_points(pp4,coordinate_list1[2]),find_angle(pp4,coordinate_list1[2],pp)))
    
    Boundary_edge_list1.append(Edge(9,2,new_length_list1[2],theta[2]))
    Boundary_edge_list1.append(Edge(1,2,new_length_list1[3],theta[3]))
    Boundary_edge_list1.append(Edge(2,2,distance_between_points(pp1,coordinate_list1[4]),find_angle(pp1,pp,coordinate_list1[4])))
    Boundary_edge_list1.append(Edge(2,1,distance_between_points(pp1,coordinate_list1[5]),find_angle(pp1,coordinate_list1[5],pp)))
    
    Boundary_edge_list1.append(Edge(1,1,new_length_list1[5],theta[5]))
    Boundary_edge_list1.append(Edge(9,1,new_length_list1[6],theta[6]))
    Boundary_edge_list1.append(Edge(8,1,distance_between_points(pp2,coordinate_list1[7]),find_angle(pp2,pp,coordinate_list1[7])))
    Boundary_edge_list1.append(Edge(8,4,distance_between_points(pp2,coordinate_list1[8]),find_angle(pp2,coordinate_list1[8],pp)))
    
    Boundary_edge_list1.append(Edge(9,4,new_length_list1[8],theta[8]))
    Boundary_edge_list1.append(Edge(1,4,new_length_list1[9],theta[9]))
    Boundary_edge_list1.append(Edge(2,4,distance_between_points(pp3,coordinate_list1[10]),find_angle(pp3,pp,coordinate_list1[10])))
    Boundary_edge_list1.append(Edge(2,3,distance_between_points(pp3,coordinate_list1[11]),find_angle(pp3,coordinate_list1[11],pp)))
    Boundary_edge_list1.append(Edge(1,3,new_length_list1[11],theta[11]))
    #print("Boundary_edge_list1 ",Boundary_edge_list1)
    
    def get_length_and_angle_list(LL,FF,index):
        [c1,r1]=get_geodesic_from_two_points(pp1,pp3)
        [c2,r2]=get_geodesic_from_two_points(pp2,pp4)
    
        pp=get_intersection_between_two_geodesics(c1,r1,c2,r2)

        if index==1:
            Boundary_edge_list=[]
            Boundary_edge_list.append(Edge(9,3,new_length_list1[0],theta[11]))
            Boundary_edge_list.append(Edge(9,2,new_length_list1[2],theta[1]))
            Boundary_edge_list.append(Edge(1,2,new_length_list1[3],theta[2]))
            Boundary_edge_list.append(Edge(1,1,new_length_list1[5],theta[4]))
            Boundary_edge_list.append(Edge(9,1,new_length_list1[6],theta[5]))
            Boundary_edge_list.append(Edge(9,4,new_length_list1[8],theta[7]))
            Boundary_edge_list.append(Edge(1,4,new_length_list1[9],theta[8]))
            Boundary_edge_list.append(Edge(1,3,new_length_list1[11],theta[10]))
        
            Boundary_edge_list.append(Edge(8,3,distance_between_points(pp4,coordinate_list1[1]),find_angle(coordinate_list1[1],pp4,coordinate_list1[0])))
            Boundary_edge_list.append(Edge(8,2,distance_between_points(pp4,coordinate_list1[2]),find_angle(pp4,coordinate_list1[2],pp)))
            
            Boundary_edge_list.append(Edge(2,2,distance_between_points(pp1,coordinate_list1[4]),find_angle(coordinate_list1[4],pp1,coordinate_list1[3])))
            Boundary_edge_list.append(Edge(2,1,distance_between_points(pp1,coordinate_list1[5]),find_angle(pp1,coordinate_list1[5],pp)))
            
            Boundary_edge_list.append(Edge(8,1,distance_between_points(pp2,coordinate_list1[7]),find_angle(coordinate_list1[7],pp2,coordinate_list1[6])))
            Boundary_edge_list.append(Edge(8,4,distance_between_points(pp2,coordinate_list1[8]),find_angle(pp2,coordinate_list1[8],pp)))
            
            Boundary_edge_list.append(Edge(2,4,distance_between_points(pp3,coordinate_list1[10]),find_angle(coordinate_list1[10],pp3,coordinate_list1[9])))
            Boundary_edge_list.append(Edge(2,3,distance_between_points(pp3,coordinate_list1[11]),find_angle(pp3,coordinate_list1[11],pp)))
            
            Boundary_edge_list.append(Edge(7,1,distance_between_points(pp2,pp),find_angle(pp2,pp,coordinate_list1[7])))
            Boundary_edge_list.append(Edge(7,2,distance_between_points(pp,pp4),find_angle(pp,pp4,pp1)))
            Boundary_edge_list.append(Edge(7,3,distance_between_points(pp,pp4),find_angle(pp4,pp,coordinate_list1[1])))
            Boundary_edge_list.append(Edge(7,4,distance_between_points(pp2,pp),find_angle(pp,pp2,pp3)))
            
            Boundary_edge_list.append(Edge(11,1,distance_between_points(pp1,pp),find_angle(pp,pp1,pp2)))
            Boundary_edge_list.append(Edge(11,2,distance_between_points(pp1,pp),find_angle(pp1,pp,coordinate_list1[4])))
            Boundary_edge_list.append(Edge(11,3,distance_between_points(pp3,pp),find_angle(pp,pp3,pp4)))
            Boundary_edge_list.append(Edge(11,4,distance_between_points(pp3,pp),find_angle(pp3,pp,coordinate_list1[10])))
        if index==0:
            Boundary_edge_list=[]
            Boundary_edge_list.append(Edge(9,3,new_length_list1[0],theta[0]))
            Boundary_edge_list.append(Edge(9,2,new_length_list1[2],theta[2]))
            Boundary_edge_list.append(Edge(1,2,new_length_list1[3],theta[3]))
            Boundary_edge_list.append(Edge(1,1,new_length_list1[5],theta[5]))
            Boundary_edge_list.append(Edge(9,1,new_length_list1[6],theta[6]))
            Boundary_edge_list.append(Edge(9,4,new_length_list1[8],theta[8]))
            Boundary_edge_list.append(Edge(1,4,new_length_list1[9],theta[9]))
            Boundary_edge_list.append(Edge(1,3,new_length_list1[11],theta[11]))
        
            Boundary_edge_list.append(Edge(8,3,distance_between_points(pp4,coordinate_list1[1]),find_angle(pp4,pp,coordinate_list1[1])))
            Boundary_edge_list.append(Edge(8,2,distance_between_points(pp4,coordinate_list1[2]),find_angle(coordinate_list1[2],coordinate_list1[3],pp4)))
            
            Boundary_edge_list.append(Edge(2,2,distance_between_points(pp1,coordinate_list1[4]),find_angle(pp1,pp,coordinate_list1[4])))
            Boundary_edge_list.append(Edge(2,1,distance_between_points(pp1,coordinate_list1[5]),find_angle(coordinate_list1[5],coordinate_list1[6],pp1)))
            
            Boundary_edge_list.append(Edge(8,1,distance_between_points(pp2,coordinate_list1[7]),find_angle(pp2,pp,coordinate_list1[7])))
            Boundary_edge_list.append(Edge(8,4,distance_between_points(pp2,coordinate_list1[8]),find_angle(coordinate_list1[8],coordinate_list1[9],pp2)))
            
            Boundary_edge_list.append(Edge(2,4,distance_between_points(pp3,coordinate_list1[10]),find_angle(pp3,pp,coordinate_list1[10])))
            Boundary_edge_list.append(Edge(2,3,distance_between_points(pp3,coordinate_list1[11]),find_angle(coordinate_list1[11],coordinate_list1[0],pp3)))
            
            Boundary_edge_list.append(Edge(7,1,distance_between_points(pp2,pp),find_angle(pp,pp1,pp2)))
            Boundary_edge_list.append(Edge(7,4,distance_between_points(pp2,pp),find_angle(pp2,coordinate_list1[8],pp)))
            Boundary_edge_list.append(Edge(7,2,distance_between_points(pp4,pp),find_angle(pp4,coordinate_list1[2],pp)))
            Boundary_edge_list.append(Edge(7,3,distance_between_points(pp4,pp),find_angle(pp,pp3,pp4)))
            
            Boundary_edge_list.append(Edge(11,1,distance_between_points(pp1,pp),find_angle(pp1,coordinate_list1[5],pp)))
            Boundary_edge_list.append(Edge(11,2,distance_between_points(pp1,pp),find_angle(pp,pp4,pp1)))
            Boundary_edge_list.append(Edge(11,3,distance_between_points(pp3,pp),find_angle(pp3,coordinate_list1[11],pp)))
            Boundary_edge_list.append(Edge(11,4,distance_between_points(pp3,pp),find_angle(pp,pp2,pp3)))
        new_Boundary_edge_list1=[]
        for i in range(len(Boundary_edge_list1)):
            this_edge=Boundary_edge_list1[i]
            p=this_edge.In_systole
            q=this_edge.In_face
            for j in range(len(LL)):
                if p==LL[j][0]:
                    pp=LL[j][1]
                    break
            for j in range(len(FF)):
                if q==FF[j][0]:
                    qq=FF[j][1]
                    break
            new__edge=Edge(pp,qq,0,0)
            for kk in range(len(Boundary_edge_list)):
                if Boundary_edge_list[kk]==new__edge:
                    new_Boundary_edge_list1.append(Boundary_edge_list[kk])
                    break
        new_length_list=[]
        new_angle_list=[]
        position=0
        while position <= 15:
            edge1= new_Boundary_edge_list1[position]
            if position==15:
                new_length_list.append(edge1.length)
                new_angle_list.append(edge1.angle)
                break
            edge2=new_Boundary_edge_list1[position+1]
            if edge2.In_systole!=edge1.In_systole:
                new_length_list.append(edge1.length)
                new_angle_list.append(edge1.angle)
                position=position+1
            else:
                new_length_list.append(edge1.length+edge2.length)
                new_angle_list.append(edge2.angle)
                position=position+2
        if index==0:
            return [new_length_list,new_angle_list]
        
        if index==1:
            new_angle_list1=[]
            for kk in range(12):
                #new_angle_list1.append(mpmath.pi-new_angle_list[(kk)%12])
                new_angle_list1.append(new_angle_list[(kk)%12])
            #new_angle_list1=list(reversed(new_angle_list1))
            #new_length_list=list(reversed(new_length_list))
            #new_length_list1.append(new_length_list[0])
            '''for kk in range(11):
                new_length_list1.append(new_length_list[kk])'''
            #print([new_length_list,new_angle_list1])
            return [new_length_list,new_angle_list1]
    #LL=[[1,7],[2,11],[8,9],[9,8],[11,2],[7,1]]
    #FF=[[1,3],[2,4],[3,1],[4,2]]
    #index=1
    #print("----------------")
    #print(get_length_and_angle_list(LL,FF,index))

    def get_extra_length(AA,index):
        new_length_list=AA[0]
        new_angle_list =AA[1]
        
        theta=new_angle_list.copy()
        new_length_list1=new_length_list.copy()
        coordinate_list=[]
        coordinate_list.append(1j)
        reverse_length_list=list(reversed(new_length_list1))
       # list(reversed(new_length_list1[k,:]))
        M2=mpmath.matrix([[mpmath.cos(mpmath.pi-theta[4*m4-1])+1, -mpmath.exp(reverse_length_list[0])*mpmath.sin(mpmath.pi-theta[4*m4-1])], [mpmath.sin(mpmath.pi-theta[4*m4-1]), mpmath.exp(reverse_length_list[0])*(mpmath.cos(mpmath.pi-theta[4*m4-1])+1)]])
        co1=(M2[0,0]*1j+M2[0,1])/(M2[1,0]*1j+M2[1,1])
        p1=mpmath.mpc(1j)
        p2=co1
        coordinate_list.append(p2)
        for j in range(len(length_list)-2):
            if j<len(length_list)-3:
                xc=get_geodesic_from_two_points(p1,p2)[0]
                rr=get_geodesic_from_two_points(p1,p2)[1]
                #[x1,r1]=from_endpoint_to_next_geodesic(xc,r2,p2,theta[4*m4-1-i])
                #print(from_one_coordinate_to_the_next(xc,rr,p2,l*reverse_length_list[i+1]))
                p3=from_endpoint_to_next(xc,rr,p2,theta[4*m4-j-2],reverse_length_list[j+1])
                
                #p3=from_geodesic_to_next_point(x1,r1,p2,reverse_length_list[i+1],1)
                coordinate_list.append(p3)
                p1=p2
                p2=p3
            if j==len(length_list)-3:
                xc=get_geodesic_from_two_points(p1,p2)[0]
                rr=get_geodesic_from_two_points(p1,p2)[1]
                #[x1,r1]=from_endpoint_to_next_geodesic(xc,r2,p2,theta[i])
                #print(from_one_coordinate_to_the_next(xc,rr,p2,l*reverse_length_list[i+1]))
                #p3=from_geodesic_to_next_point(x1,r1,p2,reverse_length_list[i+1],2)
                p3=from_endpoint_to_next(xc,rr,p2,theta[4*m4-j-2],reverse_length_list[j+1])
                #p3=-1e-20+mpmath.exp(reverse_length_list[j+2])*1j
                coordinate_list.append(p3)
                p1=p2
                p2=p3
        coordinate_list1=[]
        coordinate_list1.append(mpmath.mpc(1j))
        for j in range(len(length_list)-1):
            coordinate_list1.append(coordinate_list[len(length_list)-j-1])
        #print("coordinate_list1",coordinate_list1)
        i=2

        def distance_between_endpoints(x,y):
            L6=[0,0,0,0]
            #L6=[5,8,2,11]
            #L6=[2,5,11,8]
            if index==1:
                for kk in range(4):
                    L6[kk]=L3[i][kk]
            
            if index==0:
                for kk in range(4):
                    L6[kk]=L3[i][kk]
            [c0,r0]=get_geodesic_from_two_points(coordinate_list1[L6[0]],coordinate_list1[(L6[0]+1)%12])
            [c1,r1]=get_geodesic_from_two_points(coordinate_list1[L6[1]],coordinate_list1[(L6[1]+1)%12])
            [c2,r2]=get_geodesic_from_two_points(coordinate_list1[L6[2]],coordinate_list1[(L6[2]+1)%12])
            [c3,r3]=get_geodesic_from_two_points(coordinate_list1[L6[3]],coordinate_list1[(L6[3]+1)%12])
            
            p0=from_endpoint_to_next(c0,r0,coordinate_list1[L6[0]],theta[L6[0]-1],x)
            #p0=from_endpoint_to_next(c0,r0,coordinate_list1[L3[i][0]],theta[L3[i][0]-1],x)
            p1=from_endpoint_to_next(c1,r1,coordinate_list1[L6[1]],theta[L6[1]-1],y)
            p2=from_endpoint_to_next(c2,r2,coordinate_list1[L6[2]],theta[L6[2]-1],new_length_list1[(L6[2]-1+12)%12]-y)
            p3=from_endpoint_to_next(c3,r3,coordinate_list1[L6[3]],theta[L6[3]-1],new_length_list1[(L6[3]-1+12)%12]-x)
            
            return [distance_between_points(p0,p1),distance_between_points(p2,p3)]
        def derivative(x,y):
            return [(distance_between_endpoints(x+Epsilen3,y)[0]-distance_between_endpoints(x,y)[0])/Epsilen3+(distance_between_endpoints(x+Epsilen3,y)[1]-distance_between_endpoints(x,y)[1])/Epsilen3,(distance_between_endpoints(x,y+Epsilen3)[0]-distance_between_endpoints(x,y)[0])/Epsilen3+(distance_between_endpoints(x,y+Epsilen3)[1]-distance_between_endpoints(x,y)[1])/Epsilen3]
        initial_guess=[a,a]
        solution=mpmath.findroot(derivative, initial_guess, tol=tolerance,maxsteps=max_steps)
        if isinstance(solution, mpmath.matrix):
            solution = [solution[i] for i in range(solution.rows * solution.cols)]
        return distance_between_endpoints(solution[0],solution[1])[0]+distance_between_endpoints(solution[0],solution[1])[1]
    
    LL=[[1,7],[2,8],[8,2],[9,11],[11,9],[7,1]]
    FF=[[1,2],[2,3],[3,4],[4,1]]
    index=1
    length_systole_list[7]=get_extra_length(get_length_and_angle_list(LL,FF,index),index)

    LL=[[1,9],[2,1],[8,7],[9,8],[11,2],[7,11]]
    FF=[[1,2],[2,3],[3,4],[4,1]]
    #FF=[[1,3],[2,1],[3,2],[4,4]]
    index=1
    length_systole_list[8]=get_extra_length(get_length_and_angle_list(LL,FF,index),index)

    LL=[[1,7],[2,11],[8,9],[9,8],[11,2],[7,1]]
    FF=[[1,3],[2,2],[3,1],[4,4]]
    index=1
    length_systole_list[9]=get_extra_length(get_length_and_angle_list(LL,FF,index),index)

    
    LL=[[1,2],[2,1],[8,7],[9,11],[11,9],[7,8]]
    FF=[[1,1],[2,4],[3,3],[4,2]]
    index=1
    length_systole_list[10]=get_extra_length(get_length_and_angle_list(LL,FF,index),index)
    
    LL=[[1,1],[2,9],[8,11],[9,2],[11,8],[7,7]]
    FF=[[1,2],[2,1],[3,4],[4,3]]
    index=0
    length_systole_list[11]=get_extra_length(get_length_and_angle_list(LL,FF,index),index)
    length_systole_list_ordered=[0 for i in range(12)]
    length_systole_list_ordered[0]=length_systole_list[3]    
    length_systole_list_ordered[1]=length_systole_list[2]
    length_systole_list_ordered[2]=length_systole_list[8]
    length_systole_list_ordered[3]=length_systole_list[7]
    length_systole_list_ordered[4]=length_systole_list[10]
    length_systole_list_ordered[5]=length_systole_list[6]
    length_systole_list_ordered[6]=length_systole_list[4]
    length_systole_list_ordered[7]=length_systole_list[1]
    length_systole_list_ordered[8]=length_systole_list[0]
    length_systole_list_ordered[9]=length_systole_list[9]
    length_systole_list_ordered[10]=length_systole_list[5]
    length_systole_list_ordered[11]=length_systole_list[11]
    # 9 8 2 1 11 7 6 4 3 10 5 12
    #L_A_C_x= length_systole_list_ordered[11]#+length_systole_list_ordered[1]+length_systole_list_ordered[6]+length_systole_list_ordered[7]+length_systole_list_ordered[8]
    #print(length_systole_list)
    return  length_systole_list_ordered
#==========================================================

def length_function(x,initial,coef):
    # coef is a list with 12 entries
    l=0
    all_length=length_of_curves(x,initial)
    for i in range(12):
        l=l+coef[i]*all_length[i]
    return l


#==========================================================

def find_initial(x,initial):
    l0=x[0]
    l1=x[1]
    l2=x[2]
    the0=x[3]
    the1=x[4]
    the2=x[5]    
    m4=3
    mpmath.mp.dps = 300
    
    a=mpmath.acosh(mpmath.csc(mpmath.pi/8)/2)
    L=2*a
    Angles=[[10,9,4,3],[11,8,5,2],[0,7,6,1]]
    tolerance=1e-20
    Epsilen4=1e-20  
    max_steps=10000
    length_list=[1,2,1,1,2,1,1,2,1,1,2,1]
    for i in range(len(length_list)):
        length_list[i]=length_list[i]*L
    angle_list=[3,1,3,1,3,1,3,1,3,1,3,1]
    for i in range(len(angle_list)):
        angle_list[i]=angle_list[i]*mpmath.pi/4

    Pairs=[[6,8],[0,2],[1,7],[10,4],[3,5],[11,9]]
    color_pairs=[[0,1],[2],[3],[4,5]]
    direction=[1,2,4]
    all_direction=[]
    for i in range(2*m4):
        if i not in direction:
            all_direction.append(i)
    M1=mpmath.matrix([[1, 0], [0, 1]])
    C=M1
    for i in range(4*m4):
        M2=mpmath.matrix([[-mpmath.cos(angle_list[i])+1, -mpmath.exp(length_list[i])*mpmath.sin(angle_list[i])], [mpmath.sin(angle_list[i]), mpmath.exp(length_list[i])*(-mpmath.cos(angle_list[i])+1)]])
        C=M2*C


    def equations2(y0,y1,y2):
        theta1 = [ angle_list[Angles[i][0]] for i in range(m4)] 
        theta1[0]=theta1[0]+the0
        theta1[1]=theta1[1]+the1
        theta1[2]=theta1[2]+the2
        theta=[angle_list[i] for i in range(4*m4)] 
        for i in range(m4):
            theta[Angles[i][0]]=theta1[i]
            theta[Angles[i][1]]=mpmath.pi-theta1[i]
            theta[Angles[i][2]]=theta1[i]
            theta[Angles[i][3]]=mpmath.pi-theta1[i]
        z = [mpmath.mpf(0) for _ in range(m4*2)]  
        z[direction[0]]=y0
        z[direction[1]]=y1
        z[direction[2]]=y2
        z[all_direction[0]]=l0
        z[all_direction[1]]=l1
        z[all_direction[2]]=l2
        x=[mpmath.mpf(0) for _ in range(m4*4)]  
        for i in range(m4*2):
            x[Pairs[i][0]]=z[i]
            x[Pairs[i][1]]=z[i]
        C=M1
        for i in range(4*m4):
        #M2=(1/(2*x[i]))**(1/2)*np.array([[1, -x[i]], [1, x[i] ]])
            M2=mpmath.matrix([[-mpmath.cos(theta[i])+1, -mpmath.exp(length_list[i]+x[i])*mpmath.sin(theta[i])], [mpmath.sin(theta[i]), mpmath.exp(length_list[i]+x[i])*(-mpmath.cos(theta[i])+1)]])       
            C=M2*C
            #print(M2*M1)
        eqcons2=[]
        eqcons2.append(C[0,0]/C[1,1]-1)
        eqcons2.append(C[0,1]/C[1,1])
        eqcons2.append(C[1,0]/C[1,1])
        return eqcons2  
    initial_guess=initial
    solution=mpmath.findroot(equations2, initial_guess, tol=tolerance,maxsteps=max_steps)
    if isinstance(solution, mpmath.matrix):
        solution = [solution[i] for i in range(solution.rows * solution.cols)]
    return [solution[0],solution[1],solution[2]]
#==========================================================

def gradient_descent_convex(
    func,
    coef,
    initial_params,      
    initial,
    learning_rate,       
    max_iter=10000,     
    tol=1e-6,            
    h=1e-10               
):
    #initial=[3.801742890806985e-05, -1.528568965361195, -0.00011738469426353445]
    theta = initial_params 
    for iter_idx in range(max_iter):
        grad=[0,0,0,0,0,0]
        for j in range(len(theta)):  
            theta_plus = theta.copy()
            theta_plus[j] += h
            theta_minus = theta.copy()
            theta_minus[j] -= h
            #print("theta_plus",theta_plus)
            grad[j] = (func(theta_plus,initial,coef) - func(theta_minus,initial,coef)) / ( 2*h)
        
        if np.linalg.norm(grad) < tol:
            print("grad",grad)
            print(f"迭代 {iter_idx + 1} 次后收敛")
            break
        if iter_idx%10==0:

            theta1=[float(a) for a in theta]
            print("theta",theta1)
            grad1=[float(a) for a in grad]
            print("grad",grad1)
        for i in range(6):
            theta[i] = theta[i]-learning_rate *(grad[i])#*(abs(grad[i]))**(1/10)
        #theta -= learning_rate * grad
        #print("theta",theta)
        #print("initial",initial)
        initial=find_initial(theta,initial)
        if iter_idx%10==0:
            initial1=[float(a) for a in initial]
            print("new_initial",initial1)
            print("function_value",func(theta,initial,coef))
            print("-----------------")

    else:
        print(f"Reached maximum iterations ({max_iter}) without full convergence.")

    return theta, func(theta,initial)
#initial_guess=[-0.0093442780588633055957783696938607821351033331236947102414379771498082925862554807670879100633621816365967354781207921596877095237564814320653917022822709649787801525203741429840593639743854779524997789585,-0.0095492566856296960214801247884165012086965768239548113320664126497403883337887596735656163046565891143889795969840058961190749173944576515588569700017095237274447989247009018585860847267989453081812099661,-0.016713225918614614369210419438574806972232926085929786836386133556721844291427348966087975185577005951671649425266360852916442991153500994441019817952326287512208664011223796603549031662612295408304722878,-0.021111472131322237242738177464128458268586403846261383583843653078741025256600732397764283467929928347768780504797541356205105335631730948179917610035036488204273598648023717733074705803671971678243094907,-0.044828487765859529794436470089375920617135615556384434761620919841490871810742115545587578168977217393984267904681692856691117727634468692347622097171819141415313196728759020204323999720500980712632547253,-0.035706482319290778018912390017493000425518771372554099077763914488787907569654115308827001538164702552494280341761774329709092428623510005339132237749858301996829249254989229371653413978701436007124178873]    
initial_guess= [3.803467686750663e-05, 0.0003452656333614451, -0.00011743792417637554, -0.7853977339705183, 0.7853973897617561, -0.7853971988389467]
learning_rate = 0.003
min_params, min_value = gradient_descent_convex(
    func=length_function,
    coef=[1,1,1,1,1,1,1,1,0,1,1,1],
    initial_params=initial_guess,
    initial=[3.801742890806985e-05, -1.528568965361195, -0.00011738469426353445],    
    learning_rate=learning_rate,
    max_iter=10000,
    tol=1e-15
)
#==========================================================

def automorphism_group_quotient_hyperelliptic_involution(critical_point,orientation):
    def get_group_from_generators(generators, set_of_systoles):
        group=[]
        for i in range(len(generators)):
            for j in range(critical_point):
                generators[i][j]=generators[i][j]-1
        for i in range(len(generators)):
            for j in range(len(generators)):
                for k in range(len(generators)):
                    L=[]
                    for k1 in range(critical_point):
                        L.append(set_of_systoles[generators[k][generators[j][generators[i][k1]]]])            
                    if L not in group:
                        group.append(L)                    
        return group
    automorphism_group=[]
    if critical_point==12 and orientation==0:
        generators=[[1,2,3,4,5,6,7,8,9,10,11,12],
                  [2,3,4,1,6,7,8,5,10,11,12,9],
                  [3,4,1,2,7,8,5,6,11,12,9,10],
                  [4,1,2,3,8,5,6,7,12,9,10,11],
                  [1,4,3,2,5,8,7,6,10,9,12,11],
                  [3,2,1,4,7,6,5,8,12,11,10,9],
                  [5,6,7,8,1,2,3,4,9,10,11,12],    
                  [2,1,4,3,6,5,8,7,11,10,9,12],
                  [4,3,2,1,8,7,6,5,9,12,11,10],
                  [1,10,5,9,3,11,7,12,4,2,6,8]
                 ]
        return get_group_from_generators(generators,[1,2,3,4,5,6,7,8,9,10,11,12])
    if critical_point==12 and orientation==1:
        generators=[[1,2,3,4,5,6,7,8,9,10,11,12],
                  [2,3,4,1,6,7,8,5,10,11,12,9],
                  [1,9,5,10,3,12,7,11,2,4,8,6],
                  [2,10,6,11,4,9,8,12,3,1,5,7],
                  [3,11,7,12,1,10,5,9,4,2,6,8]
                 ]
        return get_group_from_generators(generators,[1,2,3,4,5,6,7,8,9,10,11,12])
    if critical_point==9 and orientation==0:
        generators=[[1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [1, 7, 4, 3, 9, 8, 2, 6, 5], 
                 [5, 4, 6, 2, 1, 3, 8, 7, 9], 
                 [9, 6, 7, 8, 5, 2, 3, 4, 1],
                 [1, 4, 7, 2, 5, 8, 3, 6, 9]
             ]
        return get_group_from_generators(generators,[1,2,3,5,6,8,9,11,12])
    if critical_point==9 and orientation==1:
        generators=[[1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [1, 7, 4, 3, 9, 8, 2, 6, 5], 
                 [5, 4, 6, 2, 1, 3, 8, 7, 9]
             ]
        return get_group_from_generators(generators,[1,2,3,5,6,8,9,11,12])
    if critical_point==6 and orientation==0:
        generators=[[1, 2, 3, 4, 5, 6],
                    [2,3,4,5,6,1],
                    [4,5,6,1,2,3],
                    [6,5,4,3,2,1]
             ]
        return get_group_from_generators(generators,[1,2,11,7,8,9])
    if critical_point==6 and orientation==1:
        generators=[[1, 2, 3, 4, 5, 6],
                    [2,3,4,5,6,1],
                    [4,5,6,1,2,3]
             ]
        return get_group_from_generators(generators,[1,2,11,7,8,9])
    if critical_point==5 and orientation==0:
        generators=[[1, 2, 3, 4, 5],
                    [2,3,4,5,1],
                    [4,5,1,2,3],
                    [4,3,2,1,5]
             ]
        return get_group_from_generators(generators,[1,2,11,7,5])
    if critical_point==5 and orientation==1:
        generators=[[1, 2, 3, 4, 5],
                    [2,3,4,5,1],
                    [4,5,1,2,3]
             ]
        return get_group_from_generators(generators,[1,2,11,7,5])
#==========================================================

        

from gurobipy import Model, GRB
import numpy as np
def if_in_a_minima(diff_matrix, curves_set):
    n = len(curves_set)
    m = Model("feas_qcqp")
    m.setParam("OutputFlag", 0)          # 0 为静默
    x = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    
    ee1=1
    for i in range(len( curves_set)):
        m.addConstr(x[i]<=100, name ="qc1")
        m.addConstr(x[i]>=ee1, name ="qc1")
    ee2=1e-10
    m.setParam('OptimalityTol', 1e-3)
    
    # 设置约束可行性容忍度（默认是 1e-6）
    m.setParam('FeasibilityTol', 1e-3)
    
    for i in range(6):
        a=0
        for j in range(len( curves_set)):
            a=a+x[j]*diff_matrix[i][curves_set[j]-1]
        m.addConstr(a>=-ee2, name ="qc1")   
        m.addConstr(a<=ee2, name ="qc1") 
    m.optimize()
    if m.status == GRB.OPTIMAL: 
        print("System is feasible! Feasible point=", [x[i].X for i in range(n)])
        return 1
    
    elif m.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        #print("System is feasible! ")
        return 0
   
        return 0
      
def get_second_differential(critical_point):
    if critical_point==12:
        systoles=[1,2,3,4,5,6,7,8,9,10,11,12]
        parameter=[0,0,0,0,0,0]
        initial=[0,0,0]
    if critical_point==9:
        systoles=[1,2,3,5,6,8,9,11,12]
        parameter=[]
        initial=[]
    if critical_point==5:
        systoles=[1,2,5,7,11]
        
    mm=critical_point
   
    def length_of_curves2(parameter, inintial):
        LL1=[]
        all_length=length_of_curves(parameter, initial)
        for i in range(mm):
            LL1.append(all_length[systoles[i]-1])
        return LL1
            
    MM2=[ [  # 第 0 层
        [mp.mpf(0) for j in range(6)]  # 第 i 行
        for i in range(6)
    ] for n in range(mm)]
    e1=1e-10
    e2=1e-10
    for i in range(6):
        for j in range(i+1):
            print(i,j)
            a = parameter.copy()
            b = parameter.copy()
            c = parameter.copy()
            d = parameter.copy()
            a[i]=a[i]+e1
            a[j]=a[j]+e2
            b[j]=b[j]+e2
            c[i]=c[i]+e1
            LL=[]
            LL.append(length_of_curves2(a,initial))
            LL.append(length_of_curves2(b,initial))
            LL.append(length_of_curves2(c,initial))
            LL.append(length_of_curves2(d,initial))
            LE=[]
            
            for k in range(mm):
                LE.append((1/(e1*e2))*(LL[0][k]-LL[1][k]-LL[2][k]+LL[3][k]))
            #print((length_function(a,initial)[k]-length_function(b,initial)[k])/e1)
            #print((length_function(c,initial)[k]-length_function(d,initial)[k])/e1)
            #print(a)
            #print(b)
            #print(c)
            #print(d)
            for k in range(mm):
                MM2[k][i][j]=LE[k]
                MM2[k][j][i]=LE[k]
            print(LE)
            print("---------------------------")
    return MM2

def get_third_differential(critical_point):
    if critical_point==12:
        systoles=[1,2,3,4,5,6,7,8,9,10,11,12]
        parameter=[0,0,0,0,0,0]
        initial=[0,0,0]
    if critical_point==9:
        systoles=[1,2,3,5,6,8,9,11,12]
        parameter=[]
        initial=[]
    if critical_point==5:
        systoles=[1,2,5,7,11]
        
    mm=critical_point
   
    def length_of_curves2(parameter, inintial):
        LL1=[]
        all_length=length_of_curves(parameter, initial)
        for i in range(mm):
            LL1.append(all_length[systoles[i]-1])
        return LL1
            
    MM3=[[  [  # 第 0 层
        [mp.mpf(0) for j in range(6)]  # 第 i 行
        for i in range(6)
    ] for k in range(6)]for n in range(mm)
    ]
    e1=1e-7
    e2=1e-7
    e3=1e-7
    for i in range(6):
        for j in range(i+1):
            for r in range(j+1):
            #print(i,j)
                a1 = parameter.copy()
                a2 = parameter.copy()
                a3 = parameter.copy()
                a4 = parameter.copy()
                a5 = parameter.copy()
                a6 = parameter.copy()
                a7 = parameter.copy()
                a8 = parameter.copy()
                    
                a1[i]=a1[i]+e1
                a1[j]=a1[j]+e2
                a1[r]=a1[r]+e3

                a2[i]=a2[i]+e1
                a2[j]=a2[j]+e2

                a3[i]=a3[i]+e1
                a3[r]=a3[r]+e3

                a5[j]=a5[j]+e2
                a5[r]=a5[r]+e3

                a4[i]=a4[i]+e1
                a6[j]=a6[j]+e2
                a7[r]=a7[r]+e3
                
            #print((length_function(a,initial)[k]-length_function(b,initial)[k])/e1)
            #print((length_function(c,initial)[k]-length_function(d,initial)[k])/e1)
            #print(a)
            #print(b)
            #print(c)
            #print(d)
                print(i,j,r)
                LEN=[]
                LEN.append(length_of_curves2(a1,initial))
                LEN.append(length_of_curves2(a2,initial))
                LEN.append(length_of_curves2(a3,initial))
                LEN.append(length_of_curves2(a4,initial))
                LEN.append(length_of_curves2(a5,initial))
                LEN.append(length_of_curves2(a6,initial))
                LEN.append(length_of_curves2(a7,initial))
                LEN.append(length_of_curves2(a8,initial))
                LF=[]
                for k in range(mm):
                    LF.append((1/(e1*e2*e3))*(LEN[0][k] -LEN[1][k]-LEN[2][k]+LEN[3][k]-LEN[4][k]+LEN[5][k]+LEN[6][k]-LEN[7][k]))
                print(LF)
                #D[i][j][r]=(1/(e1*e2*e3))*(length_function(a1,initial)[k] -length_function(a2,initial)[k]-length_function(a3,initial)[k]+length_function(a4,initial)[k]-length_function(a5,initial)[k]+length_function(a6,initial)[k]+length_function(a7,initial)[k]-length_function(a8,initial)[k])
                #D[i][j][r]=((length_function(a,initial)[k]-length_function(b,initial)[k])/e1-(length_function(c,initial)[k]-length_function(d,initial)[k])/e1)/e2
                for qq in range(mm):
                    MM3[qq][i][j][r]=LF[qq]
                    MM3[qq][i][r][j]=LF[qq]
                    MM3[qq][r][i][j]=LF[qq]
                    MM3[qq][r][j][i]=LF[qq]
                    MM3[qq][j][i][r]=LF[qq]
                    MM3[qq][j][r][i]=LF[qq]               
                
                #print(D[i][j][r])
                print("---------------------------")
    return MM3

#==========================================================


def if_adjacent_to_stratum(critical_point, curves_set,M1,M2,M3):
    mm=critical_point
    D=np.zeros((6,mm))
    for i in range(6):
        for j in range(mm):
            D[i][j]=M1[i][j] 
    n = 18
    if critical_point==12:
        systoles=[1,2,3,4,5,6,7,8,9,10,11,12]
    if critical_point==9:
        systoles=[1,2,3,5,6,8,9,11,12]
    if critical_point==5:
        systoles=[1,2,5,7,11]

    LL=[]
    LL1=[]
    for k in curves_set:
        for i in range(len(systoles)):
            if k==systoles[i]:
                LL.append(i)
    for i in range(len(systoles)):
        if i not in LL:
                LL1.append(i)
        # ---------- 问题维度 ----------
        
    m = Model("feas_qcqp")
    m.setParam("OutputFlag", 0)          # 0 为静默
    x = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    e = m.addVars(6, 6, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="eij")
    f = m.addVars(6, 6,6, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="fijk")
    v1=[x[i] for i in range(6)]
    v2=[x[i+6] for i in range(6)]
    v3=[x[i+12] for i in range(6)]
    # ---------- 0 目标（纯可行性） ----------
    m.setObjective(0, GRB.MINIMIZE)      # 常数目标
    for i in range(18):
        m.addConstr(x[i]>=-1, name ="qc1")
        m.addConstr(x[i]<=1, name ="qc1")
    for i in range(6):
        for j in range(6):
            m.addConstr(e[i,j]==v1[i]*v1[j], name ="qc1")

    dd3=[]
    for r in range(mm):
        bb=0
        for i in range(6):
            bb=bb+D[i][r]*v3[i]

        for i in range(6):
            for j in range(6):
                bb=bb+3*M2[r][i][j]*v1[i]*v2[j]
        for i in range(6):
            for j in range(6):
                for kk in range(6):                   
                    bb=bb+M3[r][i][j][kk]*e[i,j]*v1[kk]
        dd3.append(bb)

    dd1=[]
    for r in range(mm):
        bb=0
        for i in range(6):
            bb=bb+D[i][r]*v1[i]
        dd1.append(bb)

    dd2=[]
    for r in range(mm):
        bb=0
        for i in range(6):
            bb=bb+D[i][r]*v2[i]
        for i in range(6):
            for j in range(6):
                bb=bb+M2[r][i][j]*v1[i]*v1[j]
        dd2.append(bb)
    Length_approxi=[]
    t=0.1
    epsilen=1e-5


    
    for r in range(mm):
        a=dd1[r]*t+(1/2)*dd2[r]*t*t+(1/6)*dd3[r]*t*t*t
        Length_approxi.append(a)
    #print(np.dot(v1,D[:,LL[0]]))
    
    #m.addConstr(x[3]*2+x[1]*3<=1, name ="qc1")
    for i in range(len(LL)-1):
        #print(LL[i+1])
        #print(np.dot(v1,D[:,LL[i]]))
        m.addConstr(Length_approxi[LL[i]]-Length_approxi[LL[i+1]]<=epsilen)
        m.addConstr(Length_approxi[LL[i]]-Length_approxi[LL[i+1]]>=-epsilen)

    #---------------------------------------------
    m.addConstr(np.dot(v1,D[:,LL[0]])<=0)
    
    for i in range(len(LL1)):
        #print(LL[i+1])
        #print(np.dot(v1,D[:,LL[i]]))
        1
        #
        m.addConstr(Length_approxi[LL1[i]]>=0.0001+Length_approxi[LL[0]])
    
    
 
    for i in range(len(LL1)):
        #print(LL[i+1])
        #print(np.dot(v1,D[:,LL[i]]))
        m.addConstr(np.dot(v1,D[:,LL1[i]])>=np.dot(v1,D[:,LL[0]]))
    #------------------------------------------------------------------------

    for i in range(len(LL)-1):
        m.addConstr(dd3[LL[i]]-dd3[LL[i+1]]<=epsilen)
        m.addConstr(dd3[LL[i]]-dd3[LL[i+1]]>=-epsilen)

    for i in range(len(LL)-1):
        m.addConstr(dd2[LL[i]]-dd2[LL[i+1]]<=epsilen)
        m.addConstr(dd2[LL[i]]-dd2[LL[i+1]]>=-epsilen)

    for i in range(len(LL)-1):
        m.addConstr(dd1[LL[i]]-dd1[LL[i+1]]<=epsilen)
        m.addConstr(dd1[LL[i]]-dd1[LL[i+1]]>=-epsilen)
        
    m.optimize()
    if m.status == GRB.OPTIMAL:
        return 1
        print("The system is feasible, the feasible point is ", [x[i].X for i in range(n)])
        ''''''
        '''a=0
        for i in range(6):
            a=a+vv1[i]**2
        a=a**(1/2)
        b=0
        for i in range(6):
            b=b+vv2[i]**2
        b=b**(1/2)
        
    for i in range(6):
            vv1[i]=vv1[i]/a
    
        for i in range(6):
            vv2[i]=vv2[i]/b
            
        '''
            
        '''print("vv1",vv1)
        print("vv2",vv2)
        for i in range(12):
        #print(LL[i+1])
        #print(np.dot(v1,D[:,LL[i]]))'''
        
    elif m.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return 0
    else:
        return 0


#==========================================================



