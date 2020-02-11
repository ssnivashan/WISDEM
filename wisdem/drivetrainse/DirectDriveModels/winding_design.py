import numpy as np
import pandas as pd
from cmath import pi,exp
import math


def lcm(a,b): 
	return abs(a * b) / math.gcd(a,b) if a and b else 0

def array_seq(q1,b,c,Total_number):
	ONES =b
	ZEROS=b-c
	Seq		=np.array([1,0,0,1,0])
	diff= Total_number*5/6
	G=np.prod(Seq.shape)
	return Seq, diff,G
	


def winding_factor(S,b,c,p,m):


    #Step 1 Writing q1 as a fraction
    
    
    q1=b/c
    
    # Step 2: Writing a binary sequence of b-c zeros and b ones
    
    Total_number=int(S/b)
    
    L=array_seq(q1,b,c,Total_number)
    
    # STep 3 : Repeat binary sequence Q_s/b times
    
   
    New_seq=np.tile(L[0],Total_number)
    
    Actual_seq1=(pd.DataFrame(New_seq[:,None].T))
    
    Winding_sequence=['A','C1','B','A1','C','B1']
    

    
    New_seq2=np.tile(Winding_sequence,int(L[1]))
    
    Actual_seq2= pd.DataFrame(New_seq2[:,None].T)
    
    Seq_f= pd.concat([Actual_seq1, Actual_seq2],ignore_index=True)
    
    Seq_f.reset_index(drop=True)
    
    Slots=int(S)
    
    if int(S) %2 ==0:
        R=int(S)
    
    else:
        R=int(S)+1
        
    Windings_arrange=(pd.DataFrame(index=Seq_f.index,columns=Seq_f.columns[1:R])).fillna(0)
    
    counter=1
	#Step #4 Arranging winding in Slots
    
    for i in range(0,len(New_seq)):
        if Seq_f.loc[0,i]==1:
            Windings_arrange.loc[0,counter]=Seq_f.loc[1,i]
            counter=counter+1
            
    Windings_arrange.loc[1,1]='C1'
    
    for k in range(1,R):
        if Windings_arrange.loc[0,k]=='A':
            Windings_arrange.loc[1,k+1]='A1'
        elif Windings_arrange.loc[0,k]=='B':
            Windings_arrange.loc[1,k+1]='B1'
        elif Windings_arrange.loc[0,k]=='C':
            Windings_arrange.loc[1,k+1]='C1'
        elif Windings_arrange.loc[0,k]=='A1':
            Windings_arrange.loc[1,k+1]='A'
        elif Windings_arrange.loc[0,k]=='B1':
            Windings_arrange.loc[1,k+1]='B'
        elif Windings_arrange.loc[0,k]=='C1':
            Windings_arrange.loc[1,k+1]='C'
	
    Phase_A=np.zeros((1000,1),dtype=float)
    counter_A=0
    Windings_arrange.to_excel('test.xlsx')
    # Winding vector, W_A for Phase A
    for l in range(1,R):
        if Windings_arrange.loc[0,l]=='A' and Windings_arrange.loc[1,l]=='A':
            Phase_A[counter_A,0]=l
            Phase_A[counter_A+1,0]=l
            counter_A=counter_A+2
        elif Windings_arrange.loc[0,l]== 'A1' and Windings_arrange.loc[1,l]=='A1':
            Phase_A[counter_A,0]=-1*l
            Phase_A[counter_A+1,0]=-1*l
            counter_A=counter_A+2
        elif Windings_arrange.loc[0,l]=='A' or Windings_arrange.loc[1,l]=='A':
            Phase_A[counter_A,0]=l
            counter_A=counter_A+1
        elif Windings_arrange.loc[0,l]=='A1' or Windings_arrange.loc[1,l]=='A1':
            Phase_A[counter_A,0]=-1*l
            counter_A=counter_A+1
    
    W_A=(np.trim_zeros(Phase_A)).T
    # Calculate winding factor
    K_w=0
    
    for r in range(0,int(2*(S)/3)):
        Gamma=2*pi*p*abs(W_A[0,r])/int(S)
        K_w+=np.sign(W_A[0,r])*(exp(Gamma*1j))

    K_w=abs(K_w)/(2*int(S)/3)
    CPMR=lcm(int(S),int(2*p))
    N_cog_s=CPMR/S
    N_cog_p=CPMR/p
    N_cog_t=CPMR*0.5/p
    A=lcm(int(S),int(2*p))
    b_p_tau_p=2*1*p/S-0
    b_t_tau_s=(2)*S*0.5/p-2
        
    return K_w