# implement the feed forward neural network by backpropogation:
# intialize the inputs,weights and bias values:
import numpy as np
x1=1
x2=0
x3=1
w14=0.2
w24=0.4
w34=-0.5
w15=-0.3
w25=0.1
w35=0.2
w46=-0.3
w56=0.2
b4=-0.4
b5=0.2
b6=0.1

#implemet loop to get 5iterations:
I4,I5,I6,O4,O5,O6,E4,E5,E6 =0,0,0,0,0,0,0,0,0  #intialize values as zero
for x in range(5):
    I4=w14*x1+w24*x2+w34*x3+b4
    I5=w15*x1+w25*x2+w35*x3+b5
    I6=w46*I4+w56*I5+b6
    
    O4=1/(1+np.exp(-I4))
    O5=1/(1+np.exp(-I5))
    O6=1/(1+np.exp(-I6))
    
    E6=O6*(1-O6)*(1-O6)
    E5=O5*(1-O5)*E6*w56
    E4=O4*(1-O4)*E6*w46
    
    #to update the new values of weight and bias:
    w14=w14 + (0.9*E4*x1)
    w15=w15 + (0.9*E5*x1)
    w24=w24 + (0.9*E4*x2)
    w25=w25 + (0.9*E5*x2)
    w34=w34 + (0.9*E4*x3)
    w35=w35 + (0.9*E5*x3)
    w46=w46 + (0.9*E6*O4)
    w56=w56 + (0.9*E6*O5)
    b4=b4 + (0.9*E4)
    b5=b5 + (0.9*E5)
    b6=b6 + (0.9*E6)
    
# final values:
print("The Results getting after 5 epochs are: ")
print("The final value of W14: ",w14)
print("The final value of W15: ",w15)
print("The final value of W24: ",w24)
print("The final value of W25: ",w25)
print("The final value of W34: ",w34)
print("The final value of W35: ",w35)
print("The final value of W46: ",w46)
print("The final value of W56: ",w56)
print("The final value of bias b4: ",b4)
print("The final value of bias b4: ",b5)
print("The final value of bias b4: ",b6)
