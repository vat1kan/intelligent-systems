from matplotlib import pyplot as plt
from termcolor import colored
import numpy as np

#class generation method
#randomly create array x and y with min value 'l', max value 'h' and with the size 's'
#rounding to two decimal placer for clarity
def create_data(l,h,s):
    x = np.round(np.random.uniform(low=l,high=h,size=s),2)
    y = np.round(np.random.uniform(low=l/2,high=h/5,size=s),2)
    return x,y


#method returns coordinates of class center

def center(x,y):
    center=(max(x)+min(x))/2., (max(y)+min(y))/2.
    return np.round(center,2)


#initial data - size of testing sample
l = 1
h = 30
size = 150
x1,y1 = np.array(np.round(create_data(l,h,size),2)) #creating first class 
x2,y2 = np.array(np.round(create_data(l*1.5,h*2,size),2)) #creating second class

#get centers of each class
c1 = center(x1,y1)
c2 = center(x2,y2)

#init variable to count deviation 
dev_1 = 0
dev_2 = 0


#checking if an element belongs to the first class
print("\nIncorrectly classificated elements in first class:")
for i in range(len(x1)):
    #using euclidean metric to get the distance to class center
    d1 = np.sqrt(np.power((c1[0]-x1[i]),2)  +  np.power((c1[-1]-y1[i]),2))
    d2 = np.sqrt(np.power((c2[0]-x1[i]),2)  +  np.power((c2[-1]-y1[i]),2))
    #compare distances 
    if  d1 < d2 :
        #the element really belongs to the first class
        #print(f"{i+1})First class element {x1[i]},{y1[i]} classificated correctly!({np.round(d1)} - {np.round(d2)})")
        pass
    else:
        #the element does not belong to this class
        #print(colored(((f"{i+1}) ! First class element {x1[i]},{y1[i]} classificated incorrectly!({np.round(d1)} - {np.round(d2)})")),'red'))
        dev_1 += 1 #so counter of deviations increases by 1  
print(dev_1)


#checking if an element belongs to the second class
print("\nIncorrectly classificated elements in second class:\n")
for i in range(len(x2)):
    #using euclidean metric to get the distance to class center
    d1 = np.sqrt(np.power((c2[0]-x2[i]),2)  +  np.power((c2[-1]-y2[i]),2))
    d2 = np.sqrt(np.power((c1[0]-x2[i]),2)  +  np.power((c1[-1]-y2[i]),2))
    if  d1 < d2 :
        #the element really belongs to the second class
        #print(f"{i+1})Second class element  {x2[i]},{y2[i]} classificated correctly!({np.round(d1)} - {np.round(d2)})")
        pass
    else:
        #the element does not belong to this class
        #print(colored(((f"{i+1}) ! First class element {x2[i]},{y2[i]} classificated incorrectly!({np.round(d1)} - {np.round(d2)})")),'red'))
        dev_2 += 1 #so counter of deviations increases by 1
print(dev_2)

#total amout of deviations 
deviation = dev_1+dev_2
print (f"\nDeviation: {deviation}")

#probability calculation by formula P = 1 - (deviations/total elements)
probability = 1-(deviation/(len(x1)+len(x2)))
print(f"Probability: {np.round(probability,3)}") 



# #figure of classifications 
fig,ax = plt.subplots()
plt.plot(x1, y1, 'g^',label="First class") #first class sample
plt.plot(x2, y2, 'bs',label="Second class") #second class sample
ax.scatter(x=c1[0],y=c1[-1],s = 100,label="First class center") #center of first class
ax.scatter(c2[0],c2[-1],s = 100,label ="Second class center") # center of second class
ax.set(title="Classification")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend()
plt.grid()
plt.show()