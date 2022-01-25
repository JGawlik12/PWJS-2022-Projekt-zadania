# -*- coding: utf-8 -*-


#zad1

print("hello world!")
  
  
  
#zad2

personInfo = raw_input("date of birth: ")
print(personInfo)
  
  
  
#zad3

code = input("code: ")
code2 = input("code2: ")
if(code2 == code):
    print("codes match")
else:
    print("codes error")
  
  
  
#zad4

import os
 
dir = r"C:\Users\Jarek\Documents\Python"
lista = os.listdir(dir)
number_files = len(lista)
print number_files
  
  
  
#zad5

import os
for root, dirs, files in os.walk(r'C:\Users\Jarek\Documents\Python'): 
    print root, dirs, files
  
  
  
#zad6

import os
os.chdir(r'C:\Users\Jarek\Documents\Python')  
old = ''
new = ''
for file in os.listdir(r'C:\Users\Jarek\Documents\Python'):   
    if file.endswith(".jpg"):
        old = os.path.join(os.getcwd()) + '\\' + file
        new = old.split('.')
        print(old)
        print(new[0]+".png")
        os.rename(old , new[0]+'.png')
  
  
  
#zad7

import os
os.chdir(r'C:\Users\Jarek\Documents\Python')
infile = os.path.join(os.getcwd()) + '\\' + "nowomowa.txt"
outfile = os.path.join(os.getcwd()) + '\\' + "wynik_zadania_7.txt"

delete = ["się", " i ", "nigdy", "Nigdy", "dlaczego", "Dlaczego"]
with open(infile) as fin, open(outfile, "w+") as fout:
    for line in fin:
        for word in delete:
            line = line.replace(word, "")
        fout.write(line)



#zad8

import os
os.chdir(r'C:\Users\Jarek\Documents\Python')
infile = os.path.join(os.getcwd()) + '\\' + "nowomowa.txt"
outfile = os.path.join(os.getcwd()) + '\\' + "wynik_zadania_8.txt"

replaceDict = {"się":"oraz", " i ":" i ", "nigdy":"prawie nigdy", "Nigdy":"Prawie nigdy", "dlaczego":"czemu", "Dlaczego":"Czemu"}

with open(infile) as fin, open(outfile, "w+") as fout:
    for line in fin:
        for key in replaceDict.keys():
            line = line.replace(key, replaceDict[key])
        fout.write(line)
  
  
  
#zad9

import math
from math import sqrt
 
a = float(input("a: "))
b = float(input("b: "))
c = float(input("c: "))
delta = b**2 - 4*a*c
 
if delta > 0:
    num_roots = 2
    x1 = (((-b) + sqrt(delta))/(2*a))     
    x2 = (((-b) - sqrt(delta))/(2*a))
    print(x1, x2)
elif delta == 0:
    num_roots = 1
    x = (-b) / 2*a
    print(x)
else:
    num_roots = 0
    print("brak rozwiazan")
  
  
  
#zad10

import random
L = []
for i in range(50):
    L.append(random.random())
P = L
for i in L:
    for x in range(len(L) - 1):
        temp = L[x]
        if L[x] < L[x+1]:
            L[x] = L[x+1]
            L[x+1] = temp
   
print(L)
P.sort()
P.reverse()
print(P)
  
  
  
#zad11

a = [1, 2, 12, 4] 
b = [2, 4, 2, 8]
c = []
for i in range(len(a)):
    c.append(a[i]*b[i])
print(c)
  
  
  
#zad12

N = 128
import random
L1 = [[random.random() for i in range(N)] for j in range(N)]
L2 = [[random.random() for i in range(N)] for j in range(N)]
L3 = []
for i in range(N):
    L3.append([])
    for j in range(N):
        L3[i].append(L1[i][j] + L2[i][j])
  
  
  
#zad13

N = 8
import random
L1 = [[random.random() for i in range(N)] for j in range(N)]
L2 = [[random.random() for i in range(N)] for j in range(N)]
L3 = [[0 for i in range(N)] for j in range(N)]
for i in range(len(L1)):  
    for j in range(len(L2[0])):  
        for k in range(len(L2)):  
            L3[i][j] += L1[i][k] * L2[k][j]  
for r in L3:  
    print(r)
      
      

#zad14   

import random as rd
import numpy as np

L = []
rd.seed()

def getDet(L, m):
    for i in range(0,7):
    L.append([])
    for j in range(0,7):
        L[i].append(rd.random()*m)

    return np.linalg.det(L)

print(L)
print(getDet(L, 999))
    
      
         
#zad15

import math
from math import sqrt
  
class Complex(object):
  
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag
        print(self.real + self.imag)
  
    def __add__(self, other):
        print('\nSum:')
        return Complex(self.real + other.real, self.imag + other.imag)
  
    def __sub__(self, other):
        print('\nDifference:')
        return Complex(self.real - other.real, self.imag - other.imag)
      
    def __mul__(self, other):
        print('\nProduct:')
        return Complex((self.real * other.real) - (self.imag * other.imag),
            (self.imag * other.real) + (self.real * other.imag))
  
    def __div__(self, other):
        print('\nQuotient:')
        r = (other.real**2 + other.imag**2)
        return Complex((self.real*other.real - self.imag*other.imag)/r,
            (self.imag*other.real + self.real*other.imag)/r)
  
    def __abs__(self):
        print('\nAbsolute Value:')
        new = (self.real**2 + (self.imag**2)*-1)
        return Complex(sqrt(new.real))
      
      
      
#zad 16

while True:
    
    choice = str(input("Enter choice(1/2/3/4): "))
 
    if choice in ('1', '2', '3', '4'):
         
        real1 = float(input("Enter first real number: "))
        im1 = complex(input("Enter first im number(j): "))
 
        real2 = float(input("Enter second real number: "))
        im2 = complex(input("Enter second im number(j): "))
         
        i = Complex(real1, im1)
        k = Complex(real2, im2)
 
        if choice == '1':
            i+k
 
        elif choice == '2':
            i-k
 
        elif choice == '3':
            i*k
 
        elif choice == '4':
            i/k
        break
    else:
        print("Invalid Input")
  
  
  
#zad17

import os
os.chdir(r'C:\Users\Jarek\Documents\Python')  
import xml.etree.ElementTree as xml
students=[{'name':'Jarek','age':21,'marks':50},{'name':'Michal','age':22,'marks':60}]
root = xml.Element("students")
for student in students:
    child=xml.Element("student")
    root.append(child)
    nm = xml.SubElement(child, "name")
    nm.text = student.get('name')
    age = xml.SubElement(child, "age")
    age.text = str(student.get('age'))
    mark=xml.SubElement(child, "marks")
    mark.text=str(student.get('marks'))
tree = xml.ElementTree(root)
with open('myfile.xml', "wb") as fh:
    tree.write(fh)
      
import xml.etree.ElementTree as xml
tree = xml.ElementTree(file='myfile.xml')
root = tree.getroot()
for x in root.iter('marks'):
    mark=int (x.text)
    mark=mark+10
    x.text=str(mark)
with open("myfile_modified.xml", "wb") as fh:
    tree.write(fh)
  
  
  
#zad18

import os
os.chdir(r'C:\Users\Jarek\Documents\Python') 
import json
  
movie_dict =  {"Title": "Star Wars",
                "languages": ["English", "Fench"],
                "borrowed": True,
                "days": 32
                }
  
with open('person.json', 'w') as json_file:
    json.dump(movie_dict, json_file)
      
with open(os.getcwd()+'\person.json', 'r') as f:
    data = json.load(f)
    data["days"] = 0
    data["borrowed"] = False
    data["languages"][1] = "Polish"
      
with open(os.getcwd()+'\person.json', 'w') as f:
    json.dump(data, f)
print(data)
  
  
  
#zad19

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


mu = 100 
sigma = 15 
x = mu + sigma * np.random.randn(10000)

num_bins = 20

n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)


y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')


plt.subplots_adjust(left=0.15)
plt.show()
  
  
  
#zad20

import threading
import random
import time
 

class Philosopher(threading.Thread):
    running = True  
 

    def __init__(self, index, forkOnLeft, forkOnRight):
        threading.Thread.__init__(self)
        self.index = index
        self.forkOnLeft = forkOnLeft
        self.forkOnRight = forkOnRight
 
    def run(self):
        while(self.running):
           
            time.sleep(30)
            print ('Philosopher %s is hungry.' % self.index)
            self.dine()
 
    def dine(self):
       
        fork1, fork2 = self.forkOnLeft, self.forkOnRight
        while self.running:
            fork1.acquire() 
            locked = fork2.acquire(False) 
            if locked: break 
            fork1.release()
            print ('Philosopher %s swaps forks.' % self.index)
            fork1, fork2 = fork2, fork1
        else:
            return
        self.dining()
  
        fork2.release()
        fork1.release()
  
    def dining(self):            
        print ('Philosopher %s starts eating. '% self.index)
        time.sleep(30)
        print ('Philosopher %s finishes eating and leaves to think.' % self.index)
 
def main():
    forks = [threading.Semaphore() for n in range(5)] 
 
  
    philosophers= [Philosopher(i, forks[i%5], forks[(i+1)%5])
            for i in range(5)]
 
    Philosopher.running = True
    for p in philosophers: p.start()
    time.sleep(100)
    Philosopher.running = False
    print ("Now we're finishing.")
  
main()
