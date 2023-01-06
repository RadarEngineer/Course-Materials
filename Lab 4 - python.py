# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:56:13 2020

@author: kyled
"""


""" Files """
f = open("demo.txt", "r")
print(f.readline())
f.close()

""" Strings """

t = "Machine Learning"
print(t[0])


""" Lists """
names = ["Dave", "Mark", "Ann", "Phil"]
names.append("Stampy")
print("\n")
print(names)

a = [1,2,3]
b = [4,5]
c = a + b
print(c)


""" Dictionary """ 

car={"brand":"Ford", 
          "model":"Mustang", 
          "year": 1964}
print(car)
print(car["brand"])

""" Loops """

for n in range(1,10):
    print(n)
    
print('\n')
for n in range(1,10,2):
    print(n)
    
""" IF statement """ 

x = float(input("Enter a number for x: "))
y = float(input("Enter a number for y: "))
if x == y:
    print("x and y are equal")
if y != 0:
    print("therefore, x / y is", x/y)
elif x < y:
    print("x is smaller")
else:
    print("y is smaller")
print("thanks!")


