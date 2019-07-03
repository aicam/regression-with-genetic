import random as rnd
import math
class Individual:
    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 0
    def __init__(self,a0,a1,a2,a3):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    fitness = 0
    def calculate(self,x):
        return self.a3*math.pow(x,3) + self.a2*math.pow(x,2) + self.a1*math.pow(x,1) + self.a0
    def __lt__(self, other):
        return self.fitness < other.fitness
