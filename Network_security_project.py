'''
Yehonatan Arama 207938903
Dvir Zaguri 315602284
Eran Deutsch 209191063
Thomas Mendelson 209400654
'''
import math
import random
import sys

class Link:
    capacity = None
    users = set()

    def __init__(self,firstUser,secondUser):
        self.users.add(firstUser)
        self.users.add(secondUser)
        self.capacity = 1


    def Contains (self,friend):
        return friend in self.users

    def SetCapacity(self , n):
        self.capacity = n
class User:
    x_pos = None
    y_pos = None
    links = set()
    #power = None
    #bw = None
    Xr = None

    def __init__(self,M):
        radius = random.uniform(0, M)
        theta = random.uniform(0, 2 * math.pi)
        self.x_pos = radius * math.cos(theta)
        self.y_pos = radius * math.sin(theta)
        self.Xr = 1

    def Dist (self , friend):
        return math.sqrt((self.x_pos - friend.x_pos)*2 + (self.y_pos - friend.y_pos)*2)
    def HasLink (self , friend):
        for link in self.links:
            if link.Contains(friend):
                return True
        return False
    def AddLink(self, friend):
        link = Link(self,friend)
        self.links.add(link)
        friend.links.add(link)
        return link

    def SetXr(self,xr):
        self.Xr = xr

    def GetXr(self):
        return self.Xr

    #def SetPower (self,power):
    #    self.power = power

    #def SetBW (self,bw):
    #    self.bw = bw

class Graph:
    N = None
    M = None
    r = None
    alpha = None
    #F = None
    users = []
    links = set()
    #hI = []
    #flows =[] #Two dimensional list each line is a Flow l[0] is source l[1] is destination l[2] is data

    #def __init__(self, N,M,r,F,func_power,func_BW,func_hI):
    def __init__(self, N,M,r,alpha):
        self.N = N
        self.M = M
        self.r = r
        self.alpha = alpha
        self.CreateGraph()
        #self.SetPowerToGraph(func_power)
        #self.SetBWToGraph(func_BW)
        #self.SethIToGraph(func_hI)
        #self.CreateFlows()

    def CreateGraph(self):
        for i in range(self.N):
            self.users.append(User(self.M))
        for user in self.users:
            for friend in self.users:
                if (user == friend):
                    continue
                if user.Dist(friend)<self.r:
                    if not(user.HasLink(friend)):
                        self.links.add(user.AddLink(friend))

    def GetNumLinks(self):
        return len(self.links)

    #def SetAllCapacityTo(self, n):
    #    for link in self.links:
    #        link.SetCapacity(n)

    def Ur(self , user):
        return (user.GetXr()(1-self.alpha))/(1-self.alpha)

    #def SetPowerToGraph(self, func):
    #    func(self)

    #def SetBWToGraph(self, func):
    #    func(self)

    #def SethIToGraph(self, func):
    #    hI = func(self)

    #def CreateFlows(self):
    #    for i in range(self.F):
    #        self.flows[i][0] = random.uniform(0,len(self.users))
    #        self.flows[i][1] = random.uniform(0,len(self.users)-1)
    #        if self.flows[i][0] == self.flows[i][1]:
    #            self.flows[i][1] = len(self.users)-1
    #        self.flows[i][2] = random.uniform(0,1e6)




#def func_hIZeros(g):
#    for i in range(len(g.users)):
#        g.hI[i] = []
#        for j in range(len(g.users)):
#            g.hI[i][j] = 0

def Run_question2(N , M , r , alpha):
    g = Graph(N,M,r)
    NumOfLinks = g.GetNumLinks()



# Tamplate for run:
# python.exe .\NetworkSecurity_proj.py <question> <N> <M> <r> <alpha>
'''
Arguments for program:
question - the question that we want to test
N - number of users
M - radius of the simulation
r - radius around users for links
alpha - for alpha fairness
'''

def main():
    args = sys.argv[1:]
    question = args[0]
    N = args[1]
    M = args[2]
    r = args[3]
    alpha = args[4]
    if (question==2):
        Run_question2(N , M , r , alpha)








if __name__ == "__main__":
    main()

#yoni