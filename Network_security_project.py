'''
Yehonatan Arama 207938903
Dvir Zaguri 315602284
Eran Deutsch 209191063
Thomas Mendelson 209400654
'''
import math
import random
import sys
import numpy as np


def DefaultCostFunc(self, SigXr):
    return math.exp(SigXr)

class Link:
    capacity = None
    users = set()

    def __init__(self, firstUser, secondUser):
        self.users.add(firstUser)
        self.users.add(secondUser)
        self.capacity = 1

    def Contains(self, friend):
        return friend in self.users

    def SetCapacity(self, n):
        self.capacity = n


class User:
    x_pos = None
    y_pos = None
    links = set()
    Xr = None

    def __init__(self, M):
        radius = random.uniform(0, M)
        theta = random.uniform(0, 2 * math.pi)
        self.x_pos = radius * math.cos(theta)
        self.y_pos = radius * math.sin(theta)
        self.Xr = 1

    def __init__(self,x_pos,y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.Xr = 1

    def Dist(self, friend):
        return math.sqrt((self.x_pos - friend.x_pos) * 2 + (self.y_pos - friend.y_pos) * 2)

    def HasLink(self, friend):
        for link in self.links:
            if link.Contains(friend):
                return True
        return False

    def AddLink(self, friend):
        link = Link(self, friend)
        self.links.add(link)
        friend.links.add(link)
        return link

    def SetXr(self, xr):
        self.Xr = xr

    def GetXr(self):
        return self.Xr

class Graph:
    N = None
    M = None
    r = None
    alpha = None
    users = []
    links = set()
    NumOfLinks = None
    stepSize = None
    maxIterSteps = None
    costFunc = None
    routes = []         #Two dimensional array - the rows are source, the collomns are destination and the cells are list of links on the route

    #TODO routes are not implemented yet in the random constructor
    def __init__(self, N, M, r, alpha):
        self.N = N
        self.M = M
        self.r = r
        self.alpha = alpha
        self.CreateRandomGraph()
        self.NumOfLinks = self.GetNumLinks()

    def __init__(self, links , users , routes, alpha , costFunc = DefaultCostFunc , stepSize = lambda user: 1e-3 , maxIterSteps = 2000):
        self.N = len(users)
        self.users = users
        self.links = links
        self.alpha = alpha
        self.routes = routes
        self.NumOfLinks = len(links)
        self.stepSize = stepSize
        self.maxIterSteps = maxIterSteps
        self.costFunc = costFunc

    def CreateRandomGraph(self):
        for i in range(self.N):
            self.users.append(User(self.M))
        for user in self.users:
            for friend in self.users:
                if (user == friend):
                    continue
                if user.Dist(friend) < self.r:
                    if not (user.HasLink(friend)):
                        self.links.add(user.AddLink(friend))

    def GetNumLinks(self):
        return len(self.links)

    def Ur(self, user):
        return (user.GetXr()(1 - self.alpha)) / (1 - self.alpha)

    def DevUr(self, user):
        return 1/(user.GetXr()**self.alpha)

    def Kr(self, user):
        return self.stepSize(user)

    def PrimalIterStep(self,xr,dUr):
        SigXr = 0
        for user in self.users:
            #TODO need to go over the links in the user route, and for each link to sum up the Xr of all users that uses this link
            #TODO problem  - we need to fix this function and also the one that calls it.
        return
    def runPrimal(self):
        XrsForToPlot = np.zeros((len(self.users),1))
        for i in len(self.users):
            XrsForToPlot[i].insert(0,self.users[i].GetXr())
        for i in range(self.maxIterSteps):
            randIndex = random.randint(0,len(self.users)-1)
            chosenUser = self.users[randIndex]
            chosenUser.SetXr(self.PrimalIterStep(chosenUser.GetXr() , self.DevUr(chosenUser)))
            XrsForToPlot[randIndex].append(chosenUser.GetXr())






# Tamplate for run:
# python.exe .\NetworkSecurity_proj.py <question> <N> <M> <r> <alpha>
'''
Arguments for program:
question - the question that we want to test
-N - number of users
-M - radius of the simulation
-r - radius around users for links
-alpha - for alpha fairness
'''



def main():
    args = sys.argv[1:]
    question = args[0]
    N = args[1]
    M = args[2]
    r = args[3]
    alpha = args[4]



if __name__ == "__main__":
    main()
