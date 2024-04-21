'''
Yehonatan Arama 207938903
Dvir Zaguri 315602284
Eran Deutsch 209191063
Thomas Mendelson 209400654
'''
import math
import random
import sys
import matplotlib.pyplot as plt


def DefaultCostFunc(self, sigXr):
    return math.exp(sigXr)


def get_link(links_set, first_user, second_user):
    for link in links_set:
        if link.Connected(first_user) and link.Connected(second_user):
            return link
    return None  # If the link is not found


class Link:
    capacity = None
    connectedTo = set()
    users = set()

    def __init__(self, firstUser, secondUser, capacity):
        self.connectedTo.add(firstUser)
        self.connectedTo.add(secondUser)
        self.capacity = capacity

    def Addconection(self, user):
        self.connectedTo.add(user)
    def Connected(self, user):
        return user in self.connectedTo

    def SetCapacity(self, n):
        self.capacity = n

    def GetCapacity(self):
        return self.capacity

    def Adduser(self, user):
        self.users.add(user)

    def RemoveUser(self, user):
        if user in self.users:
            self.users.remove(user)


class User:
    x_pos = None
    y_pos = None
    connectedLinks = set()
    linksInRoutes = set()
    Xr = None

    # def __init__(self, M):
    #    radius = random.uniform(0, M)
    #    theta = random.uniform(0, 2 * math.pi)
    #    self.x_pos = radius * math.cos(theta)
    #    self.y_pos = radius * math.sin(theta)
    #    self.Xr = 1

    def __init__(self, connectedLinks, linksInRoutes):
        self.connectedLinks = connectedLinks
        self.linksInRoutes = linksInRoutes
        self.Xr = 1

    def Dist(self, friend):
        return math.sqrt((self.x_pos - friend.x_pos) * 2 + (self.y_pos - friend.y_pos) * 2)

    def IsConnectedTo(self, friend):
        for link in self.connectedLinks:
            if link.Connected(friend):
                return True
        return False

    def AddConnectedLink(self, friend, capacity):
        link = Link(self, friend, capacity)
        self.connectedLinks.add(link)
        friend.connectedLinks.add(link)
        return link

    # def AddConnectedLink(self, link):
    #     self.connectedLinks.add(link)

    def AddlinksInRoutes(self, link):
        self.linksInRoutes.add(link)

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

    # routes = []         #Two dimensional array - the rows are source, the collomns are destination and the cells are list of links on the route

    # TODO routes are not implemented yet in the random constructor
    # def __init__(self, N, M, r, alpha):
    #    self.N = N
    #    self.M = M
    #    self.r = r
    #    self.alpha = alpha
    #    self.CreateRandomGraph()
    #    self.NumOfLinks = self.GetNumLinks()

    def __init__(self, links, users, alpha, costFunc=DefaultCostFunc, stepSize=lambda user: 1e-3, maxIterSteps=2000):
        self.N = len(users)
        self.users = users
        self.links = links
        self.alpha = alpha
        self.NumOfLinks = len(links)
        self.stepSize = stepSize
        self.maxIterSteps = maxIterSteps
        self.costFunc = costFunc

    # def CreateRandomGraph(self):
    #    for i in range(self.N):
    #        self.users.append(User(self.M))
    #    for user in self.users:
    #       for friend in self.users:
    #            if (user == friend):
    #                continue
    #            if user.Dist(friend) < self.r:
    #                if not (user.HasLink(friend)):
    #                    self.links.add(user.AddLink(friend))

    def GetNumLinks(self):
        return self.NumOfLinks

    def Ur(self, user):
        return (user.GetXr()(1 - self.alpha)) / (1 - self.alpha)

    def DevUr(self, user):
        return 1 / (user.GetXr() ** self.alpha)

    def Kr(self, user):
        return self.stepSize(user)

    def PrimalIterStep(self, chosenUser):
        dUr = self.DevUr(chosenUser)
        kr = self.Kr(chosenUser)
        SigCost = 0
        for l in chosenUser.linksInRoutes:
            SigXr = 0
            for user in l.users:
                SigXr += user.GetXr()
            SigCost += self.costFunc(DefaultCostFunc, SigXr)
        return kr * (dUr - SigCost)

    def runPrimal(self):
        XrsToPlot = []
        for i in range(len(self.users)):
            XrsToPlot.insert(i, [])
            XrsToPlot[i].insert(0, self.users[i].GetXr())
        for i in range(self.maxIterSteps):
            randIndex = random.randint(0, len(self.users) - 1)
            chosenUser = self.users[randIndex]
            chosenUser.SetXr(self.PrimalIterStep(chosenUser))
            XrsToPlot[randIndex].append(chosenUser.GetXr())
        #add plot of XrsToPlot
        for i in range(len(XrsToPlot)):
            plt.plot(XrsToPlot[i], label=f"User {i + 1}")
        plt.xlabel("Iteration")
        plt.ylabel("Xr")
        plt.title("Xr vs. Iteration for Each User")
        plt.legend()
        plt.show()


# class Link:
#     capacity = None
#     connectedTo = set()
#     users = set()
#     def __init__(self, firstUser, secondUser, capacity):

# class User:
#     x_pos = None
#     y_pos = None
#     connectedLinks = set()
#     linksInRoutes = set()
#     Xr = None
#     def __init__(self, connectedLinks, linksInRoutes):

def q4(alpha, numOfLinks):
    users = []
    links = set()
    # creat empty users
    for i in range(numOfLinks + 1):
        users.append(User(set(), set()))

    # creat the links
    for i in range(1, numOfLinks):
        link = users[i].AddConnectedLink(users[i + 1], 1)
        users[0].AddlinksInRoutes(link)
        users[i].AddlinksInRoutes(link)
        link.Adduser(users[0])
        link.Adduser(users[i])
        links.add(link)

    G = Graph(links, users, alpha)
    G.runPrimal()





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

    print(f"args[0]: {question}, args[1]: {N}, args[2]: {M}, args[3]: {r}, args[4]: {alpha}")
    if question == "q4":
        # N = L + 1 = 5 +1
        q4(alpha, 5)


if __name__ == "__main__":
    main()
