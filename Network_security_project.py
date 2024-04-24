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
from heapq import heappush, heappop
from copy import deepcopy


# 0.1823
def DefaultCostFunc(self, sigXr, cap):
    if cap >= sigXr:
        return 0
    return math.exp(sigXr)


def get_link(links_set, first_user, second_user):
    for link in links_set:
        if link.Connected(first_user) and link.Connected(second_user):
            return link
    return None  # If the link is not found


class Link:
    def __init__(self, firstUser, secondUser, capacity, lambd=0.20633):
        self.connectedTo = set()
        self.connectedTo.add(firstUser)
        self.connectedTo.add(secondUser)
        self.capacity = capacity
        self.users = set()  # that using this link
        self.lambd = lambd

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

    def Getlambd(self):
        return self.lambd

    def Setlambd(self, lambd):
        self.lambd = lambd


class User:

    def __init__(self, M=None, connectedLinks=set(), linksInRoutes=set(),
                 x_pos=None, y_pos=None):
        self.connectedLinks = connectedLinks
        self.linksInRoutes = linksInRoutes
        self.Xr = 1
        if M is None:
            self.x_pos = x_pos
            self.y_pos = y_pos
        else:
            radius = random.uniform(0, M)
            theta = random.uniform(0, 2 * math.pi)
            self.x_pos = radius * math.cos(theta)
            self.y_pos = radius * math.sin(theta)

    # def __init__(self, connectedLinks, linksInRoutes):
    #
    #     self.connectedLinks = connectedLinks
    #     self.linksInRoutes = linksInRoutes
    #     self.Xr = 1

    def Dist(self, friend):
        return math.sqrt((self.x_pos - friend.x_pos) ** 2 + (self.y_pos - friend.y_pos) ** 2)

    def IsConnectedTo(self, friend):
        for link in self.connectedLinks:
            if link.Connected(friend):
                return True
        return False

    def AddConnectedLink(self, friend, capacity=1):
        link = Link(self, friend, capacity)
        self.connectedLinks.add(link)
        friend.connectedLinks.add(link)
        return link

    def AddlinksInRoutes(self, link):
        self.linksInRoutes.add(link)

    def SetXr(self, xr):
        self.Xr = xr

    def GetXr(self):
        return self.Xr

    def HasNoRoutes(self):
        return 0 == len(self.linksInRoutes)


class Graph:

    def __init__(self, N, alpha, M=None, r=None, users=[], links=set(), costFunc=DefaultCostFunc,
                 stepSize=lambda user: 1e-3,
                 maxIterSteps=1000000, type="random"):
        self.N = N
        self.M = M
        self.r = r
        self.users = users
        self.links = links
        self.alpha = alpha
        self.stepSize = stepSize
        self.maxIterSteps = maxIterSteps
        self.costFunc = costFunc
        if type == "random":
            self.CreateRandomGraph()
        self.forDebug = 10

    # def __init__(self, links, users, alpha, costFunc=DefaultCostFunc, stepSize=lambda user: 1e-3, maxIterSteps=1000000):
    #     M = None
    #     r = None
    #     self.N = len(users)
    #     self.users = users
    #     self.links = links
    #     self.alpha = alpha
    #     self.stepSize = stepSize
    #     self.maxIterSteps = maxIterSteps
    #     self.costFunc = costFunc
    #     self.forDebug = 10

    def CreateRandomGraph(self):
        for i in range(self.N):
            self.users.append(User(M=self.M))
        for i, user in enumerate(self.users):
            for j, friend in enumerate(self.users):
                if user != friend:
                    if user.Dist(friend) < self.r and not (user.IsConnectedTo(friend)):
                        link = user.AddConnectedLink(friend=friend)
                        self.links.add(link)

    def Ur(self, user):
        return (user.GetXr()(1 - self.alpha)) / (1 - self.alpha)

    def DevUr(self, user):
        return 1 / (user.GetXr() ** self.alpha)

    def Kr(self, user):
        return self.stepSize(user)

    def Hr(self, user):
        return self.stepSize(user)

    def printGraph(self):
        print("users:")
        for i in range(len(self.users)):
            print(f"user {i}:")
            print("connected links with:")
            for j in range(len(self.users)):
                if not (i == j):
                    for l in self.users[i].connectedLinks:
                        if l.Connected(self.users[j]):
                            print(f"connected to user {j}")
            print("")

    def plotRun(self, XrsToPlot, Type):
        for i in range(len(XrsToPlot)):
            plt.plot(XrsToPlot[i], label=f"User {i}")
        plt.xlabel("Iteration")
        plt.ylabel("Xr")
        plt.title(f"Xr vs. Iteration for Each User {Type}")
        plt.legend()
        plt.show()

    def PrimalIterStep(self, chosenUser):
        dUr = self.DevUr(chosenUser)
        kr = self.Kr(chosenUser)
        SigCost = 0
        for l in chosenUser.linksInRoutes:
            SigXr = 0
            for user in l.users:
                SigXr += user.GetXr()
            SigCost += self.costFunc(DefaultCostFunc, SigXr, l.GetCapacity())
        return kr * (dUr - SigCost)

    def DualIterStep(self, chosenUser):
        dUr = self.DevUr(chosenUser)
        Hr = self.Hr(chosenUser)
        Siglambd = 0
        for l in chosenUser.linksInRoutes:
            Siglambd += l.Getlambd()
            yl = 0
            for user in l.users:
                yl += user.GetXr()
            if l.Getlambd() > 0:
                l.Setlambd(l.Getlambd() + Hr * (yl - l.GetCapacity()))
            else:
                l.Setlambd(l.Getlambd() + Hr * max((yl - l.GetCapacity()), 0))
        return 1 / (Siglambd ** (1 / self.alpha))

    def getDijkstraMat(self):
        all_shortest_paths = []

        for source in range(self.N):
            # Initialize distances from the source node to all other nodes as infinite
            distances = [float('inf')] * self.N
            paths = [[] for _ in range(self.N)]
            distances[source] = 0
            pq = [(0, source)]

            while pq:
                curr_dist, idx_curr_user = heappop(pq)
                curr_user = self.users[idx_curr_user]
                for idx_friend, friend in enumerate(self.users):
                    if curr_user == friend:
                        pass
                    elif curr_user.IsConnectedTo(friend):
                        distance_through_current = curr_dist + curr_user.Dist(friend)  # todo 1 or dist by coordinates

                        if distance_through_current < distances[idx_friend]:
                            distances[idx_friend] = distance_through_current
                            paths[idx_friend] = deepcopy(paths[idx_curr_user])
                            paths[idx_friend].append(curr_user)
                            heappush(pq, (distance_through_current, idx_friend))

            all_shortest_paths.append(paths)
        return all_shortest_paths

    def run(self, Type):
        XrsToPlot = []
        for i in range(len(self.users)):
            XrsToPlot.insert(i, [])
            XrsToPlot[i].insert(0, self.users[i].GetXr())
        for i in range(self.maxIterSteps):
            randIndex = random.randint(0, len(self.users) - 1)
            if self.users[randIndex].HasNoRoutes():
                continue
            chosenUser = self.users[randIndex]

            if Type == "Primal":
                chosenUser.SetXr(chosenUser.GetXr() + self.PrimalIterStep(chosenUser))
            if Type == "Dual":
                chosenUser.SetXr(self.DualIterStep(chosenUser))
            for j in range(len(self.users)):
                XrsToPlot[j].append(self.users[j].GetXr())
        for i in range(len(self.users)):
            print(f"user's {i} Xr Value is: {self.users[i].GetXr()}")
        self.plotRun(XrsToPlot, f"{Type} run")

def q4(alpha):  # N = L + 1 = 5 +1
    numOfLinks = 5
    users = []
    links = set()
    # create empty users
    for i in range(numOfLinks + 2):
        users.append(User(connectedLinks=set(), linksInRoutes=set()))

    # create the graph
    for i in range(1, numOfLinks + 1):
        link = users[i].AddConnectedLink(friend=users[i + 1])
        users[0].AddlinksInRoutes(link)
        users[i].AddlinksInRoutes(link)
        link.Adduser(users[0])
        link.Adduser(users[i])
        links.add(link)

    # G_primal = Graph(N=len(links), alpha=alpha, users=users, links=links, type="graph of Q1")
    # G_primal.run("Primal")
    G_dual = Graph(N=len(links), alpha=alpha, users=users, links=links, type="graph of Q1")
    G_dual.run("Dual")


def q5(N, M, r, alpha):
    G = Graph(N=N, M=M, r=r, alpha=alpha)
    G.printGraph()
    dijk_mat = G.getDijkstraMat()

    shuffled_idx = list(range(len(G.users)))
    random.shuffle(shuffled_idx)
    for i in range(0, len(shuffled_idx), 2):
        idx1 = shuffled_idx[i]
        idx2 = shuffled_idx[i + 1] if i + 1 < len(shuffled_idx) else None
        print(f"idx1: {idx1}, idx2: {idx2}")
        if idx2 is not None:
            if dijk_mat[idx1][idx2]:  # we can get from idx1 to idx2
                for j, user in enumerate(dijk_mat[idx1][idx2]):
                    # if user == G.users[idx2]:  # todo check if we add the dest to the mat in getDijkstraMat
                    print(f"len(dijk_mat[idx1][idx2]): {len(dijk_mat[idx1][idx2])}")
                    if j + 1 == len(dijk_mat[idx1][idx2]):
                        print(f"in last link, j: {j}")
                        link = get_link(G.links, user, G.users[idx2])
                        print(f"link2: {link}")
                    else:
                        next_user = dijk_mat[idx1][idx2][j + 1]
                        link = get_link(G.links, user, next_user)
                        print(f"link3: {link}")
                    print(f"link4: {link}")
                    G.users[idx1].AddlinksInRoutes(link)
                    link.Adduser(G.users[idx1])
    G.run("Primal")
    G.run("Dual")


# Template for run:
# python.exe .\Network_security_project.py <options>
'''
Arguments for program:
-q - the question that we want to test question is 4 by default
-alpha - for alpha fairness the alpha is 1 by default
'''


def main():
    # Default values
    question = 4
    alpha = 1
    N = 10
    M = 50
    r = 8
    random.seed(6)
    for arg in sys.argv[1:]:
        if arg.startswith("-q"):
            question = int(arg[2:])
        if arg.startswith("-alpha"):
            alpha = int(arg[6:])
        if arg.startswith("-N"):
            N = int(arg[2:])
        if arg.startswith("-M"):
            M = int(arg[2:])
        if arg.startswith("-r"):
            r = int(arg[2:])

    if question == 4:
        q4(alpha)
    if question == 5:
        q5(N, M, r, alpha)


if __name__ == "__main__":
    main()
