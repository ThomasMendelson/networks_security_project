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
import numpy as np



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

def copy_array(arr):
    new_arr = []
    for i in arr:
        new_arr.append(i)
    return new_arr


class Flow:
    nameOfFlowWithinClass = 0
    def __init__(self, destUser, srcUser, pktSize):
        self.destUser = destUser
        self.srcUser = srcUser
        self.pktSize = pktSize
        self.links = None
        self.name = Flow.nameOfFlowWithinClass
        Flow.nameOfFlowWithinClass += 1
        self.Xr = 1
        srcUser.AddFlow(self)

    def Getname(self):
        return self.name

    def GetSource(self):
        return self.srcUser

    def GetDest(self):
        return self.destUser

    def GetpktSize(self):
        return self.pktSize

    def SetLinks(self , links):
        self.links = links

    def printFlow(self):
        print(f"Flow number : {self.name}")
        print(f"Source : {self.srcUser}")
        print(f"Destination : {self.destUser}")
        print(f"pkt size : {self.pktSize}")
        print("links in flow:")
        for link in self.links:
            print(f"{link.Getname()} ",end="")
        print("")

    def SetXr(self, xr):
        self.Xr = xr

    def GetXr(self):
        return self.Xr


class Link:
    nameOfLinkWithinClass = 0

    def __init__(self, firstUser, secondUser, capacity = None, lambd=0.20633 , scale=np.sqrt(0.5)):
        self.connectedTo = []
        self.connectedTo.append(firstUser)
        self.connectedTo.append(secondUser)
        self.flows = set()  # that using this link
        self.lambd = lambd
        self.scale = scale
        self.gain = self.CalcGain(self.scale)
        if capacity == None:
            capacity = self.CalcCapacity()
        self.capacity = capacity
        self.name = Link.nameOfLinkWithinClass
        Link.nameOfLinkWithinClass += 1

    def Addconection(self, user):
        self.connectedTo.append(user)

    def Connected(self, user):
        return user in self.connectedTo

    def SetCapacity(self, n):
        self.capacity = n

    def GetCapacity(self):
        return self.capacity

    def AddFlow(self, flow):
        self.flows.add(flow)

    def RemoveFlow(self, flow):
        if flow in self.flows:
            self.flows.remove(flow)

    def Getlambd(self):
        return self.lambd

    def Setlambd(self, lambd):
        self.lambd = lambd

    def Getname(self):
        return self.name

    def CalcGain(self, scale):
        smallScaleFading = np.random.rayleigh(scale=scale, size=1)
        dist = self.connectedTo[0].Dist(self.connectedTo[1])
        gain = 1 / (smallScaleFading * np.sqrt(dist + (4 * np.pi / 3e8)))
        return np.abs(gain)

    def CalcCapacity(self):
        capToReturn = []
        user1Bw = self.connectedTo[0].GetBw()
        user2Bw = self.connectedTo[1].GetBw()
        power1 = self.connectedTo[0].GetPower() * self.gain
        power2 = self.connectedTo[1].GetPower() * self.gain
        interference1 = self.connectedTo[0].GetSumInterference()
        interference2 = self.connectedTo[1].GetSumInterference()
        sinr1 = power1 / (interference2 + 1)
        sinr2 = power2 / (interference1 + 1)
        capToReturn.append = user1Bw * math.log2(1 + sinr1)
        capToReturn.append = user2Bw * math.log2(1 + sinr2)

    def printLink(self):
        print(f"printing Link num {self.name} (nameOfLinkInClass)")
        print(f"Link is connecting between:")
        for user in self.connectedTo:
            print(f"user : {user.Getname()} (nameOfUserInClass)")


class User:
    nameOfUserWithinClass = 0

    def __init__(self, M=None, x_pos=None, y_pos=None , powerFunc = lambda: 0, bwFunc = lambda: 1):
        self.connectedLinks = set()
        self.flowFromMe = set()
        self.name = User.nameOfUserWithinClass
        User.nameOfUserWithinClass += 1
        if M is None:
            self.x_pos = x_pos
            self.y_pos = y_pos
        else:
            radius = random.uniform(0, M)
            theta = random.uniform(0, 2 * math.pi)
            self.x_pos = radius * math.cos(theta)
            self.y_pos = radius * math.sin(theta)
        self.interference = None
        self.power = powerFunc()
        self.bw = bwFunc()

    def Dist(self, friend):
        return math.sqrt((self.x_pos - friend.x_pos) ** 2 + (self.y_pos - friend.y_pos) ** 2)

    def GetBw(self):
        return self.bw

    def GetPower(self):
        return self.power


    def IsConnectedTo(self, friend):
        for link in self.connectedLinks:
            if link.Connected(friend):
                return True
        return False

    def AddConnectedLink(self, friend, capacity=[1,1]):
        link = Link(self, friend, capacity)
        self.connectedLinks.add(link)
        friend.connectedLinks.add(link)
        return link

    def AddFlow(self, flow):
        self.flowFromMe.add(flow)

    def HasNoFlows(self):
        return 0 == len(self.flowFromMe)

    def Getname(self):
        return self.name

    def printUser(self):
        print(f"printing User num {self.name} (nameOfLinkInClass)")
        if (self.x_pos is not None) and (self.y_pos is not None):
            print(f"x pos: {self.x_pos}\ny pos: {self.y_pos}")
        print(f"User is connected to Links:")
        for link in self.connectedLinks:
            print(f"link : {link.Getname()} (nameOfUserInClass)")
        print(f"User flows:")
        for flow in self.flowFromMe:
            print(f"{flow.Getname()} ",end="")

    def SetSumInterference(self, sumInterference):
        self.interference = sumInterference

    def GetSumInterference(self):
        return self.interference


class Graph:

    def __init__(self, N, alpha, M=None, r=None, users=None, flows = None, links=None, costFunc=DefaultCostFunc, stepSize=lambda flow: 1e-3, maxIterSteps=1000000 , interferenceFunc = lambda: 0):
        self.N = N
        self.M = M
        self.r = r
        if users == None:
            users = []
        self.users = users
        if flows == None:
            flows = []
        self.flows = flows
        if links == None:
            links = set()
        self.links = links
        self.alpha = alpha
        self.stepSize = stepSize
        self.maxIterSteps = maxIterSteps
        self.costFunc = costFunc
        if users == []:
            self.CreateRandomGraph()
        self.CalcInterfirence(interferenceFunc)

    def CreateRandomGraph(self):
        for i in range(self.N):
            self.users.append(User(M=self.M))
        for i, user in enumerate(self.users):
            for j, friend in enumerate(self.users):
                if user != friend:
                    if user.Dist(friend) < self.r and not (user.IsConnectedTo(friend)):
                        link = user.AddConnectedLink(friend=friend)
                        self.links.add(link)

    def CalcInterference(self, func):
        for user in self.users:
            sumInterference = 0
            for friend in self.users:
                if user == friend:
                    continue
                sumInterference += user.Dist(friend) * func()
            user.SetSumInterference(sumInterference)


    def GetUsers(self):
        return self.users

    def resetXrs(self, val = 1):
        for flow in self.flows:
            flow.SetXr(val)

    def AddFlow(self, flow):
        self.flows.append(flow)

    def Ur(self, flow):
        return (flow.GetXr()(1 - self.alpha)) / (1 - self.alpha)

    def DevUr(self, flow):
        return 1 / (flow.GetXr() ** self.alpha)

    def Kr(self, flow):
        return self.stepSize(flow)

    def Hr(self, flow):
        return self.stepSize(flow)

    def printGraph(self):
        print("users:")
        for i in range(len(self.users)):
            print(f"user {i}:")
            self.users[i].printUser()
        print("")

    def plotRun(self, XrsToPlot, Type):
        for i in range(len(XrsToPlot)):
            plt.plot(XrsToPlot[i], label=f"User {i}")
        plt.xlabel("Iteration")
        plt.ylabel("Xr")
        plt.title(f"Xr vs. Iteration for Each User {Type}")
        plt.legend()
        plt.show()

    def PrimalIterStep(self, chosenFlow):
        dUr = self.DevUr(chosenFlow)
        kr = self.Kr(chosenFlow)
        SigCost = 0
        for l in chosenFlow.links:
            SigXr = 0
            for flow in l.flows:
                SigXr += flow.GetXr()
            SigCost += self.costFunc(DefaultCostFunc, SigXr, l.GetCapacity())
        return kr * (dUr - SigCost)

    def DualIterStep(self, chosenFlow):
        dUr = self.DevUr(chosenFlow)
        Hr = self.Hr(chosenFlow)
        Siglambd = 0
        for l in chosenFlow.links:
            Siglambd += l.Getlambd()
            yl = 0
            for flow in l.flows:
                yl += flow.GetXr()
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
                        distance_through_current = curr_dist + 1  # todo 1 or dist by coordinates

                        if distance_through_current < distances[idx_friend]:
                            distances[idx_friend] = distance_through_current
                            paths[idx_friend] = copy_array(paths[idx_curr_user])
                            paths[idx_friend].append(curr_user)
                            heappush(pq, (distance_through_current, idx_friend))

            all_shortest_paths.append(paths)
        return all_shortest_paths

    def run(self, Type):
        XrsToPlot = []
        for i , flow in enumerate(self.flows):
            XrsToPlot.insert(i, [])
            XrsToPlot[i].insert(0, flow.GetXr())
        for i in range(self.maxIterSteps):
            randIndex = random.randint(0, len(self.flows) - 1)
            chosenFlow = self.flows[randIndex]
            if Type == "Primal":
                chosenFlow.SetXr(chosenFlow.GetXr() + self.PrimalIterStep(chosenFlow))
            if Type == "Dual":
                chosenFlow.SetXr(self.DualIterStep(chosenFlow))
            for j, flow in enumerate(self.flows):
                XrsToPlot[j].append(flow.GetXr())
        for i , flow in enumerate(self.flows):                                        #for debug
            print(f"flow's {i} Xr Value is: {flow.GetXr()}")                          #for debug
        self.plotRun(XrsToPlot, f"{Type} run")




def q4(alpha):  # N = L + 1 = 5 +1
    numOfLinks = 5
    users = []
    links = set()
    flows = []
    # create empty users
    for i in range(numOfLinks + 1):
        users.append(User())

    flow0 = []
    #create empty flows
    flows.append(Flow(users[0], users[5], 1))
    for i in range(numOfLinks):
        flows.append(Flow(users[i], users[i+1], 1))

    # create links and connect to flows
    for i in range(numOfLinks):
        link = users[i].AddConnectedLink(friend=users[i + 1])
        flow0.append(link)
        flows[i+1].SetLinks([link])
        link.AddFlow(flows[0])
        link.AddFlow(flows[i+1])
    flows[0].SetLinks(flow0)

    G = Graph(6, alpha, users = users, flows = flows, links = links)
    G.run("Primal")
    G.resetXrs()
    G.run("Dual")


def GetRandomFlow(N):
    source = random.randint(0, N-1)
    while True:
        dest = random.randint(0, N-1)
        if dest != source:
            break
    pkt_size = random.randint(1, 10)*5
    return source, dest, pkt_size


def q5(N, M, r, alpha, flowsNum):

    G = Graph(N=N, M=M, r=r, alpha=alpha)
    dijk_mat = G.getDijkstraMat()

    shuffled_idx = list(range(len(G.users)))
    random.shuffle(shuffled_idx)
    users = G.GetUsers()
    for i in range(0, flowsNum):
        source, dest, pkt_size = GetRandomFlow(len(G.users))
        if dijk_mat[source][dest]:  # we can get from idx1 to idx2
            flow = Flow(users[source], users[dest], pkt_size)
            G.AddFlow(flow)
            links = []
            for j, user in enumerate(dijk_mat[source][dest]):
                if j + 1 == len(dijk_mat[source][dest]):
                    link = get_link(G.links, user, G.users[dest])
                else:
                    next_user = dijk_mat[source][dest][j + 1]
                    link = get_link(G.links, user, next_user)
                if link is not None:
                    links.append(link)
                    link.AddFlow(flow)
                else:
                    print("Link is None!!!!")
            flow.SetLinks(links)
    G.run("Primal")
    G.run("Dual")

def q6(N, M, r, alpha):
    G = Graph(N=N, M=M, r=r, alpha=alpha)
    dijk_mat = G.getDijkstraMat()
    shuffled_idx = list(range(len(G.users)))
    random.shuffle(shuffled_idx)
    for i in range(0, len(shuffled_idx), 2):
        idx1 = shuffled_idx[i]
        idx2 = shuffled_idx[i + 1] if i + 1 < len(shuffled_idx) else None
        if idx2 is not None:
            if dijk_mat[idx1][idx2]:  # we can get from idx1 to idx2
                for j, user in enumerate(dijk_mat[idx1][idx2]):
                    if j + 1 == len(dijk_mat[idx1][idx2]):
                        link = get_link(G.links, user, G.users[idx2])
                    else:
                        next_user = dijk_mat[idx1][idx2][j + 1]
                        link = get_link(G.links, user, next_user)
                    if link is not None:
                        G.users[idx1].AddlinksInRoutes(link)
                        link.Adduser(G.users[idx1])
                    else:
                        print("Link is None!!!!")


# Template for run:
# python.exe .\Network_security_project.py <options>
'''
Arguments for program:
-N - the number of users                                  10 by default
-M - the radius of the world                              50 by default
-r - the maximus radius between two connected users       8  by default
-q - the question that we want to test question           4  by default
-alpha - for alpha fairness the alpha                     1  by default
-flows - number of flow to randomize                       N  by default
'''


def main():
    # Default values
    question = 5
    alpha = 1
    N = 10
    M = 50
    r = 8
    flowsNum = N
    random.seed(15)
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
        if arg.startswith("-flows"):
            flowsNum = int(arg[6:])

    if question == 4:
        q4(alpha)
    if question == 5:
        q5(N, M, r, alpha, flowsNum)
    if question == 6:
        q6(N, M, r, alpha, flowsNum)


if __name__ == "__main__":
    main()
