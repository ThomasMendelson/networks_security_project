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
from heapq import heappush, heappop, _heappop_max
import numpy as np
import queue
import pulp


def DefaultCostFunc(sigXr, cap, alpha):
    if cap >= sigXr:
        return 0
    return math.exp(alpha * sigXr)


def get_link(links_set, fromUser, toUser):
    for link in links_set:
        if link.GetFromUser() == fromUser and link.GetToUser() == toUser:
            return link
    return None  # If the link is not found


def copy_array(arr):
    new_arr = []
    for i in arr:
        new_arr.append(i)
    return new_arr


def GetFlowByName(flows, flowName):
    for flow in flows:
        if flowName == flow.Getname():
            return flow
    return None


def DefaultStepSizeFunc(tp, alpha):
    if tp == "dual":
        return 0.01
    else:
        if alpha < 3:
            return 0.001
        if alpha < 5:
            return 0.00001
        if alpha < 7:
            return 0.000001
        else:
            return 1 / (10 ** (alpha))


class Flow:
    nameOfFlowWithinClass = 0

    def __init__(self, destUser, srcUser, pktSize, K=1):
        self.destUser = destUser
        self.srcUser = srcUser
        self.pktSize = pktSize
        self.links = None
        self.name = Flow.nameOfFlowWithinClass
        Flow.nameOfFlowWithinClass += 1
        self.Xr = 1
        srcUser.AddFlow(self)
        self.K = K
        self.linkChannels = {}

    def Getname(self):
        return self.name

    def GetSource(self):
        return self.srcUser

    def GetDest(self):
        return self.destUser

    def GetpktSize(self):
        return self.pktSize

    def SetLinks(self, links):
        self.links = links

    def GetLinks(self):
        return self.links

    def printFlow(self):
        print(f"Flow number : {self.name}")
        print(f"Source : {self.srcUser.Getname()}")
        print(f"Destination : {self.destUser.Getname()}")
        print(f"pkt size : {self.pktSize}")
        print("links in flow:")
        for link in self.links:
            print(f"{link.Getname()} ", end="")
        print("")

    def SetXr(self, xr):
        self.Xr = xr

    def GetXr(self):
        return self.Xr

    def AddChannelToLink(self, link, k):
        self.linkChannels[link] = k

    def GetK(self, link):
        return self.linkChannels[link]


class Link:
    nameOfLinkWithinClass = 0

    def __init__(self, fromUser, toUser, capacity=None, lambd=0.20633, scale=np.sqrt(0.5)):
        self.fromUser = fromUser
        self.toUser = toUser
        self.flows = set()  # that using this link
        self.lambd = lambd
        self.scale = scale
        self.gain = self.CalcGain(self.scale)
        self.wantedCapacity = capacity
        self.capacity = None
        self.name = Link.nameOfLinkWithinClass
        Link.nameOfLinkWithinClass += 1

    def initiateCapacity(self):
        if self.wantedCapacity is not None:
            self.capacity = self.wantedCapacity
        else:
            self.capacity = self.CalcCapacity()

    def GetFromUser(self):
        return self.fromUser

    def GetToUser(self):
        return self.toUser

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
        if self.fromUser.HasLocation() and self.toUser.HasLocation():
            smallScaleFading = np.random.rayleigh(scale=scale, size=1)
            dist = self.fromUser.Dist(self.toUser)
            gain = 1 / (smallScaleFading * np.sqrt(dist + (4 * np.pi / 3e8)))
            return np.abs(gain)
        else:
            return None

    def CalcCapacity(self):
        if self.gain is not None:
            userFromBw = self.fromUser.GetBw()
            userFromPower = self.fromUser.GetPower() * self.gain
            userToInterference2 = self.toUser.GetSumInterference()
            SINR = userFromPower / (userToInterference2 + 1)
            if hasattr(userFromBw, '__iter__'):
                userFromBw = float(userFromBw[0])
            if hasattr(SINR, '__iter__'):
                SINR = float(SINR[0])
            return userFromBw * math.log2(1 + SINR)
        else:
            return 1

    def GetTDMAVal(self, flow):
        sumOfData = 0
        for f in self.flows:
            if flow.GetK(self) == f.GetK(self):
                sumOfData += f.GetpktSize()
        return self.capacity * (flow.GetpktSize() / sumOfData)

    def GetFlows(self):
        return self.flows

    def printLink(self):
        print(f"printing Link num {self.name} (nameOfLinkInClass)")
        print(f"Link is From user : {self.fromUser.Getname()} to user : {self.toUser.Getname()}")


class User:
    nameOfUserWithinClass = 0

    def __init__(self, M=None, x_pos=None, y_pos=None, powerFunc=lambda: 10, bwFunc=lambda: 1):
        self.linkFromUser = set()
        self.linkToUser = set()
        self.flowsFromMe = set()
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

    def CanGetToUser(self, friend):
        for link in self.linkFromUser:
            if link.GetToUser() == friend:
                return True
        return False

    def UserCanGetToMe(self, friend):
        for link in self.linkToUser:
            if link.GetFromUser() == friend:
                return True
        return False

    def AddLinkTo(self, friend, capacity=None):
        link = Link(self, friend, capacity=capacity)
        self.linkFromUser.add(link)
        friend.linkToUser.add(link)
        return link

    def AddFlow(self, flow):
        self.flowsFromMe.add(flow)

    def HasNoFlows(self):
        return 0 == len(self.flowsFromMe)

    def Getname(self):
        return self.name

    def HasLocation(self):
        return (self.x_pos is not None) and (self.y_pos is not None)

    def printUser(self):
        print(f"printing User num {self.name} (nameOfLinkInClass)")
        if self.HasLocation():
            print(f"x pos: {self.x_pos}\ny pos: {self.y_pos}")
        print(f"User is Has Going Out Links:")
        for link in self.linkFromUser:
            print(f"link : {link.Getname()} (nameOfUserInClass)")
        print(f"User is Has Going In Links:")
        for link in self.linkToUser:
            print(f"link : {link.Getname()} (nameOfUserInClass)")
        print(f"User flows:")
        for flow in self.flowsFromMe:
            print(f"{flow.Getname()} ", end="")

    def SetSumInterference(self, sumInterference):
        self.interference = sumInterference

    def GetSumInterference(self):
        return self.interference


class Graph:

    def __init__(self, N, alpha, M=None, r=None, users=None, flows=None, links=None, costFunc=DefaultCostFunc,
                 stepSize=DefaultStepSizeFunc, maxIterSteps=None, interferenceFunc=lambda: 0, K=1):
        self.N = N
        self.M = M
        self.r = r
        if users is None:
            users = []
        self.users = users
        if flows is None:
            flows = []
        self.flows = flows
        if links is None:
            links = set()
        self.links = links
        self.alpha = alpha
        self.stepSize = stepSize
        self.maxIterSteps = maxIterSteps
        self.costFunc = costFunc
        if not users:
            self.CreateRandomGraph()
        self.CalcInterference(interferenceFunc)
        self.CreateCapacities()
        self.K = K
        self.dijkMat = None
        self.maxIterSteps = self.FindNumIterations()

    def FindNumIterations(self):
        return max(100000, int(5000 * (2 ** self.alpha)))

    def CreateRandomGraph(self):
        for i in range(self.N):
            self.users.append(User(M=self.M))
        for i, user in enumerate(self.users):
            for j, friend in enumerate(self.users):
                if user != friend:
                    if user.Dist(friend) < self.r:
                        link = user.AddLinkTo(friend=friend)
                        self.links.add(link)

    def CreateCapacities(self):
        for link in self.links:
            link.initiateCapacity()

    def CalcInterference(self, func):
        for user in self.users:
            if user.HasLocation():
                sumInterference = 0
                for friend in self.users:
                    if user == friend:
                        continue
                    sumInterference += user.Dist(friend) * func()
                user.SetSumInterference(sumInterference)
            else:
                user.SetSumInterference(0)

    def GetUsers(self):
        return self.users

    def resetXrs(self, val=1):
        for flow in self.flows:
            flow.SetXr(val)

    def AddFlow(self, flow):
        self.flows.append(flow)

    def Ur(self, flow):
        return (flow.GetXr()(1 - self.alpha)) / (1 - self.alpha)

    def DevUr(self, flow):
        return 1 / (flow.GetXr() ** self.alpha)

    def Kr(self):
        return self.stepSize("primal", self.alpha)

    def Hr(self):
        return self.stepSize("dual", self.alpha)

    def printGraph(self):
        print("users:")
        for i in range(len(self.users)):
            print(f"user {i}:")
            self.users[i].printUser()
        print("")

    def plotRun(self, xrsToPlot, tp):
        for i in range(len(xrsToPlot)):
            plt.plot(xrsToPlot[i], label=f"User {i}")
        plt.xlabel("Iteration")
        plt.ylabel("Xr")
        plt.title(f"Xr vs. Iteration for Each User {tp}")
        plt.legend()
        plt.show()

    def PrimalIterStep(self, chosenFlow):
        dUr = self.DevUr(chosenFlow)
        kr = self.Kr()
        SigCost = 0
        for l in chosenFlow.links:
            SigXr = 0
            for flow in l.flows:
                SigXr += flow.GetXr()
            SigCost += self.costFunc(SigXr, l.GetCapacity(), self.alpha)
        return kr * (dUr - SigCost)

    def DualIterStep(self, chosenFlow):
        Hr = self.Hr()
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
                    elif curr_user.CanGetToUser(friend):
                        distance_through_current = curr_dist + 1  # todo 1 or dist by coordinates
                        if distance_through_current < distances[idx_friend]:
                            distances[idx_friend] = distance_through_current
                            paths[idx_friend] = copy_array(paths[idx_curr_user])
                            paths[idx_friend].append(curr_user)
                            heappush(pq, (distance_through_current, idx_friend))
            all_shortest_paths.append(paths)
        self.dijkMat = all_shortest_paths
        return all_shortest_paths

    def CreatKminHeap(self):
        KminHeap = []
        for i in range(self.K):
            heappush(KminHeap, (0, i))
        return KminHeap

    def CreateFlowDataMaxHeap(self, link):
        flowDataMaxHeap = []
        for flow in link.GetFlows():
            heappush(flowDataMaxHeap, (flow.GetpktSize(), flow.Getname()))
        return flowDataMaxHeap

    def ConnectLinksToFlowsChannels(self):
        for link in self.links:
            KminHeap = self.CreatKminHeap()
            flowDataMaxHeap = self.CreateFlowDataMaxHeap(link)
            while flowDataMaxHeap:
                pktSize, flowName = _heappop_max(flowDataMaxHeap)
                sumData, k = heappop(KminHeap)
                flow = GetFlowByName(link.GetFlows(), flowName)
                if flow is not None:
                    flow.AddChannelToLink(link, k)
                sumData += pktSize
                heappush(KminHeap, (sumData, k))

    def runTDMA(self):
        self.ConnectLinksToFlowsChannels()
        xrsVec = [0] * len(self.flows)
        for i, flow in enumerate(self.flows):
            xrsInLinkForFlow = []
            for j, link in enumerate(flow.GetLinks()):
                xrsInLinkForFlow.append(link.GetTDMAVal(flow))
            xrsVec[i] = min(xrsInLinkForFlow)
        for i in range(len(xrsVec)):  # for debug
            print(f"flow's {i} Xr Value is: {xrsVec[i]}")  # for debug
        return xrsVec

    def run(self, tp):
        XrsToPlot = []
        for i, flow in enumerate(self.flows):
            XrsToPlot.insert(i, [])
            XrsToPlot[i].insert(0, flow.GetXr())
        for i in range(self.maxIterSteps):
            randIndex = random.randint(0, len(self.flows) - 1)
            chosenFlow = self.flows[randIndex]
            if tp == "Primal":
                chosenFlow.SetXr(chosenFlow.GetXr() + self.PrimalIterStep(chosenFlow))
            if tp == "Dual":
                chosenFlow.SetXr(self.DualIterStep(chosenFlow))
            for j, flow in enumerate(self.flows):
                XrsToPlot[j].append(flow.GetXr())
        for i, flow in enumerate(self.flows):  # for debug
            print(f"flow's {i} Xr Value is: {flow.GetXr()}")  # for debug
        self.plotRun(XrsToPlot, f"{tp} run")


class PaperFlow(Flow):
    def __init__(self, srcUser, detUser, pktSize):
        super().__init__(srcUser, detUser, pktSize)

    def SetPacketSize(self, newPktSize):
        self.pktSize = newPktSize


class PaperLink(Link):
    def __init__(self, fromUser, toUser):
        super().__init__(fromUser, toUser)
        self.weights = {}
        self.maxWeight = 0
        self.argMaxUser = None
        self.slotRate_Dest = None

    def GetWeight(self, user):
        return self.weights[user]

    def SetWeight(self, user, weight):
        self.weights[user] = weight

    def SetWeights(self, weights):
        self.weights = weights

    def SetMaxWeight(self, weight):
        self.maxWeight = weight

    def SetArgMaxUser(self, user):
        self.argMaxUser = user

    def GetMaxWeight(self):
        return self.maxWeight

    def GetArgMaxUser(self):
        return self.argMaxUser

    def SetSlotRate_Dest(self, bool):
        if bool:
            self.slotRate_Dest = (self.capacity, self.argMaxUser)
        else:
            self.slotRate_Dest = None

    def GetSlotRate_Dest(self):
        return self.slotRate_Dest


class PaperUser(User):
    def __init__(self, M):
        super().__init__(M=M)
        self.queues = {}
        self.R = {}

    def AddToQueue(self, user, flow):
        if user not in self.queues:
            self.queues[user] = queue.Queue()
        self.queues[user].put(flow)

    def PopFromQueue(self, user):
        if (user not in self.queues) or not isinstance(self.queues[user], queue.Queue) or self.queues[user].empty():
            return None
        return self.queues[user].get()

    def PeekFromQueue(self, user):
        if (user not in self.queues) or not isinstance(self.queues[user], queue.Queue) or self.queues[user].empty():
            return None
        return self.queues[user].queue[0]

    def GetQueueLen(self, user):
        if not isinstance(self.queues[user], queue.Queue):
            return 0
        return self.queues[user].qsize()

    def AddLinkTo(self, friend):
        link = PaperLink(self, friend)
        self.linkFromUser.add(link)
        friend.linkToUser.add(link)
        return link

    def SetR(self, R):
        self.R = R

    def GetR(self):
        return self.R


class PaperGraph(Graph):
    def __init__(self, N, M, r):
        super().__init__(N, alpha=1, M=M, r=r)
        self.CalcRs()

    def CreateRandomGraph(self):
        for i in range(self.N):
            self.users.append(PaperUser(M=self.M))
        for i, user in enumerate(self.users):
            for j, friend in enumerate(self.users):
                if user != friend:
                    if user.Dist(friend) < self.r:
                        link = user.AddLinkTo(friend=friend)
                        self.links.add(link)

    @staticmethod
    def h(x):
        return math.log(math.e + math.log(1 + x, math.e), math.e)

    @staticmethod
    def g(x):
        return math.log(1 + x, math.e) / PaperGraph.h(x)

    @staticmethod
    def RemoveCS(mapLinks, link):  # CS stands for Conflict Set
        fromUser = link.GetFromUser()
        toUser = link.GetFromUser()
        linksToDeleteFromMap = []
        for l in mapLinks:
            if l.GetFromUser() == fromUser or l.GetFromUser() == toUser or l.GetToUser() == fromUser or l.GetToUser() == toUser:
                linksToDeleteFromMap.append(l)
        for l in linksToDeleteFromMap:
            del mapLinks[l]

    def CalcWeigths(self):
        for link in self.links:
            fromUser = link.GetFromUser()
            toUser = link.GetToUser()
            maxWeight = 0
            argMaxUser = None
            for destUser in self.users:
                if destUser.GetR()[fromUser][toUser] == 1:
                    dWeight = PaperGraph.g(fromUser.GetQueueLen(destUser)) - PaperGraph.g(toUser.GetQueueLen(destUser))
                    link.SetWeight(destUser, dWeight)
                    if dWeight > maxWeight:
                        maxWeight = dWeight
                        argMaxUser = destUser
            link.SetMaxWeight(maxWeight)
            link.SetArgMaxUser(argMaxUser)

    def CalcRs(self):
        dijkMat = self.getDijkstraMat()
        for i, destUser in enumerate(self.users):
            R = {}
            for user1 in self.users:
                R[user1] = {}
                for user2 in self.users:
                    R[user1][user2] = 0
            for j, fromUser in enumerate(self.users):
                route = dijkMat[j][i]
                lastUser = fromUser
                for user in route:
                    R[lastUser][user] = 1
                    lastUser = user
            destUser.SetR(R)

    def CreateMapCapWeightAndSetRatesNone(self):
        mapToReturn = {}
        for link in self.links:
            link.SetSlotRate_Dest(False)
            mapToReturn[link] = link.GetCapacity() * link.GetMaxWeight()
        return mapToReturn

    """ we assume that the article is trying to solve the max weight problem for each time slot 
    i.e. each node can eiter send or receive data in each time slot and thus the max rate of
    each link at each time is the capacity of the link 
    """

    def FindRates(self):
        self.CalcWeigths()
        mapCapWeight = self.CreateMapCapWeightAndSetRatesNone()
        while mapCapWeight:
            MaxValInMap = max(mapCapWeight.items(), key=lambda x: x[1])
            maxValLink, maxValue = MaxValInMap
            del mapCapWeight[maxValLink]
            PaperGraph.RemoveCS(mapCapWeight, maxValLink)
            maxValLink.SetSlotRate_Dest(True)
            # TODO We dicided to solve it with greedy algorithm but its not optimal. Need to ask Kobi or maybe change

    def CreateRandomFlows(self, flowsToRand):
        for i in range(flowsToRand):
            flowNotCreated = True
            while flowNotCreated:
                source = random.randint(0, self.N - 1)
                dest = random.randint(0, self.N - 1)
                if not self.dijkMat[source][dest]:
                    continue
                pkt_size = 0.5 + random.randint(1, 10) * 0.3
                newFlow = PaperFlow(self.users[source], self.users[dest], pkt_size)
                flowNotCreated = False
                newFlow.GetSource().AddToQueue(newFlow.GetDest(),
                                               newFlow)  # adding the flow to the destination queue of the source user

    def SendPackets(self):
        for link in self.links:
            if link.GetSlotRate_Dest() is not None:
                rate, finalDest = link.GetSlotRate_Dest()
                sender = link.GetFromUser()
                receiver = link.GetToUser()
                notFinishSending = True
                while notFinishSending:
                    flowToSend = sender.PeekFromQueue(finalDest)
                    if flowToSend is not None:
                        if rate >= flowToSend.GetpktSize():  # in this case we can send the whole packet in one time
                            sender.PopFromQueue(finalDest)
                            if receiver != finalDest:
                                receiver.AddToQueue(finalDest, PaperFlow(receiver, finalDest, flowToSend.GetpktSize()))
                            rate = rate - flowToSend.GetpktSize()
                            if rate <= 0:
                                notFinishSending = False
                        else:  # in this case we need to split thw packet into two packets
                            flowToSend.SetPktSize(flowToSend.GetpktSize() - rate)
                            if receiver != finalDest:
                                receiver.AddToQueue(finalDest, PaperFlow(receiver, finalDest, rate))
                            notFinishSending = False
                    else:
                        notFinishSending = False


def q4(alpha):  # N = L + 1 = 5 +1
    numOfLinks = 5
    users = []
    links = set()
    flows = []
    # create empty users
    for i in range(numOfLinks + 1):
        users.append(User())

    flow0 = []
    # create empty flows
    flows.append(Flow(users[0], users[5], 1))
    for i in range(numOfLinks):
        flows.append(Flow(users[i], users[i + 1], 1))

    # create links and connect to flows
    for i in range(numOfLinks):
        link = users[i].AddLinkTo(friend=users[i + 1])
        links.add(link)
        flow0.append(link)
        flows[i + 1].SetLinks([link])
        link.AddFlow(flows[0])
        link.AddFlow(flows[i + 1])
    flows[0].SetLinks(flow0)

    G = Graph(6, alpha, users=users, flows=flows, links=links)
    G.run("Primal")
    G.resetXrs()
    G.run("Dual")


def GetRandomFlow(N):
    source = random.randint(0, N - 1)
    while True:
        dest = random.randint(0, N - 1)
        if dest != source:
            break
    pkt_size = random.randint(1, 10) * 5
    return source, dest, pkt_size


def CreateGraphWithDijkstraRandomFlows(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 0, K=1):
    G = Graph(N=N, M=M, r=r, alpha=alpha, interferenceFunc=interferenceFunc, K=K)
    dijk_mat = G.getDijkstraMat()
    shuffled_idx = list(range(len(G.users)))
    random.shuffle(shuffled_idx)
    users = G.GetUsers()
    for i in range(0, flowsNum):
        source, dest, pkt_size = GetRandomFlow(len(G.users))
        if dijk_mat[source][dest]:  # we can get from idx1 to idx2
            flow = Flow(users[source], users[dest], pkt_size, K=K)
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
    return G


def q5(N, M, r, alpha, flowsNum):
    G = CreateGraphWithDijkstraRandomFlows(N, M, r, alpha, flowsNum)
    G.run("Primal")
    G.run("Dual")


def plotGraph(graph, runFor, title):
    plt.figure(figsize=(10, 6))  # Increasing plot size
    for i in range(len(graph)):
        plt.plot(range(len(graph[i])), graph[i], label=f'run for ={runFor} = {i + 1}*{runFor}')
    plt.xlabel('Flows')
    plt.ylabel('Rates')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def GraphVarNumUsers(N, M, r, alpha, flowsNum, interferenceFunc, K=1):
    dataToGraph = []
    for i in range(1, 11):
        G = CreateGraphWithDijkstraRandomFlows(i * N, M, r, alpha, flowsNum, interferenceFunc=interferenceFunc, K=K)
        dataToGraph.append(G.runTDMA())
    plotGraph(dataToGraph, "N", "Xr vs. Number of Users")


def GraphVarNumFlows(N, M, r, alpha, flowsNum, interferenceFunc, K=1):
    dataToGraph = []
    for i in range(1, 11):
        G = CreateGraphWithDijkstraRandomFlows(N, M, r, alpha, i * flowsNum, interferenceFunc=interferenceFunc, K=K)
        dataToGraph.append(G.runTDMA())
    plotGraph(dataToGraph, "Number of Flows", "Xr vs. Number of Flows")


def GraphVarRadiusM(N, M, r, alpha, flowsNum, interferenceFunc, K=1):
    dataToGraph = []
    for i in range(1, 11):
        G = CreateGraphWithDijkstraRandomFlows(N, i * M, r, alpha, flowsNum, interferenceFunc=interferenceFunc, K=K)
        dataToGraph.append(G.runTDMA())
    plotGraph(dataToGraph, "M", "Xr vs. Radius (M)")


def GraphVarRadiusR(N, M, r, alpha, flowsNum, interferenceFunc, K=1):
    dataToGraph = []
    for i in range(1, 11):
        G = CreateGraphWithDijkstraRandomFlows(N, M, i * r, alpha, flowsNum, interferenceFunc=interferenceFunc, K=K)
        dataToGraph.append(G.runTDMA())
    plotGraph(dataToGraph, "r", "Xr vs. Radius (r)")


def q6(N, M, r, alpha, flowsNum):
    GraphVarNumUsers(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1)
    GraphVarNumFlows(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1)
    GraphVarRadiusM(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1)
    GraphVarRadiusR(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1)


def q7(N, M, r, alpha, flowsNum, K=1):
    GraphVarNumUsers(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1, K=K)
    GraphVarNumFlows(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1, K=K)
    GraphVarRadiusM(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1, K=K)
    GraphVarRadiusR(N, M, r, alpha, flowsNum, interferenceFunc=lambda: 1, K=K)


def PartB(N, M, r, t, flowsNum):
    graph = PaperGraph(N=N, M=M, r=r)
    for i in range(t):
        graph.CreateRandomFlows(flowsNum)
        graph.FindRates()
        graph.SendPackets()


# Template for run:
# python.exe .\Network_security_project.py <options>
'''
Arguments for program:
-N - the number of users                                        10 by default
-M - the radius of the world                                    10 by default
-r - the maximus radius between two connected users             5  by default
-q - the question that we want to test question                 4  by default
-alpha - for alpha fairness the alpha                           1  by default
-flows - number of flow to randomize                            N  by default
-K - number of orthogonal channels                              1  by default
-part - the part we are running                                 a  by default
-t - in part b how many time slots the simulation will run      10 by default
'''


def main():
    # Default values
    part = "a"
    question = 5
    alpha = 1
    N = 10
    M = 10
    r = 5
    flowsNum = N
    K = 3
    t = 10
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
        if arg.startswith("-K"):
            K = int(arg[2:])
        if arg.startswith("-part"):
            part = arg[5:]
        if arg.startswith("-t"):
            t = int(arg[2:])

    if part == "a":
        if question == 4:
            q4(alpha)
        if question == 5:
            q5(N, M, r, alpha, flowsNum)
        if question == 6:
            q6(N, M, r, alpha, flowsNum)
        if question == 7:
            q7(N, M, r, alpha, flowsNum, K)

    if part == "b":
        PartB(N, M, r, t, flowsNum)


if __name__ == "__main__":
    main()
