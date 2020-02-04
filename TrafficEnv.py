import numpy as np

import os, sys

# sumo_dir = 'C:\Program Files (x86)\Eclipse\Sumo'
# tools = os.path.join(sumo_dir, 'tools')
# sys.path.append(tools)

import traci


class TrafficEnv:

    def __init__(self, mode):

        # ------ Define SUMO directories
        if str(mode) == 'gui':
            self.sumoBinary = "F:/SUMO/bin/sumo-gui"
            # self.sumoBinary = "F:/SUMO/bin/sumo-gui"
        else:
            self.sumoBinary = "F:/SUMO/bin/sumo.exe"
            # self.sumoBinary = "F:/SUMO/bin/sumo.exe"
        self.cfgBinary = "tutorial.sumocfg"
        self.sumoCmd = [self.sumoBinary, "-c", self.cfgBinary]

        # ------ Simulation setup
        ##-- Geometric properties
        self.edgeId = ["3to1", "5to1", "4to1", "2to1"]
        self.outEdgeId = ["1to5", "1to3", "1to2", "1to4"]
        self.nodeId = ['1']
        self.numLanes = [3, 3, 3, 3]
        self.linkLength = [500, 500, 500, 500]
        self.lastQVeh = [[], [], [], []]
        self.inVehIDs = [[], [], [], []]
        self.outVehIDs = [[], [], [], []]

        ##-- Signal setting
        self.cycle = 90
        self.initialProgram = []
        self.signalProgram = ['action-0.2', 'action-0.4', 'action-0.6', 'action-0.8']
        self.currentProgram = []

        ##-- Simulation time setup
        self.timestep = None
        # self.warmupTime = 10
        self.warmupTime = 360

        ##-- Traffic Parameters
        self.linkCapacityPerLane = 1680
        self.linkFreeSpeed = [50, 50, 50, 50]

        ##-- Reinforcement Learning parameters
        self.stateDim = 4
        self.periodicState = np.zeros((self.cycle, self.stateDim))
        self.state = None

    def startSUMO(self):
        traci.start(self.sumoCmd)
        self.timestep = -1

    def endSUMO(self):
        traci.close()
        self.timesetp = -1

    def stepSimulation(self):
        '''
        proceed only one timestep of the simulation
        '''
        traci.simulationStep()
        self.timestep += 1

    def reset(self):
        '''
        Initialize the simulation with 4-mins warm-up session
        Get current waiting queues (# of waiting vehicles)
        '''
        if self.timestep == -1:
            warmupState = np.zeros((self.warmupTime, self.stateDim))
            for t in range(self.warmupTime):
                self.stepSimulation()
                warmupState[t, :] = self.getInstantQs()

                if self.timestep == (self.cycle - 1):
                    self.setSignalProgram()
                    traci.trafficlight.setProgram(self.nodeId[0], 'Initial')
                    self.currentProgram = traci.trafficlight.getProgram(self.nodeId[0])

            self.state = self.getMaxQs(warmupState[(self.cycle - 1):, ])

        else:
            print("Error! You may close the current simulation.")

        self.setSignalProgram()

        return self.state

    def step(self, action):
        '''
        Proceed one step of Traffic simulation with given action
        It gives 'next state', 'reward', 'terminal'
        '''
        self.setAction(action)

        prdState = np.copy(self.periodicState)

        for t in range(self.cycle):

            # -- Proceeding one timestep
            self.stepSimulation()

            # -- Get Instant Queues regarding to the signal phase
            phase = traci.trafficlight.getPhase(self.nodeId[0])
            tmpQ = self.getInstantQs()
            if phase == 1:
                tmpQ[2:4] = [-1, -1]
            elif phase == 3:
                tmpQ[0:2] = [-1, -1]
            prdState[t, :] = tmpQ

            # -- Get outflow vehicles
            self.getDetectingInfo()

        # -- Get Next state
        nextState = self.getMaxQs(prdState)

        # Get terminal
        terminal = self.getTerminal(nextState)

        # Get reward
        if terminal:
            reward = -1
        else:
            reward = self.getReward()
        self.outVehIDs = [[], [], [], []]

        # Updating the currentState
        self.state = nextState

        return nextState, reward, terminal

    def setSignalProgram(self):
        '''
        action is the green time ratio of N-S direction (0.2, 0.4, 0.6, 0.8)
        cycle is fixed to be 120 (sec)
        yellow time is set up to be 3 sec in default
        and offset is set to be zero
        '''
        # -- Initial Random signal
        rdmRatio = round(np.random.uniform(0.2, 0.8, 1)[0], 1)
        greentime_NS = int(round(self.cycle * rdmRatio * 1000)) - 3000
        greentime_EW = int(self.cycle * 1000 - (greentime_NS + 3000)) - 3000

        phase = []
        phase.append(traci.trafficlight.Phase(3000, 0, 0, "rrryyyrrryyy"))  # phase : 3 (EW-yellow)
        phase.append(traci.trafficlight.Phase(greentime_NS, 0, 0, "GGGrrrGGGrrr"))  # phase : 0 (NS-green)
        phase.append(traci.trafficlight.Phase(3000, 0, 0, "yyyrrryyyrrr"))  # phase : 1 (NS-yellow)
        phase.append(traci.trafficlight.Phase(greentime_EW, 0, 0, "rrrGGGrrrGGG"))  # phase : 2 (EW-green)

        programId = 'Initial'
        logic = traci.trafficlight.Logic(programId, 0, 0, 0, phase)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.nodeId[0], logic)
        self.initialProgram = ''.join(['action-', str(rdmRatio)])

        for action in range(0, 4):
            gr = round((action + 1) * 0.2, 1)
            greentime_NS = int(round(self.cycle * gr * 1000)) - 3000  # Phase number : 0 // yellow time : 1
            greentime_EW = int(self.cycle * 1000 - (greentime_NS + 3000)) - 3000  # Phase number : 2 // yellow time : 3

            phase = []
            phase.append(traci.trafficlight.Phase(3000, 0, 0, "rrryyyrrryyy"))  # phase : 3 (EW-yellow)
            phase.append(traci.trafficlight.Phase(greentime_NS, 0, 0, "GGGrrrGGGrrr"))  # phase : 0 (NS-green)
            phase.append(traci.trafficlight.Phase(3000, 0, 0, "yyyrrryyyrrr"))  # phase : 1 (NS-yellow)
            phase.append(traci.trafficlight.Phase(greentime_EW, 0, 0, "rrrGGGrrrGGG"))  # phase : 2 (EW-green)

            programId = ''.join(['action-', str(gr)])
            logic = traci.trafficlight.Logic(programId, 0, 0, 0, phase)
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.nodeId[0], logic)

    def getInstantQs(self):
        '''
        Return waiting queues of each timestep (# of vehicles)
        '''
        edgeQs = []
        for i in range(0, len(self.edgeId)):
            edge = self.edgeId[i]
            edgeVehIds = traci.edge.getLastStepVehicleIDs(edge)
            edgeQ = 0
            if len(edgeVehIds) != 0:
                for idv in reversed(edgeVehIds):
                    vehPos = traci.vehicle.getLanePosition(idv)
                    vehSpd = traci.vehicle.getSpeed(idv)

                    if vehPos > 100 and vehSpd < 8.3:
                        edgeQ += 1
                        self.lastQVeh[i] = idv

            edgeQs.append(edgeQ)

        return edgeQs

    def getMaxQs(self, periodicState):
        '''
        Return maximum column element from (timestep * stateDim) matrix
        '''
        maxQs = []
        for i in range(self.stateDim):
            maxQ = np.max(periodicState[:, i])
            maxQs.append(maxQ)

        return maxQs

    def setAction(self, action):
        '''
        Set signal program which is predefined in self.setSignalProgram
        '''
        program = self.signalProgram[action]
        traci.trafficlight.setProgram(self.nodeId[0],
                                      program)  # '1' is the node name located on the center of the intersection

        # -- Get the current Signal program information
        self.currentProgram = traci.trafficlight.getProgram(self.nodeId[0])

    def getReward(self):
        '''
        reward is defined by the throughput according to the signal action
        The throughput is normalized with the link capacity
        '''
        score = 0

        for i in range(len(self.edgeId)):
            capa = self.linkCapacityPerLane * self.numLanes[i] * self.cycle / 3600
            counting = len(list(set(self.outVehIDs[i])))

            score += counting / capa

        if score > 1:
            score = 1

        return score

    def getTerminal(self, nextState):

        min_state = min(nextState)
        max_state = max(nextState)

        if max_state - min_state >= 40:
            terminal = True
        else:
            terminal = False

        return terminal

    def getDetectingInfo(self):
        '''
        This function is working like a loop detector at upstream and downstream respectively.
        When calling this function, the detecting information is saved on self.inVehIDs and self.outVehIDs both
        '''
        for i in range(len(self.outEdgeId)):
            inLink = self.edgeId[i]
            outLink = self.outEdgeId[i]

            inIDlist = traci.edge.getLastStepVehicleIDs(inLink)
            outIDlist = traci.edge.getLastStepVehicleIDs(outLink)

            for idv in inIDlist:
                vehPos = traci.vehicle.getLanePosition(idv)

                if vehPos < 15:
                    self.inVehIDs[i].append(idv)

            for idv in outIDlist:
                vehPos = traci.vehicle.getLanePosition(idv)

                if vehPos < 15:
                    self.outVehIDs[i].append(idv)
