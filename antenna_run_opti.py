from necpp import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

class GeneticAlgorithm:
    popLength = 0
    nbNewIndiv = 0
    mutationRate = 0
    nbMutationAttempts = 0
    overlap = 0
    (frequency, thickness, nb_segments, nb_wires, distFromFloor, firstWireLength) = (0, 0, 0, 0, 0, 0)

    population = []
    
    def __init__(self, popLength, nbNewIndiv, mutationRate, nbMutationAttempts, overlap, requirements):
        self.popLength = popLength
        self.nbNewIndiv = nbNewIndiv
        self.mutationRate = mutationRate
        self.nbMutationAttempts = nbMutationAttempts
        self.overlap = overlap
        (self.frequency, self.thickness, self.nb_segments, self.nb_wires, self.distFromFloor, self.firstWireLength) = requirements

        k = 0
        while k < popLength:
            indiv = self.createIndiv()
            if(indiv != 0):
                self.population.append(indiv)
                k += 1

    def handle_nec(self, result):
        if(result != 0):
            #print(nec_error_message())
            return 0

    def createIndiv(self, shape = 0):
            if(shape == 0):
                shape = self.generateStructure()
                
            nec = nec_create()
            wireTag = 0
            for j in range(0, len(shape)-3, 3):
                wireTag += 1
                x = float(shape[j:j+3][0])
                y = float(shape[j:j+3][1])
                z = float(shape[j:j+3][2])
                xp = float(shape[j+3:j+6][0])
                yp = float(shape[j+3:j+6][1])
                zp = float(shape[j+3:j+6][2])
                
                createWire = self.handle_nec(nec_wire(nec, wireTag, self.nb_segments, x, y, self.distFromFloor + z, xp, yp, self.distFromFloor + zp, self.thickness, 1.0, 1.0))
                if createWire == 0:
                    #print(j)
                    return 0
                
            if(self.handle_nec(nec_geometry_complete(nec, 1)) != 0):
                self.handle_nec(nec_gn_card(nec, 1, 0, 0, 0, 0, 0, 0, 0)) # 1:PerfectGround; -1:FreeSpace
                self.handle_nec(nec_excitation_voltage(nec, 0, 1, 1, 0))
                self.handle_nec(nec_fr_card(nec, 0, 1, self.frequency, 0))
                self.handle_nec(nec_xq_card(nec, 0))
                #self.handle_nec(nec_rp_card(nec, 0, 90, 360, 1, 5, 1, 0, 1, 1, 1, 1, 0, 0))
                
                cost = self.costFunction(nec)
                VSWR = self.getVSWR(nec)
                self.handle_nec(nec_delete(nec))
                return [VSWR, cost, shape]
            else:
                return 0                

    def start(self, nb_iterations):
        print('Starting genetic algorithm...')
        for k in range(nb_iterations):
            print('Loop : ' + str(k))
            self.mutation()
            self.sort()
            self.crossOver()
            self.fillPopulation()
            self.sort()
            if k%10 == 0:
                self.stats()

        print('  ')
        print('End') 
                
    def sort(self):
        self.population = sorted(self.population, key = lambda col: col[1])

    def mutation(self):
        for k in range(self.popLength):
            p = np.random.uniform(0, 100)
            if(p < self.mutationRate * 100 and self.nb_wires > 1):
                newIndivShape = np.copy(self.population[k][2]).tolist()
                oldCost = np.copy(self.population[k][1])
                ok = False
                nbMutationAttempts = self.nbMutationAttempts
                flagMutation = 0
                while ok == False:
                    i = np.random.random_integers(7, 3*(self.nb_wires + 1) - 1)
                    if (i-2)%3 == 0:
                        v = np.random.uniform(self.firstWireLength, self.firstWireLength + 0.08)
                    else:
                        v = np.random.uniform(-0.03, 0.03)
                    newIndivShape[i] = v
                    finalIndiv = self.createIndiv(newIndivShape)
                    if(finalIndiv != 0):
                        newCost = finalIndiv[1]
                        if newCost < oldCost:
                            self.population[k] = finalIndiv
                            ok = True
                            flagMutation = 0
                            #print('Antenna has been enhanced')
                        elif flagMutation == nbMutationAttempts:
                            self.population[k] = finalIndiv
                            ok = True
                            flagMutation = 0
                            #print('Antenna has been ramdomly mutated')
                        else:
                            flagMutation += 1

    def crossOver(self):
        del self.population[int(self.popLength*self.overlap):]
        while len(self.population) < self.popLength - self.nbNewIndiv:
            #n1 = np.random.random_integers(0, len(self.population)-1)
            n1 = 0
            n2 = np.random.random_integers(0, len(self.population)-1)
            while(n2 == n1):
                n2 = np.random.random_integers(0, len(self.population)-1)

            p1 = self.population[n1]
            p2 = self.population[n2]

            ok = False
            while ok == False:
                if(self.nb_wires > 1):
                    x = np.random.random_integers(7, len(p1[2])-1)
                    childShape = np.copy(p1[2][:x]).tolist() + np.copy(p2[2][x:]).tolist()
                    
                    child = self.createIndiv(childShape)
                    
                    if(child != 0):
                        self.population.append(child)
                        ok = True
                else:
                    childShape = np.copy(p1[2]).tolist()
                    child = self.createIndiv(childShape)
                    if(child != 0):
                        self.population.append(child)
                        ok = True

    def fillPopulation(self):
        while len(self.population) < self.popLength:
            indiv = self.createIndiv()
            if(indiv != 0):
                self.population.append(indiv)
                
    def generateStructure(self):
        shape = [0, 0, 0, 0, 0, self.firstWireLength]
        for k in range(1, self.nb_wires):
            shape.append(np.random.uniform(-0.03, 0.03))
            shape.append(np.random.uniform(-0.03, 0.03))
            shape.append(np.random.uniform(self.firstWireLength, self.firstWireLength + 0.1))
        
        return shape

    def costFunction(self, nec):
        f_VSWR = self.getVSWR(nec)
        return (1 - f_VSWR)**2

    def getVSWR(self, nec):
        Z0 = 50
        z = complex(nec_impedance_real(nec, 0), nec_impedance_imag(nec, 0))
        gamma = (z - Z0)/(z + Z0)
        VSWR = (1 + np.abs(gamma))/(1 - np.abs(gamma))
        f_VSWR = VSWR
        return f_VSWR
    
    def displayStructure(self, shape):
        x = []
        y = []
        z = []
        for k in range(0, len(shape), 3):
            x.append(shape[k:k+3][0])
            y.append(shape[k:k+3][1])
            z.append(shape[k:k+3][2])

        fig = plt.figure()
        ax = Axes3D(fig)
        cset = ax.plot(x, y, z, zdir='z')
        ax.clabel(cset, fontsize=9, inline=1)
        plt.show()

    def stats(self):
        print('  ')
        print('-----')
        print('  ')
        for k in range(len(self.population)):
            print("Cost Result :", self.population[k][1], "- VSWR :", self.population[k][0], "- Antenna :", np.round(self.population[k][2], 3).tolist())
        
popLength = 20
mutationRate = 0.05
nbMutationAttempts = 3000
overlap = 0.3
nbNewIndiv = 10

frequency = 80
thickness = 0.002
nb_segments = 4
nb_wires = 6
distFromFloor = 0
firstWireLength = 0.02

requirements = (frequency, thickness, nb_segments, nb_wires, distFromFloor, firstWireLength)

nb_iterations = 10000

geneAlgo = GeneticAlgorithm(popLength, nbNewIndiv, mutationRate, nbMutationAttempts, overlap, requirements)
geneAlgo.start(nb_iterations)
