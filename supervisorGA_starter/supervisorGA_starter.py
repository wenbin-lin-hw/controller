from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy,struct
import ga,os
import sys
import numpy as np


class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.time_step = 32 # ms
        self.time_experiment = 150 # s
        
        # Initiate Supervisor Module
        self.supervisor = Supervisor()

        # time_step = int(gaModel.supervisor.getBasicTimeStep())


        # Check if the robot node exists in the current world file
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the robots translation and rotation current parameters    
        self.trans_field = self.robot_node.getField("translation")  
        self.rot_field = self.robot_node.getField("rotation")
        
        # Check Receiver and Emitter are enabled
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        # Initialize the receiver and emitter data to null
        self.receivedData = "" 
        self.receivedWeights = "" 
        self.receivedFitness = "" 
        self.emitterData = ""
        self.current_generation=0
        
        ### Define here the GA Parameters
        self.num_generations = 120
        self.num_population = 60
        self.num_elite = 6
        
        # size of the genotype variable
        self.num_weights = 0
        
        # Creating the initial population
        self.population = []
        
        # All Genotypes
        self.genotypes = []
        self.real_speed = 0.0
        
        # Display: screen to plot the fitness values of the best individual and the average of the entire population
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0;
        self.prev_average_fitness = 0.0;
        self.display.drawText("Fitness (Best - Red)", 0,0)
        self.display.drawText("Fitness (Average - Green)", 0,10)
        self.position_history = []

    def detect_circles(self, close_threshold=0.05, min_circle_len=2):
        """
        从一系列顺序点中检测出机器人画出的圈，并计算每个圈的周长。

        参数:
          points: list of [x, z] 坐标（顺序的轨迹点）
          close_threshold: 判定“回到起点”的距离阈值（单位: m）
          min_circle_len: 每圈最小路径长度，避免噪声误判（单位: m）

        返回:
          circles: list，每个元素是 (圈的周长, 该圈的起止索引)
        """
        points = self.position_history
        circles = []
        start_idx = 0
        accumulated_len = 0.0

        for i in range(1, len(points)):
            p0 = np.array(points[i - 1])
            p1 = np.array(points[i])
            step_dist = np.linalg.norm(p1 - p0)
            accumulated_len += step_dist

            # 判断是否回到当前圈的起点附近
            if np.linalg.norm(p1 - np.array(points[start_idx])) < close_threshold and accumulated_len > min_circle_len:
                # 计算该圈的总路径长度
                circle_points = points[start_idx:i + 1]
                circle_len = np.sum(np.linalg.norm(np.diff(circle_points, axis=0), axis=1))
                circles.append((circle_len, (start_idx, i)))

                # 更新起点
                start_idx = i
                accumulated_len = 0.0

        return circles

    def createRandomPopulation(self):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if(self.num_weights > 0):
            # Define the size of the population
            pop_size = (self.num_population,self.num_weights)
            # Create the initial population with random weights
            self.population = numpy.random.uniform(low=-1.0, high=1.0, size=pop_size)

    def handle_receiver(self):
        while(self.receiver.getQueueLength() > 0):
            # Webots 2022: 
            # self.receivedData = self.receiver.getData().decode("utf-8")
            # Webots 2023: 
            self.receivedData = self.receiver.getString()
            typeMessage = self.receivedData[0:7]
            # print(self.receivedData)
            # Check Message 
            if(typeMessage == "weights"):
                self.receivedWeights = self.receivedData[9:len(self.receivedData)] 
                self.num_weights = int(self.receivedWeights)
            elif(typeMessage == "fitness"):  
                self.receivedFitness = float(self.receivedData[9:len(self.receivedData)])
            self.receiver.nextPacket()
        
    def handle_emitter(self):
        if(self.num_weights > 0):
            # Send genotype of an individual
            string_message = "genotype: "+str(self.emitterData)
            string_message = string_message.encode("utf-8")
            #print("Supervisor send:", string_message)
            self.emitter.send(string_message)
        self.emitter.send("current_generation: {}".format(self.current_generation).encode("utf-8"))
        self.emitter.send("num_generations: {}".format(self.num_generations).encode("utf-8"))
        v=self.robot_node.getVelocity()  # ✅ 返回长度为6的list
        self.real_speed = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        self.emitter.send("real_speed: {}".format(self.real_speed).encode("utf-8"))
        pos = self.robot_node.getPosition()
        self.position_history.append(([pos[0],pos[1]]))
        # print("Position:",pos)
        self.emitter.send("position: {} ".format([pos[0],pos[1],pos[2]]).encode("utf-8"))
        # print("position: {} ".format([pos[0],pos[1],pos[2]]))




        
    def run_seconds(self,seconds):
        #print("Run Simulation")
        stop = int((seconds*1000)/self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if(stop == iterations):
                break    
            iterations = iterations + 1
                
    def evaluate_genotype(self,genotype,generation):
        # Send genotype to robot for evaluation
        self.emitterData = str(genotype)
        
        # Reset robot position and physics
        INITIAL_TRANS = [0.47, 0.16, 0]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 0, 1, 1.57]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment)
    
        # Measure fitness
        fitness = self.receivedFitness
        # print("Fitness: {}".format(fitness))
        current = (generation,genotype,fitness)
        self.genotypes.append(current)  
        
        return fitness

    def run_demo(self):
        # Read File
        genotype = numpy.load("Best.npy")
        # Send Genotype to controller
        self.emitterData = str(genotype) 
        
        # Reset robot position and physics
        INITIAL_TRANS = [4.48, 0, 7.63]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 1, 0, -0.0]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment)    
    
    def run_optimization(self):
        # Wait until the number of weights is updated
        while(self.num_weights == 0):
            self.handle_receiver()
            self.createRandomPopulation()
        
        print("starting GA optimization ...\n")
        
        # For each Generation
        for generation in range(self.num_generations):
            print("Generation: {}".format(generation))
            current_population = []
            self.current_generation = generation
            # Select each Genotype or Individual
            for population in range(self.num_population):
                self.position_history = []
                genotype = self.population[population]

                # print("  Individual: {}".format(population))
                # Evaluate
                fitness = self.evaluate_genotype(genotype,generation)
                circles = self.detect_circles()
                # print(circles)
                for (length,(start_idx,end_idx)) in circles:
                    if length/4.0>0.8 and length/4.0<1.2:
                        fitness += 0.1
                        break

                print(population,fitness)
                # Save its fitness value
                current_population.append((genotype,float(fitness)))
                #print(current_population)
                
            # After checking the fitness value of all indivuals
            # Save genotype of the best individual
            best = ga.getBestGenotype(current_population);
            average = ga.getAverageGenotype(current_population);
            numpy.save("Best.npy",best[0])
            self.plot_fitness(generation, best[1], average);
            
            # Generate the new population using genetic operators
            if (generation < self.num_generations - 1):
                self.population = ga.population_reproduce(current_population,self.num_elite);
        
        #print("All Genotypes: {}".format(self.genotypes))
        print("GA optimization terminated.\n")   
    
    
    def draw_scaled_line(self, generation, y1, y2): 
        # Define the scale of the fitness plot
        XSCALE = int(self.width/self.num_generations);
        YSCALE = 100;
        self.display.drawLine((generation-1)*XSCALE, self.height-int(y1*YSCALE), generation*XSCALE, self.height-int(y2*YSCALE));
    
    def plot_fitness(self, generation, best_fitness, average_fitness):
        if (generation > 0):
            self.display.setColor(0xff0000);  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness);
    
            self.display.setColor(0x00ff00);  # green
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness);
    
        self.prev_best_fitness = best_fitness;
        self.prev_average_fitness = average_fitness;
    
if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   
    gaModel = SupervisorGA()

    
    # Function used to run the best individual or the GA
    keyboard = Keyboard()
    keyboard.enable(50)
    
    # Interface
    print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if(resp == 83 or resp == 65619):
            gaModel.run_optimization()
            print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        elif(resp == 82 or resp == 65619):
            gaModel.run_demo()
            print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        