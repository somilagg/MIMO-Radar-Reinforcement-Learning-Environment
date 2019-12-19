#import statements
import numpy as np
import random
import os
import csv
import math
import cmath
import array
import time
from scipy.special import iv

#sarsa object
class SARSA(object):

    #constructor
    def __init__(self, l_range, g_range, g, b, e, power, nt, start_a, end_a, sig, p, t, timesteps, tl, r_n, v):        
        
        #there are two systems - radar system and RL system
        #--------------------------------------------------#
        #variables for radar system
        self.l_r = l_range
        self.g_r = g_range
        self.start_angle = start_a
        self.end_angle = end_a
        self.sigma = sig
        self.pt = power
        self.nt = nt
        self.art = []
        self.thres = math.exp(p/2.0)
        self.l_angles = np.zeros((l_range), dtype=float)
        diff = end_a - start_a
        increment = diff/l_range
        for x in range(0, l_range):
            self.l_angles[x] = start_a + x * increment

        #variables for RL system
        self.curr_st = 0
        self.curr_action_index = 0
        self.next_st = 0
        self.next_action_index = 0
        self.gamma = g
        self.beta = b
        self.epsilon = e
        self.reward = 0.0
        self.env = None
        self.hyp_test = np.zeros((self.l_r, self.g_r), dtype=float)
        self.ac_opt = []
        self.t_max = t
        self.init_q_val = 0.0
        self.q_table = np.zeros((self.t_max+1, self.t_max+1), dtype=float)
        for y in range(0, len(self.q_table)):
            for z in range(0, len(self.q_table[0])):
                self.q_table[y][z] = self.init_q_val
        self.k = timesteps
        self.tar_loc = tl
        self.version = v
        self.rand_num = r_n
        self.matlab_gen = open('matlab_gen.txt','w')
        self.python_gen = open('python_gen.txt','w')

    #set current state
    def set_current_state(self, cs):
        self.curr_st = cs

    #set index of current action
    def set_current_action(self, ca):
        self.curr_action_index = ca

    #set next state
    def set_next_state(self, ns):
        self.next_st = ns

    #set index of next action
    def set_next_action(self, na):
        self.next_action_index = na

    def get_K(self):
        return self.k

    #reading var.txt for system parameters
    def read_file():
        
        #statements that read file
        var_file = open("var.txt","r")
        var_file.readline()
        l_range = int(var_file.readline())
        var_file.readline()
        g_range = int(var_file.readline())
        var_file.readline()
        power = float(var_file.readline())
        var_file.readline()
        nt = int(var_file.readline())
        var_file.readline()
        start_angle = float(var_file.readline())
        var_file.readline()
        end_angle = float(var_file.readline())
        var_file.readline()
        gamma = float(var_file.readline())
        var_file.readline()
        beta = float(var_file.readline())
        var_file.readline()
        epsilon = float(var_file.readline())
        var_file.readline()
        sigma = float(var_file.readline())
        var_file.readline()
        pfa = float(var_file.readline())
        var_file.readline()
        t_m = int(var_file.readline())
        var_file.readline()
        k_steps = int(var_file.readline())
        var_file.readline()
        version = int(var_file.readline())
        print("#: INPUT FILE READ")
        
        #based on version we choose how many targets
        if version == 1 or version == 2:
            rand_num = 4
        else:
            rand_num = random.randint(0, t_m)

        #writing input parameters for matlab file to read
        input_param = open("..\Desktop\input_param.txt",'w')
        input_param.write(str(power) + "\n")
        input_param.write(str(l_range) + "\n")
        input_param.write(str(nt) + "\n")
        input_param.write(str(start_angle) + "\n")
        input_param.write(str(end_angle) + "\n")
        input_param.write(str(g_range) + "\n")
        input_param.write(str(t_m) + "\n")
        input_param.write(str(rand_num) + "\n")
        input_param.write(str(version))
        input_param.close()
        base = float(power/(nt * nt))
        print("#: MATLAB INPUT FILE CREATED")

        #make other files
        target_locations = open('target_locations.txt', 'w')

        #make generic weight matrix
        C_main = np.zeros((nt, nt), dtype=float)
        for x in range(0, nt):
            for y in range(0, nt):
                C_main[x][y] = base
        obj = SARSA(l_range, g_range, gamma, beta, epsilon, power, nt, start_angle, end_angle, sigma, pfa, t_m, k_steps, target_locations, rand_num, version)

        #returns SARSA object with all the parameters and generic weight matrix
        return obj, C_main


    def retrieve_y_s_s(self):

        #runs the matlab script and collects y_rec
        os.system(r"cd ..\Desktop\\")
        os.system(r'matlab -batch "run generateY.m"')
        os.system('cp y_rec.csv ../mimo2/y_rec.csv')
        with open('y_rec.csv', 'rt') as f:
            reader = csv.reader(f)
            y_rec = list(reader)

        return y_rec

    def retrieve_y_s_d(self):

        #reads data from input file
        with open('../Desktop/input_param.txt', 'r') as file:
            data = file.readlines()

        #keeps lines 0-7
        new_data = data[0:9]
        new_data[8] = new_data[8] + "\n"
        print(new_data)

        #adds newly generated locations for same targets
        for x in range(0, self.rand_num):
            rnum = random.randint(0, self.l_r * self.g_r)
            new_data.append(str(rnum) + "\n")
        
        #rewrites new file back
        with open('../Desktop/input_param.txt', 'w') as file:
            file.writelines(new_data)

        #runs the matlab script and collects y_rec
        os.system(r"cd ..\Desktop\\")
        os.system(r'matlab -batch "run generateY.m"')
        os.system('cp y_rec.csv ../mimo2/y_rec.csv')
        with open('y_rec.csv', 'rt') as f:
            reader = csv.reader(f)
            y_rec = list(reader)

        return y_rec

    #read matlab script that generates environment to observe for DYNAMIC DYNAMIC
    def retrieve_y_d_d(self):

        #retrieve data from input file
        with open('../Desktop/input_param.txt', 'r') as file:
            data = file.readlines()
        
        #keep lines that are not affected by target generation
        new_data = data[0:9]
        new_data[8] = new_data[8] + "\n"

        #generate new number of targets and random positions
        previous_generated = []
        self.rand_num = random.randint(0, self.t_max) 
        new_rand_num = self.rand_num 
        for x in range(0, self.rand_num):
            rnum = random.randint(0, self.l_r * self.g_r)
            if (rnum / self.l_r) in previous_generated:
                new_rand_num -= 1
                print("WE ARE HERE")
            else:
                previous_generated.append(rnum / self.l_r)
                new_data.append(str(rnum) + "\n")
        
        #write the new number of targets to the file
        new_data[7] = str(new_rand_num) + "\n"

        #rewrites new file back
        with open('../Desktop/input_param.txt', 'w') as file:
            file.writelines(new_data)

        #runs the matlab script and collects y_rec
        os.system(r"cd ..\Desktop\\")
        os.system(r'matlab -batch "run generateY.m"')
        os.system('cp y_rec.csv ../mimo2/y_rec.csv')
        with open('y_rec.csv', 'rt') as f:
            reader = csv.reader(f)
            y_rec = list(reader)

        return y_rec
    
    #generates next state from current state and action
    def generate_next_state(self, weight_matrix):

        #initial statemetns for calculation
        C_t = np.matrix.transpose(np.matrix(weight_matrix))
        sk = 0
        reward_matrix = np.zeros((self.l_r, self.g_r), dtype=float)

        #runs the matlab script and collects y_rec
        os.system(r"cd ..\Desktop\\")
        os.system(r'matlab -batch "run generateY.m"')
        os.system('cp y_rec.csv ../mimo2/y_rec.csv')
        with open('y_rec.csv', 'rt') as f:
            reader = csv.reader(f)
            y_rec = list(reader)

        yk_main = y_rec

        #choosing which way to retrieve locations based on version
        #yk_main = None
        #if self.version == 1:
        #    print("WE ARE HERE RIGHT")
        #    yk_main = self.retrieve_y_s_s()
        #elif self.version == 2:
        #    yk_main = self.retrieve_y_s_d()
        #else:
        #    yk_main = self.retrieve_y_d_d()

        for t in range(0, len(yk_main)):
            for s in range(0, len(yk_main[0])):
                new_val = yk_main[t][s].replace(" ","")
                new_val = yk_main[t][s].replace("i","j")
                yk_main[t][s] = complex(new_val)
        self.env = yk_main

        #main for loop that iterates through and evaluates yk_main
        for x in range(0, self.l_r):
            a_rt = np.zeros((self.nt, 1), dtype=complex)
            a_rt[0][0] = 1

            #makes a_r and a_t
            for y in range(1, self.nt):
                a_rt[y][0] = cmath.exp(complex(0, math.pi * y * math.sin(self.l_angles[x])))
            
            #add it to main matrix
            self.art.append(a_rt)

            #get h and h_hermitian
            h_x = np.matrix(np.kron(np.matmul(C_t, a_rt), a_rt))
            h_spec = h_x.getH()
            
            #calculate likelihood detection statistics 
            for a in range(0, self.g_r):

                #generate noise
                n = np.identity(self.nt)
                n *= self.sigma ** 2
                noise = []
                for g in range(0, self.nt):
                    for f in range(0, self.nt):
                        noise.append(n[g][f])

                #properly format y
                yk = []
                for m in range(0, self.nt * self.nt):
                    yk.append(yk_main[m][x * self.g_r + a])

                #why transpose yk
                yk = np.matrix.transpose(np.matrix(yk, dtype=complex))

                #calculate glr statistic and compare to threshold
                num = np.matmul(h_spec, yk)
                num2 = (2 * np.linalg.norm(np.array(num)) ** 2) / (self.sigma ** 2 * (np.linalg.norm(np.array(h_x)) ** 2))

                #create matrix that holds information about hypothesis test at each (l, g)
                if float(num2) > float(self.thres):
                    self.hyp_test[x][a] = 1
                    sk += 1
                    reward_matrix[x][a] = num2
                else:
                    self.hyp_test[x][a] = 0

        #takes indices where 1s are and limits them to t_max or less
        indices = []
        for x in range(0, len(self.hyp_test)):
            for y in range(0, len(self.hyp_test[0])):
                if self.hyp_test[x][y] == 1:
                    indices.append(x)
                    break

        print("sk: ")
        print(sk)
        while(len(indices) > self.t_max):
            random.shuffle(indices)
            z = indices.pop()
            for a in range(0, len(self.hyp_test[0])):
                self.hyp_test[z][a] = 0
        if self.t_max == len(indices):
            sk = self.t_max


        os.system(r"cd ..\Desktop\\")
        os.system('cp target_temp.csv ../mimo2/target_temp.csv')
        with open('target_temp.csv', 'rt') as tt:
            reader = csv.reader(tt)
            target_temp = list(reader)

        target_temp = np.ravel(target_temp)
        for ss in range(0, len(target_temp)):
            sample_num = int(target_temp[ss])
            sample_num /= self.l_r
            target_temp[ss] = int(sample_num)
        
        #remove duplicates
        target_temp = list(dict.fromkeys(target_temp))
        
        #target_temp.sort()
        self.tar_loc.write("--------------------------\n")
        self.tar_loc.write("number of targets generated: " + str(len(target_temp)) + "\n")
        self.matlab_gen.write(str(len(target_temp)) + "\n")
        #self.tar_loc.write("TARGETS AT" + "\n")
        #for x in range(0, len(target_temp)):
        #    self.tar_loc.write(str(target_temp[x]) + "\n")
        #get state from sample space of sk
        self.next_st = sk
        self.tar_loc.write("next state: " + str(self.next_st) + "\n")
        self.python_gen.write(str(self.next_st) + "\n")
        #returns number of targets and reward reward_matrix 
        return self.next_st, reward_matrix

    #convert matlab output of C into python matrix
    def retrieve_C(self):

        #accessing matlab output file
        os.system(r"cd ..\Desktop\\")
        os.system(r'matlab -batch "run matlabscript.m"')
        os.system('cp C.csv ../mimo2/C.csv')
        with open('C.csv', 'rt') as f:
            reader = csv.reader(f)
            C = list(reader)

        #turn C into proper complex format
        for t in range(0, len(C)):
            for s in range(0, len(C[0])):
                new_val = C[t][s].replace(" ","")
                new_val = C[t][s].replace("i","j")
                C[t][s] = complex(new_val)
        
        #return complex C
        return C

    #returns next action
    def generate_next_action(self):

        #find highest q_value for that specific state
        high = self.init_q_val
        index = 0
        for x in range(0, len(self.q_table[0])):
            if self.q_table[self.curr_st][x] > high:
                high = self.q_table[self.curr_st][x]
                index = x

        #if there was no highest action, choose a random one
        if high == self.init_q_val:
            index = random.randint(0, self.t_max)
            while(self.q_table[self.curr_st][index] < high):
                index = random.randint(0, self.t_max)
            self.tar_loc.write("no previous q value, choosing random sample: " + str(index) + "\n")
        else:
            self.tar_loc.write("best action found at: " + str(index) + "\n")

        #get angle ranges where there are targets
        ind = []
        curr_input = open("..\Desktop\curr_angle.txt","w")
        for d in range(0, len(self.hyp_test)):
            for e in range(0, len(self.hyp_test[0])):
                if self.hyp_test[d][e] == 1:
                    ind.append(d)
                    break

        #epsilon-greedy algorithm - exploration step else exploitation step
        if (random.random() > self.epsilon):
            index = random.randint(0, self.t_max)
            self.tar_loc.write("taken random action. new index is: " + str(index) + "\n")

        self.tar_loc.write("q table populated at: (" + str(self.curr_st) + ", " + str(self.curr_action_index) + ")\n")

        #randomly select from indexes
        while(len(ind) > index):
            random.shuffle(ind)
            ind.pop()
        for q in range(0, len(ind)):
            num = ind[q]
            curr_input.write(str(self.l_angles[num]) + "\n")

        #close input file and generate action
        curr_input.close()
        C = self.retrieve_C()
        self.ac_opt = C

        #returns next action
        return self.ac_opt, index
    
    #return reward for state and action
    def generate_reward(self, weight_matrix, r_mat):

        C_t = np.matrix.transpose(np.matrix(weight_matrix))
        
        #sums up probability distribution
        main_sum = 0.0
        for a in range(0, self.l_r):
            for b in range(0, self.g_r):

                #check if hypothesis is 1
                if self.hyp_test[a][b] == 1:

                    #make pdf function and test it
                    yk = []
                    for m in range(0, self.nt * self.nt):
                        yk.append(self.env[m][a * self.g_r + b])
                    a_rt = self.art[a]
                    h_x = np.matrix(np.kron(np.matmul(C_t, a_rt), a_rt))
                    h_spec = h_x.getH()
                    yk = np.matrix.transpose(np.matrix(yk, dtype=complex))
                    num = np.matmul(h_spec, yk)
                    num2 = (2 * np.linalg.norm(np.array(num)) ** 2) / (self.sigma ** 2 * (np.linalg.norm(np.array(h_x)) ** 2))
                    
                    first_part = math.exp(-(num2 + r_mat[a][b])/2.0)
                    second_part = iv(0, math.sqrt(num2 * r_mat[a][b]))
                    if np.isinf(second_part):
                        second_part = 0.0

                    pdf_num = 0.5 * first_part * second_part
                    main_sum += pdf_num

        #returns sum of probability distributions
        return main_sum

    #performs q update after every step
    def q_function_update(self, n_a, r_mat):

        #setting new q value
        self.reward = self.generate_reward(n_a, r_mat)
        old_value = self.q_table[self.curr_st][self.curr_action_index]
        future_reward = self.q_table[self.next_st][self.next_action_index]
        new_value = old_value + self.beta * (self.reward + self.gamma * future_reward - old_value)
        self.q_table[self.curr_st][self.curr_action_index] = new_value

        self.tar_loc.write("q table after:\n")
        self.tar_loc.write(str(self.q_table) + "\n")

    #closes output file
    def close_files(self):
        
        #write q table and close files
        self.matlab_gen.close()
        self.python_gen.close()
        self.tar_loc.close()

#main loop
if __name__ == "__main__":
    
    #initial print statements
    print("#------------------------------------------------#")
    print("#: !! STARTING PROGRAM !! :#")
    print("#------------------------------------------------#")

    #get object and initial weight matrix
    obj, C_main = SARSA.read_file()
    current_state = 0

    #print statements for entering cycle
    print("#: ENTERING MAIN LOOP CYCLE")
    print("#------------------------------------------------#")

    #get next state and time steps
    next_state, reward_mat = obj.generate_next_state(C_main)
    print("#: 1")
    print("#-----------#")
    obj.set_next_state(next_state)
    print("#: CALCULATED NEXT STATE")  
    k_time = obj.get_K()

    #main control loop for time steps
    for main in range(2, k_time+2):

        #get next action
        next_action, i = obj.generate_next_action()
        obj.set_next_action(i)
        
        #check if run solved or failed
        matlab_status = open("..\Desktop\matlab_status.txt", 'r')
        stat = str(matlab_status.readline())
        matlab_status.close()

        #control loop 
        if stat == "Failed":
            
            #print statement
            print("FAILED")

            #iterate k
            continue
        
        else:

            #print statement
            print("SOLVED")

            #set next action
            obj.set_next_action(i)
            print("#: CALCULATED NEXT ACTION")

            #perform q update
            obj.q_function_update(next_action, reward_mat)
            print("#: UPDATED Q FUNCTION")
            print("#------------------------------------------------#")

            #last control loop
            if main != (k_time+1):
                #next state becomes current state and next action becomes current action
                current_state = next_state
                obj.set_current_state(current_state)            
                C_main = next_action
                obj.set_current_action(i)
            
                #get next state
                next_state, reward_mat = obj.generate_next_state(C_main)
                obj.set_next_state(next_state)
                print("#: " + str(main))
                print("#-----------#")
                print("#: CALCULATED NEXT STATE")

    #end simulation
    obj.close_files()
    print("#: SIMULATION COMPLETE")