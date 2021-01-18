import os
import numpy as np
import pandas as pd
import time
import csv
import tqdm
import random
from scipy.integrate import odeint, ode,solve_ivp
from joblib import Parallel, delayed
import multiprocessing
import pdb
import matplotlib.pyplot as plt
import argparse
import h5py

parser = argparse.ArgumentParser()


parser.add_argument("--data_dir", type=str, 
    default="data_dump", help="Directory to the data folder!")
parser.add_argument('--scenario', type=str,
    default="baseline", help="f5")
parser.add_argument('--num_cores', type=int,
    default=24)
parser.add_argument('--start_idx', type=int,
    default=0)
parser.add_argument('--end_idx', type=int,
    default=400)
parser.add_argument('--K', type=int,
    default=5000000,help = "Carrying capacity")
parser.add_argument('--A_add', type=int,
    default=10000, help = "Initial value of aphid population")
parser.add_argument('--Ratio', type=int,
    default=2000, help = "A_add/H_add")

args = parser.parse_args()

# get data and shared across threads
h5fn = os.path.join(
    args.data_dir, "{}.h5".format(args.scenario))

with h5py.File(h5fn, "r") as f:
    data_temps = np.asarray(f["data"][args.start_idx:args.end_idx])
    lats = np.asarray(f["lats"][args.start_idx:args.end_idx])
    lons = np.asarray(f["lons"][args.start_idx:args.end_idx])

#processor
num_cores = args.num_cores 
#num_cores =1
# StartdateA = 'StartdateA.csv'
# StartdateH = 'StartdateH.csv'

def draw_multi_lines(x,lat,lon,pltt,foder,xlabel,ylabel):
    # x is a array: (N, m)
    # fn is the file name "xxx_xxx.csv"
    # outfoder
    # pltt: str, "Anp", "Ap","H","AH"
    for i in range(x.shape[0]):
        x_ = x[i]
        plt.plot(x_)
    plt.title(pltt + ': lat = {} lon = {}'.format(lat,lon))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fn = '{}_{}.png'.format(lat,lon)
    out_file = os.path.join(foder, fn)
    if not os.path.isfile(out_file):
        plt.savefig(out_file,bbox_inches='tight')
    plt.close()

def draw_multi_scatters(x,y,lat,lon,foder,xlabel,ylabel):
    # x is a array: (N, m)
    # fn is the file name "xxx_xxx.csv"
    # Di is the date that aphid starts to grow
    # foder
    for i in range(x.shape[0]):
        x_ = x[i]
        y_ = y[i]
        plt.scatter(x_,y_)
    plt.title('AH: lat = {} lon = {}'.format(lat,lon))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fn = '{}_{}.png'.format(lat,lon)
    out_file = os.path.join(foder, fn)
    if not os.path.isfile(out_file):
        plt.savefig(out_file,bbox_inches='tight')
    plt.close()

def writer_csv(rows, filename):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def batch(idx_pt, export_fns):
    tic = time.time()

    temps = data_temps[idx_pt].copy()
    lat = lats[idx_pt].copy()
    lon = lons[idx_pt].copy()
    export_folder, plotAnp_folder,plotAp_folder, plotH_folder, plotAH_folder = export_fns   
    
    temp_fn = "{}_{}.csv".format(lat, lon)
    out_file = os.path.join(export_folder, temp_fn)
    if os.path.exists(out_file):
        print('{} exits!!!'.format(temp_fn))
        return False

    # PARAMETER VALUES FOR APHID
    # Fecundity-Apterous
    m_fap = 1
    Tmin_fap = 8.506
    Tmax_fap = 39.18
    Tref_fap = 12.15
    q1_fap = 1.704
    q2_fap = 1.545

    # Development-Instar
    m_varphi = 1
    Tmin_varphi = 7.11
    Tmax_varphi = 36.09
    Tref_varphi = 31.49
    q1_varphi = 1.267
    q2_varphi = 0.2349

    # Mortality-Apterous
    a0_ap = 0.2546 #-0.1417
    a1_ap = -0.02322 #0.0333
    a2_ap = 0.0006687 #-0.001844
    a3_ap = -0.000001919 #0.00003341


    # Mortality- Instar
    a0_inst = 0.4326
    a1_inst = -0.02443
    a2_inst = -0.0003494
    a3_inst = 0.00002881

    # PARAMETER VALUES FOR LADYBIRD
    # Development
    TminH = 10
    TmaxHf = 32 # fecundity rate
    TmaxH = 35 # developent rate & mortality rate
    TrefH = 22.5
    m_egg = 0.2142 #0.2162
    q1_egg = 1 #1
    q2_egg = 0.1043 #0.1077


    m_inst1 = 0.3778 #0.385
    q1_inst1 =1 #1
    q2_inst1 = 0.1996 #0.224

    m_inst2 = 0.4101 #0.4077
    q1_inst2 = 0.7233 # 0.6861
    q2_inst2 = 0.2 # 0.2

    m_inst3 = 0.2288 #0.2394
    q1_inst3 = 1
    q2_inst3 = 0.09727 #0.2672

    m_inst4 = 0.1229 #0.129
    q1_inst4 = 1.787 #1.088
    q2_inst4 = 0.2

    m_pupa = 0.1648 #0.1631
    q1_pupa = 1.052 #1
    q2_pupa = 0.2 #0.223


    # Mortality
    a0_egg = 0.5983
    a1_egg = -0.06596
    a2_egg = 0.002192
    a3_egg = -0.00001827

    a0_inst1 = 0.7137
    a1_inst1 = -0.06532
    a2_inst1 = 0.001471
    a3_inst1 = 0.000001783

    a0_inst2 = 0.5222
    a1_inst2 = -0.05861
    a2_inst2 = 0.002042
    a3_inst2 = -0.00001966

    a0_inst3 = 0.3779
    a1_inst3 = -0.04105
    a2_inst3 = 0.001332
    a3_inst3 = -0.00001101

    a0_inst4 = 0.1103
    a1_inst4 = -0.009311
    a2_inst4 = 0.0001455
    a3_inst4 = 0.000002274

    a0_pupa = 0.2089
    a1_pupa = -0.02769
    a2_pupa = 0.001182
    a3_pupa = -0.00001519

    a0_adu = 0.1842
    a1_adu = -0.02218
    a2_adu = 0.0008944
    a3_adu = -0.00001074


    # Predation
    a1 = 1.464
    Th1 = 0.01613
    #max = 1/Th  = 61.99 per day per ladybird
    a2 = 1.177
    Th2 = 0.008982
    #max = 1/Th  = 111.33 per day per ladybird
    a3 = 1.437
    Th3 = 0.01155
    #max = 1/Th  = 86.58 per day per ladybird
    a4 = 1.219
    Th4 = 0.003985
    #max = 1/Th  = 250.9
    af = 1.461
    Thf = 0.004453
    #max = 1/Th  = 224.56
    am = 1.461
    Thm = 0.004453

    # Other parameters
    K = args.K #carrying cacpacity of ladybird
    #coefficients for eqn 13
    m_beta = 0.9984
    q1_beta = 1.897
    q2_beta = 1.8
    Tref_beta = 22.46

    Qp = 100 # transformation rate of ladybird
    v_max = 0.3 # maximal mortality rate for aphid and ladybird
    theta = 0.5 # ratio of female ladybird to male ladybird

    A_add = args.A_add
    Ratio = args.Ratio
    
    def interpolated_temp(temp,t):
        #integrate temperature at each dt
        # Tlen: length of temperature data
        # temp: temperature data
        temp_len = len(temp)
        t_tmp = int(np.floor(t))
        if t_tmp <= (temp_len-1):
            Temp_t = temp[t_tmp-1] + (temp[t_tmp] - temp[t_tmp-1])*(t - t_tmp)
        else:
            Temp_t = temp[temp_len-1]
        return Temp_t
        # #Stella's temperature
        # t_tmp = int(np.floor(t))
        # return temp[t_tmp]

    def indicator(t,Tmin,Tmax):
        #Tmin = Tmin_fap or TminH
        #Tmax = TmaxH or TmaxH
        i = int(np.floor(t))
        if (t >= 5) & (np.all(temp[i-5:i+1] >= Tmin)) & (np.all(temp[i-5:i+1] <= Tmax)):
                return True
        else: return False

    def thorneley_france(m,Tmin,Tmax,Tref,q1,q2,T):
        # Generic temperature-dependent function
        if (T>=Tmin) & (T<=Tmax):
            return m*(((T-Tmin)**q1)*((Tmax-T)**q2))/(((Tref-Tmin)**q1)*((Tmax-Tref)**q2))
        else: return 0

    def polynomial(a0,a1,a2,a3,Tmin,Tmax,T):
        if (T>=Tmin) & (T<=Tmax):
            return np.minimum(a0+a1*T+a2*T**2+a3*T**3,v_max)
        else: return v_max

    def fdpr(a,Th,Aden_t):
        #food_dependent_predation_rate
        return a*Aden_t/(1+a*Th*Aden_t)

    def carring_capacity(Aden, K):
        #carring capacity
        #k_effect_t = 1-Aden_t/K
        if Aden <= K: 
            return 1-Aden/K
        else: return 0


    def fecH(Aden,Temp_t):
        # Fecundity rate of female ladybirds
        if Temp_t <= TmaxHf:
            tdpr_t = thorneley_france(m_beta,TminH,TmaxH,Tref_beta,q1_beta,q2_beta,Temp_t)
            return tdpr_t*fdpr(af, Thf, Aden)/Qp
        else:
            return 0

    #Model equations:
    def Solve_euler_model(var0,t_start,t_end,dt,predation,num_change_Aap = 0,num_change_Hegg = 0):
        ts = np.arange(t_start,t_end,dt)
        n_t=len(ts)
        Aap = np.zeros([n_t]);A1 = np.zeros([n_t]); A2 = np.zeros([n_t]); A3 = np.zeros([n_t]); A4 = np.zeros([n_t])
        Hegg = np.zeros([n_t]); H1 = np.zeros([n_t]); H2 = np.zeros([n_t]); H3 = np.zeros([n_t]); H4 = np.zeros([n_t]); Hpupa = np.zeros([n_t]); Hf = np.zeros([n_t]); Hm = np.zeros([n_t])
        Aden = np.zeros([n_t]); Hden = np.zeros([n_t]); Predated_prey = np.zeros([n_t]); Adeath = np.zeros([n_t])
        Aap[0],A1[0],A2[0],A3[0],A4[0],Hegg[0],H1[0],H2[0],H3[0],H4[0],Hpupa[0],Hf[0],Hm[0] = var0
        Aden[0] = Aap[0] + A1[0] + A2[0] + A3[0] + A4[0]
        Hden[0] = Hegg[0] + H1[0] + H2[0] + H3[0] + H4[0] + Hpupa[0] + Hf[0] + Hm[0]

        for i in range(1, n_t):
            t = ts[i-1] #previous time step
            t_cur = ts[i] #current time step
            Aap_t = Aap[i-1]; A1_t = A1[i-1]; A2_t = A2[i-1]; A3_t = A3[i-1]; A4_t = A4[i-1]; Aden_t = Aden[i-1]
            Hegg_t = Hegg[i-1]; H1_t = H1[i-1]; H2_t = H2[i-1]; H3_t = H3[i-1]; H4_t = H4[i-1]; Hpupa_t = Hpupa[i-1]; Hf_t = Hf[i-1]; Hm_t = Hm[i-1]; Hden_t = Hden[i-1]
            #integrated temperature
            
            Temp_t = interpolated_temp(temp,t)
            ## Temperature-dependent parameters of aphid
            #fecudity rate
            f_ap_t = thorneley_france(m_fap, Tmin_fap, Tmax_fap, Tref_fap, q1_fap, q2_fap, Temp_t)
            # development
            varphi_t = thorneley_france(m_varphi, Tmin_varphi, Tmax_varphi, Tref_varphi, q1_varphi, q2_varphi, Temp_t)
            # mortality
            mu_inst_t = polynomial(a0_inst,a1_inst,a2_inst,a3_inst,Tmin_varphi,Tmax_varphi,Temp_t)
            mu_ap_t = polynomial(a0_ap,a1_ap,a2_ap,a3_ap,Tmin_fap,Tmax_fap,Temp_t)
            #carring capacity
            #k_effect_t = 1-Aden_t/K
            k_effect_t = carring_capacity(Aden_t,K)

            #Predartion rate as a function of temperature at maximal aphid density
            #temperature dependent predation rate
            tdpr_t = thorneley_france(m_beta,TminH,TmaxH,Tref_beta,q1_beta,q2_beta,Temp_t)
            # Fecundity rate of female ladybirds
            f_H_t = fecH(Aden_t,Temp_t) 
            #Stage-specific Development rates
            #Temperature-dependent development rates for egg and pupa
            delta_egg_t = thorneley_france(m_egg, TminH, TmaxH, TrefH, q1_egg, q2_egg, Temp_t)
            delta_pupa_t = thorneley_france(m_pupa, TminH, TmaxH, TrefH, q1_pupa, q2_pupa, Temp_t)
            #Temperature-dependent developments rares at prey saturation
            delta_inst1_prey_saturation_t = thorneley_france(m_inst1, TminH, TmaxH, TrefH, q1_inst1, q2_inst1, Temp_t)
            delta_inst2_prey_saturation_t = thorneley_france(m_inst2, TminH, TmaxH, TrefH, q1_inst2, q2_inst2, Temp_t)
            delta_inst3_prey_saturation_t = thorneley_france(m_inst3, TminH, TmaxH, TrefH, q1_inst3, q2_inst3, Temp_t)
            delta_inst4_prey_saturation_t = thorneley_france(m_inst4, TminH, TmaxH, TrefH, q1_inst4, q2_inst4, Temp_t)
            # Mortality rate of various stages
            gamma_egg_t = polynomial(a0_egg,a1_egg,a2_egg,a3_egg,TminH,TmaxH,Temp_t)
            gamma_inst1_t = polynomial(a0_inst1,a1_inst1,a2_inst1,a3_inst1,TminH,TmaxH,Temp_t)
            gamma_inst2_t = polynomial(a0_inst2,a1_inst2,a2_inst2,a3_inst2,TminH,TmaxH,Temp_t)
            gamma_inst3_t = polynomial(a0_inst3,a1_inst3,a2_inst3,a3_inst3,TminH,TmaxH,Temp_t)
            gamma_inst4_t = polynomial(a0_inst4,a1_inst4,a2_inst4,a3_inst4,TminH,TmaxH,Temp_t)
            gamma_pupa_t = polynomial(a0_pupa,a1_pupa,a2_pupa,a3_pupa,TminH,TmaxH,Temp_t) 
            gamma_f_t = polynomial(a0_adu,a1_adu,a2_adu,a3_adu,TminH,TmaxH,Temp_t)
            gamma_m_t = gamma_f_t
            # common parameters for dA_dt
            common_pA_t = tdpr_t*(H1_t*a1/(1+a1*Th1*Aden_t) + H2_t*a2/(1+a2*Th2*Aden_t) + H3_t*a3/(1+a3*Th3*Aden_t) + H4_t*a4/(1+a4*Th4*Aden_t) + Hf_t*af/(1+af*Thf*Aden_t) + Hm_t*am/(1+am*Thm*Aden_t))
            dA1_dt = f_ap_t*k_effect_t*Aap_t - mu_inst_t*A1_t - varphi_t*A1_t - A1_t*common_pA_t
            dA2_dt = varphi_t*A1_t - mu_inst_t*A2_t - varphi_t*A2_t - A2_t*common_pA_t
            dA3_dt = varphi_t*A2_t - mu_inst_t*A3_t - varphi_t*A3_t - A3_t*common_pA_t
            dA4_dt = varphi_t*A3_t - mu_inst_t*A4_t - varphi_t*A4_t - A4_t*common_pA_t
            dAap_dt = varphi_t*A4_t - mu_ap_t*Aap_t - Aap_t*common_pA_t

            #parameters for dH_dt
            delta_H1_t = delta_inst1_prey_saturation_t*tdpr_t*fdpr(a1,Th1,Aden_t)*Th1
            delta_H2_t = delta_inst2_prey_saturation_t*tdpr_t*fdpr(a2,Th2,Aden_t)*Th2
            delta_H3_t = delta_inst3_prey_saturation_t*tdpr_t*fdpr(a3,Th3,Aden_t)*Th3
            delta_H4_t = delta_inst4_prey_saturation_t*tdpr_t*fdpr(a4,Th4,Aden_t)*Th4

            dHegg_dt = f_H_t*Hf_t - (gamma_egg_t + delta_egg_t)*Hegg_t
            dH1_dt = delta_egg_t*Hegg_t - delta_H1_t*H1_t - gamma_inst1_t*H1_t
            dH2_dt = delta_H1_t*H1_t - delta_H2_t*H2_t - gamma_inst2_t*H2_t
            dH3_dt = delta_H2_t*H2_t - delta_H3_t*H3_t - gamma_inst3_t*H3_t
            dH4_dt = delta_H3_t*H3_t - delta_H4_t*H4_t - gamma_inst4_t*H4_t
            dHpupa_dt = delta_H4_t*H4_t - delta_pupa_t*Hpupa_t - gamma_pupa_t*Hpupa_t
            dHf_dt = theta*delta_pupa_t*Hpupa_t - gamma_f_t*Hf_t
            dHm_dt = (1-theta)*delta_pupa_t*Hpupa_t - gamma_m_t*Hm_t

            dAH_dt = [dAap_dt,dA1_dt,dA1_dt,dA2_dt,dA3_dt,dA4_dt,dHegg_dt,dH1_dt,dH2_dt,dH3_dt,dH4_dt,dHpupa_dt,dHf_dt,dHm_dt]
            Predated_prey[i] = common_pA_t*(Aap_t + A1_t + A2_t + A3_t + A4_t)
            Adeath[i] = mu_ap_t*Aap_t + mu_inst_t*A1_t + mu_inst_t*A2_t + mu_inst_t*A3_t + mu_inst_t*A4_t

            Aap[i] = dt*dAap_dt + Aap_t
            A1[i] =dt*dA1_dt + A1_t
            A2[i] = dt*dA2_dt + A2_t
            A3[i] = dt*dA3_dt + A3_t
            A4[i] = dt*dA4_dt + A4_t
            Aden[i] = Aap[i] + A1[i] + A2[i] + A3[i] + A4[i]
            
            Hegg[i] = dt*dHegg_dt + Hegg_t
            H1[i] = dt*dH1_dt + H1_t
            H2[i] = dt*dH2_dt + H2_t
            H3[i] = dt*dH3_dt + H3_t
            H4[i] = dt*dH4_dt + H4_t
            Hpupa[i] = dt*dHpupa_dt + Hpupa_t
            Hf[i] = dt*dHf_dt + Hf_t
            Hm[i] = dt*dHm_dt + Hm_t
            Hden[i] = Hegg[i] +  H1[i] + H2[i] + H3[i] + H4[i] +  Hpupa[i] + Hf[i] + Hm[i]

            if Aap[i] < 0: Aap[i] = 0
            if A1[i] < 0: A1[i] = 0
            if A2[i] < 0: A2[i] = 0
            if A3[i] < 0: A3[i] = 0
            if A4[i] < 0: A4[i] = 0
            if Aden[i] <1: Aden[i] = 0;Aap[i] = 0; A1[i] = 0; A2[i] = 0; A3[i] = 0; A4[i] = 0

            if Hegg[i] < 0: Hegg[i] = 0
            if H1[i] < 0: H1[i] = 0
            if H2[i] < 0: H2[i] = 0
            if H3[i] < 0: H3[i] = 0
            if H4[i] < 0: H4[i] = 0
            if Hpupa[i] < 0: Hpupa[i] = 0
            if Hf[i] < 0: Hf[i] = 0
            if Hm[i] < 0: Hm[i] = 0
            if Hden[i] < 1: Hden[i] = 0;Hegg[i] = 0; H1[i] = 0; H2[i] = 0; H3[i] = 0; H4[i] = 0; Hpupa[i] = 0; Hf[i] = 0; Hm[i] = 0

            # Add aphid and ladybird into the population
            # if t_cur >=5:
            #     import pdb;pdb.set_trace()
            if indicator(t_cur, Tmin_fap, Tmax_fap) and (num_change_Aap==0):
                Aap[i] = A_add/5; A1[i] = Aap[i]; A2[i] = Aap[i]; A3[i]=Aap[i]; A4[i] = Aap[i]
                Aden[i] = A_add 
                num_change_Aap += 1
                # with open(StartdateA, 'a') as f:
                #     f.write('{},{},{}'.format(temp_fn, year, t_cur) +"\n")
            if indicator(t_cur, TminH, TmaxH) and (num_change_Hegg==0) and (predation == True):
                Hegg[i] = (A_add/Ratio)/8
                H1[i] = Hegg[i]; H2[i] = Hegg[i];H3[i] = Hegg[i];H4[i] = Hegg[i];
                Hpupa[i] = Hegg[i];Hf[i] = Hegg[i];Hm[i] = Hegg[i]
                Hden[i] = A_add/Ratio
                num_change_Hegg += 1
                # with open(StartdateH, 'a') as f:
                #     f.write('{},{},{}'.format(temp_fn, year, t_cur) +"\n")
            Pre_AH = [Aap_t,A1_t,A2_t,A3_t,A4_t,Hegg_t,H1_t,H2_t,H3_t,H4_t,Hpupa_t,Hf_t,Hm_t]
            Cur_AH = [Aap[i],A1[i],A2[i],A3[i],A4[i],Hegg[i],H1[i],H2[i],H3[i],H4[i],Hpupa[i],Hf[i],Hm[i]]
            outputs = [Aden, Hden, Predated_prey, Adeath, Aap, A1, A2, A3, A4, Hegg, H1, H2, H3, H4, Hpupa, Hf, Hm]
            if Aden[i] == 0 and Hden[i] == 0 and num_change_Aap > 0 and num_change_Hegg >0:
                break
        return outputs

    # temp_path = os.path.join("{}".format(scenario),temp_fn)
    # temps = pd.read_csv(temp_path).to_numpy().reshape(30, 365)
    years = np.arange(0,30)
    Adens_np = [];Adens_p = [];Hdens_p = [];Predated_preys = [];Adeaths = []
    Adens_np_day = [];Adens_p_day =[];Hdens_p_day = [];Predated_preys_day = []; Adeaths_day = []
    for year in years:
        temp = temps[year]
        var0 = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        #model outputs (with predation and no predation)
        outputs_p = Solve_euler_model(var0,t_start = 0, t_end = 365,dt=0.01,predation = True)
        outputs_np = Solve_euler_model(var0,t_start = 0, t_end = 365,dt=0.01,predation = False)
        Aden_p = outputs_p[0]; Hden_p = outputs_p[1]; Predated_prey = outputs_p[2]; Adeath = outputs_p[3]
        Aden_np = outputs_np[0]; Hden_np = outputs_np[1]
        #daily outputs
        Aden_np_day = Aden_np[::100]
        Aden_p_day = Aden_p[::100]
        Hden_p_day = Hden_p[::100]
        Predated_prey_day = Predated_prey[::100]
        Adeath_day = Adeath[::100]
        # Adens_np += [Aden_np]
        # Adens_p += [Aden_p]
        # Hdens_p += [Hden_p]
        # Predated_preys += [Predated_prey]
        # Adeaths += [Adeath]
        Adens_np_day += [Aden_np_day]
        Adens_p_day += [Aden_p_day]
        Hdens_p_day += [Hden_p_day]
        Predated_preys_day += [Predated_prey_day]
        Adeaths_day += [Adeath_day]
    Adens_np_day = np.array(Adens_np_day)
    Adens_p_day = np.array(Adens_p_day)
    Hdens_p_day = np.array(Hdens_p_day)
    Predated_preys_day = np.array(Predated_preys_day)
    Adeaths_day = np.array(Adeaths_day)
    AH = np.concatenate((Adens_np_day,Adens_p_day,Hdens_p_day,Predated_preys_day,Adeaths_day))
    AH = pd.DataFrame(np.transpose(AH))
    if not os.path.isfile(out_file):
        AH.to_csv(out_file,header=None)
    draw_multi_lines(Adens_np_day, lat,lon,'Anp',plotAnp_folder,xlabel = 'Day', ylabel = 'Aphid Population Abundance')
    draw_multi_lines(Adens_p_day,lat,lon,'Ap',plotAp_folder,xlabel = 'Day', ylabel = 'Aphid Population Abundance')
    draw_multi_lines(Hdens_p_day,lat,lon,'H',plotH_folder,xlabel = 'Day', ylabel = 'Ladybird Population Abundance')        
    draw_multi_scatters(Adens_p_day,Hdens_p_day,lat,lon,plotAH_folder, xlabel = 'Aphid Population Abundance', ylabel = 'Ladybird Population Abundance')

    toc = time.time()
    print(temp_fn + " " + "Elapsed time: {}".format(toc - tic))
    return True
    
if __name__ == '__main__':
    fold_dir = "exports"
    export_fns = []
    for folder_name in ["data", "plotAnp", "plotAp", "plotH", "plotAH"]:
        folder = os.path.join(
            fold_dir, "{}_{}".format(args.scenario, folder_name))
        assert os.path.exists(folder), "{} doesn't exist!".format(folder)
        export_fns += [folder] 
    num_pts = len(data_temps) # 
    processed_list = Parallel(n_jobs=num_cores)(delayed(batch)(i, export_fns) for i in range(num_pts))
