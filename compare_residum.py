import numpy as np
import pickle
from numpy.linalg import LinAlgError
from cahn_hilliard_willmore import compute_solution
from dolfinx.mesh import CellType, create_unit_square, create_interval,Mesh
from mpi4py import MPI
import matplotlib.pyplot as plt

if __name__ == "__main__":

    def initial_condition_2circ_2d(x:np.ndarray,mp:str,eps:float):
        dist_to_center1 = (1/8-((x[0]-1/2)**2 + (x[1]-1/4)**2)**(1/2))
        dist_to_center2 = (1/8-((x[0]-1/2)**2 + (x[1]-3/4)**2)**(1/2))
        return  (dist_to_center1>=dist_to_center2)\
                *np.tanh(eps**(-1)*(np.sqrt(2))**(-1)* (1/8-((x[0]-1/2)**2 + (x[1]-1/4)**2)**(1/2)))+\
                (dist_to_center1<dist_to_center2)*\
                np.tanh(eps**(-1)*(np.sqrt(2))**(-1)* (1/8-((x[0]-1/2)**2 + (x[1]-3/4)**2)**(1/2)))

    param_dict = {
            "name"  : "2_circ",   
            "d"     : 2,
            #"dt"    : 5e-7,
            "theta" : 1, # time stepping family, e.g. theta:1 -> backward Euler,
                # theta" :0.5 -> Crank-Nicholson
            "rtol_newton": 1e-9,
            "comp_eigenvalue": False,
            "p":np.inf,
            "r_tol_accept":1e-4,
            "alpha_penalty": 100,
            "method":"ciarlet_raviart",
            "t": 0.0,
            "test": False,
            "offscreen":True,
            "plotting_3d":True,
            "delta_diffusive":0.2*1.4**(-6),
            "comp_residum":True,
            "reload_time":48*3.125e-6
    }
    param_dict["reload_time_res"] = param_dict["reload_time"]
    #param_dict["reload_time_res"] = -1 


    load = True
    plot = True
    calculate_l2_int = True

    msh_larger_factor = 2

    i = 3
    eps = 0.2*(1.4)**(-i) 
    param_dict["eps"] = eps


    j_list = [0,1,2,3,4,5]#,5]#,6]
    h_list = [1/40*np.sqrt(2)*((1/2)**(2/3))**(4)  for j in j_list]
    N_list = [ int(1/(h/np.sqrt(2))) for h in h_list]
    dt_list = [1e-4*2**(-j) for j in j_list]


    # fixed h
    #h_list = [1/40*np.sqrt(2)*((1/2)**(2/3))**(j)  for j in j_list]
    # fixed dt
    #dt_list = [1e-4*2**(-3) for j in j_list]

    T = 3e-4
    param_dict["T"] = T


    if calculate_l2_int:

        res_dict_list = {0:[],1:[],2:[],3:[]}
    
    for i,j in enumerate(j_list):
        N = N_list[i]
        dt = dt_list[i]
        param_dict["N"] = N
        param_dict["dt"] = dt

        if param_dict["d"] == 1 :
            msh = create_interval(MPI.COMM_WORLD,param_dict["N"],[0,1])
            msh_larger = create_interval(MPI.COMM_WORLD,int(msh_larger_factor*param_dict["N"]),[0,1])
        elif param_dict["d"] == 2:
            msh = create_unit_square(MPI.COMM_WORLD, param_dict["N"], param_dict["N"], CellType.triangle)
            msh_larger = create_unit_square(MPI.COMM_WORLD, int(msh_larger_factor*param_dict["N"]),
                                       int(msh_larger_factor* param_dict["N"]), CellType.triangle)
        param_dict["msh"] = msh
        param_dict["msh_larger"] = msh_larger

        param_dict["initial_condition"]= lambda x,mp: initial_condition_2circ_2d(x,mp,eps) 

        if not load:
            compute_solution(**param_dict)


        if plot:
            with open(f'graphics/plot_residuals/reisudal_eps_{eps}_N_{N}_dt_{dt}.pkl', 'rb') as f:
                    res_dict = pickle.load(f)

                    eta_space_list = res_dict["eta_space_list"]
                    eta_time_list = res_dict["eta_time_list"]
                    nu_space_list = res_dict["nu_space_list"]
                    nu_time_list = res_dict["nu_time_list"]

            n_steps = int(np.round(param_dict["T"]/param_dict["dt"]))
            time = np.linspace(0,param_dict["T"],n_steps)
            #plt.figure(0)
            #plt.plot(time[1:],eta_space_list[1:n_steps],"-x",label=f"$\\tau=${dt:1.1e} h={h_list[i]:1.1e}")

            #plt.figure(1)
            #plt.plot(time[1:],eta_time_list[1:n_steps],"-x",label=f"$\\tau=${dt:1.1e} h={h_list[i]:1.1e}")

            #plt.figure(2)
            #plt.plot(time[1:],nu_space_list[1:n_steps],"-x",label=f"$\\tau=${dt:1.1e}  h={h_list[i]:1.1e}")
            
            #plt.figure(3)
            #plt.plot(time[1:],nu_time_list[1:n_steps],"-x",label=f"$\\tau=${dt:1.1e} h={h_list[i]:1.1e}")

            plt.figure(0)
            plt.plot(time,eta_space_list[:n_steps],"-x",label=f"$\\tau=${dt:1.1e} h={h_list[i]:1.1e}")
            plt.yscale("log")

            plt.figure(1)
            plt.plot(time,eta_time_list[:n_steps],"-x",label=f"$\\tau=${dt:1.1e} h={h_list[i]:1.1e}")
            plt.yscale("log")

            plt.figure(2)
            plt.plot(time,nu_space_list[:n_steps],"-x",label=f"$\\tau=${dt:1.1e}  h={h_list[i]:1.1e}")
            plt.yscale("log")
            
            plt.figure(3)
            plt.plot(time,nu_time_list[:n_steps],"-x",label=f"$\\tau=${dt:1.1e} h={h_list[i]:1.1e}")
            plt.yscale("log")

        if calculate_l2_int:
            res_list = [eta_space_list,eta_time_list,nu_space_list,nu_time_list[1:]]
            #res_list = [eta_space_list,eta_time_list,nu_space_list,nu_time_list[1:]]
            for k,res in enumerate(res_list):
                res = res**2
                l1_norm = dt*np.sum( np.min(np.vstack((res[:-1],res[1:])),axis=0)
                    +1/2*np.sum(np.abs(np.diff(res))))
                res_dict_list[k].append(l1_norm**(1/2))


    if calculate_l2_int:


        ############### space
        eta_space_list_l1 = np.array(res_dict_list[0])
        nu_space_list_l1 = np.array(res_dict_list[2])
        plt.figure(4)
        plt.loglog(dt_list,eta_space_list_l1,"-x",label="$\eta_{space}$")
        plt.loglog(dt_list,nu_space_list_l1,"-x",label="$\\nu_{space}$")
        plt.xlabel("$\tau$")

        try:
            poly_order_eta = (np.polynomial.polynomial.Polynomial.fit(
                                np.log(np.array(dt_list)),np.log(eta_space_list_l1),1))
            poly_order_eta = poly_order_eta.convert()
            order_int_eta = poly_order_eta.coef
            order_eta = f"{order_int_eta[1]:1.3g}"
            y_order_eta = np.exp(poly_order_eta(np.log(dt_list))) 
            plt.loglog(dt_list,y_order_eta,"-",label=f"order $\eta$ ={order_eta}")
        except :
            y_order_eta = "ud"



        try:
            poly_order_nu = (np.polynomial.polynomial.Polynomial.fit(
                                np.log(np.array(dt_list)),np.log(nu_space_list_l1),1))
            poly_order_nu = poly_order_nu.convert()
            order_int_nu = poly_order_nu.coef
            order_nu = f"{order_int_nu[1]:1.3g}"
            y_order_nu = np.exp(poly_order_nu(np.log(dt_list))) 
            plt.loglog(dt_list,y_order_nu,"-",label=f"order $\\nu$ ={order_nu}")
        except :
            y_order_nu = "ud"
        plt.legend(loc="lower right")
        plt.savefig(f"graphics/plot_residuals/space_l1_eps_{eps}.png")

        eta_time_list_l1 = res_dict_list[1]
        nu_time_list_l1 = res_dict_list[3]

        ############### time
        plt.figure(5)
        plt.loglog(dt_list,eta_time_list_l1,"-x",label="$\eta_{time}}$")
        plt.loglog(dt_list,nu_time_list_l1,"-x",label="$\\nu_{time}$")
        plt.xlabel("$\\tau$")

        try:
            poly_order_eta = (np.polynomial.polynomial.Polynomial.fit(
                                np.log(np.array(dt_list[1:])),np.log(eta_time_list_l1[1:]),1))
            poly_order_eta = poly_order_eta.convert()
            order_int_eta = poly_order_eta.coef
            order_eta = f"{order_int_eta[1]:1.3g}"
            y_order_eta = np.exp(poly_order_eta(np.log(dt_list))) 
            plt.loglog(dt_list,y_order_eta,"-",label=f"order $\eta$ ={order_eta}")
        except :
            y_order_eta = "ud"


        try:
            poly_order_nu = (np.polynomial.polynomial.Polynomial.fit(
                                np.log(np.array(dt_list[1:])),np.log(nu_time_list_l1[1:]),1))
            poly_order_nu = poly_order_nu.convert()
            order_int_nu = poly_order_nu.coef
            order_nu = f"{order_int_nu[1]:1.3g}"
            y_order_nu = np.exp(poly_order_nu(np.log(dt_list))) 
            plt.loglog(dt_list,y_order_nu,"-",label=f"order $\\nu$ ={order_nu}")
        except :
            y_order_nu = "ud"
        plt.legend(loc="lower right")
        plt.savefig(f"graphics/plot_residuals/time_l1_eps_{eps}.png")


        


    if plot:
        plt.figure(0)
        plt.title("$\eta_{space}$"+ f" $\epsilon = {eps:1.1e}$")
        plt.legend(loc="upper right")
        plt.xlabel("t")
        plt.savefig(f"graphics/plot_residuals/eta_space_eps_{eps}.png")

        plt.figure(1)
        plt.title("$\eta_{time}$" +f" $\epsilon = {eps:1.1e}$")
        plt.legend(loc="upper right")
        plt.xlabel("t")
        plt.savefig(f"graphics/plot_residuals/eta_time_eps_{eps}.png")

        plt.figure(2)
        plt.title("$\\nu_{space}$"+f" $\epsilon = {eps:1.1e}$")
        plt.legend(loc="upper right")
        plt.xlabel("t")
        plt.savefig(f"graphics/plot_residuals/nu_space_eps_{eps}.png")

        plt.figure(3)
        plt.title("$\\nu_{time}$"+f" $\epsilon = {eps:1.1e}$")
        plt.legend(loc="upper right")
        plt.xlabel("t")
        plt.savefig(f"graphics/plot_residuals/nu_time_eps_{eps}.png")
        plt.show()