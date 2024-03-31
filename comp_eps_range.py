import numpy as np
import pickle
from cahn_hilliard_willmore import compute_solution
from dolfinx.mesh import CellType, create_unit_square, create_interval
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

    eps_range = {
            "name"  : "2_circ",   
            "d"     : 2,
            #"dt"    : 5e-7,
            "dt"    : 5e-6,
            "theta" : 1, # time stepping family, e.g. theta:1 -> backward Euler,
                # theta" :0.5 -> Crank-Nicholson
            "N"      :  400,
            #"N"      :  2024,
            "rtol_newton": 1e-9,
            "comp_eigenvalue": False,
            "p":np.inf,
            "r_tol_accept":1e-4,
            "alpha_penalty": 100,
            "method":"ciarlet_raviart",
            "t": 0.0,
            "test": False,
            "offscreen":True,
            "plotting_3d":False,
            "delta_diffusive":0.2*1.4**(-6),
            "comp_residum":False,
            "comp_discrete_lap":False

    }
    #eps_range["T"] = eps_range["dt"] * 200
    #eps_range["T"] = eps_range["dt"] * 100
    eps_range["T"] = eps_range["dt"] * 60
    param_dict = eps_range

    load = False
    plot = False
    calculate_l1_int = False

    if param_dict["d"] == 1 :
        msh = create_interval(MPI.COMM_WORLD,param_dict["N"],[0,1])
    elif param_dict["d"] == 2:
        msh = create_unit_square(MPI.COMM_WORLD, param_dict["N"], param_dict["N"], CellType.triangle)
    param_dict["msh"] = msh

    i_list = [0,1,2,3,4,5,6]
    eps_list = [0.2*(1.4)**(-i) for i in i_list]
    

    plt.figure(0)
    for eps in eps_list:
        print("".center(80,"-"))
        print(f"{eps}")
        print("".center(80,"-"))
        param_dict["eps"] = eps
        param_dict["initial_condition"]= lambda x,mp: initial_condition_2circ_2d(x,mp,eps) 
        shift_numbers = [-8*eps**(-i) for i in reversed(range(1,5))]
        shift_numbers += [8*eps**(-i) for i in range(1,5)]
        param_dict["shift_numbers"] = shift_numbers


        if not load:
            compute_solution(**param_dict)

        name = param_dict["name"]
        N = param_dict["N"]
        dt = param_dict["dt"]
        with open(f'graphics/plot_eigenvalue/eigenvalue_{name}_eps_{eps}_N_{N}_dt_{dt}.pkl', 'rb') as f:
            lmb_list = pickle.load(f)["lmb_list"]



        if plot:
            plt.plot(np.linspace(0,param_dict["T"],int(param_dict["T"]/param_dict["dt"])+1),lmb_list, label=f"eps = {eps:1.1e}")
            plt.legend(loc="upper right")
    
    if plot:
        plt.title(f" h = {1/N*np.sqrt(2):1.1e}")
        plt.savefig(f"graphics/plot_eigenvalue/eigenvalue_continous_comparison_{param_dict['name']}_N_{param_dict['N']}_dt_{param_dict['dt']}.png")


    if calculate_l1_int:
        plt.figure(1)
        #calculate L^1 norm of eigenvalues
        l1_int = np.zeros(len(eps_list))
        for i,eps in enumerate(eps_list):
            name = param_dict["name"]
            N = param_dict["N"]
            with open(f'graphics/plot_eigenvalue/eigenvalue_{name}_eps_{eps}_N_{N}_dt_{param_dict["dt"]}.pkl', 'rb') as f:
                lmb_list = np.array(pickle.load(f)["lmb_list"]).real
                print(lmb_list)

            lmb_list_g0 = -lmb_list*np.array(lmb_list<=0,dtype=int)
            tmp = param_dict["dt"]*np.sum( np.min(np.vstack((lmb_list_g0[:-1],lmb_list_g0[1:])),axis=0)
                +1/2*np.abs(np.diff(lmb_list_g0)))
            l1_int[i] = tmp
        plt.loglog(eps_list,np.exp(l1_int),"-x",label="$\exp(\int\lambda^+)$")
        plt.xlabel("$\epsilon$")

        poly_order = (np.polynomial.polynomial.Polynomial.fit(
                            np.log(eps_list),l1_int,1))
        poly_order = poly_order.convert()
        order_int = poly_order.coef
        order = f"{order_int[1]:1.3g}"
        y_order = np.exp(poly_order(np.log(eps_list))) 
        plt.loglog(eps_list,y_order,"-",label="order")
        plt.legend(loc="lower left")

        plt.title(f"order = {order}")
        plt.savefig(f"graphics/plot_eigenvalue/l1_norm_eps_range_{name}.png")
        plt.show()
        
        
