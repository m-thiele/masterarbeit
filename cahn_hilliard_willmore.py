
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ufl

from dolfinx import plot
from dolfinx.fem import Function, FunctionSpace,form,assemble_scalar
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_unit_square, create_interval,Mesh
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx,ds, grad, inner, FacetNormal, dot, div
from matplotlib.colors import ListedColormap



from eigenvalue import compute_eigenvalue
from residum import compute_residum,comp_discrete_laplace

from mpi4py import MPI
from petsc4py import PETSc


try:
    import pyvista as pv
    have_pyvista = True
except ModuleNotFoundError:

    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False




def comp_norm(c:Function,msh:Mesh,op:MPI.SUM)->float:
    """
    computes the L^2 norm of c on a given mesh msh

    Args:
        c (Function): function 
        msh (Mesh): mesh 
        op (MPI.SUM): operator on mesh 

    Returns:
        float:error 
    """

    E = form((c)**2 * dx)
    error = msh.comm.allreduce(assemble_scalar(E), op=op)
    return error**(1/2)

def plot_pv(u:Function, dofs0:np.ndarray, grid:pv.UnstructuredGrid):
    """
    plots a function u on grid
    Args:
        u (Function): function to plot 
        dofs0 (np.ndarray): if mixed element, which part to plot 
        grid (pv.UnstructuredGrid): grid on which to plot 
    """

    grid.point_data["c"] = u.x.array[dofs0].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "c.png"
    warped = grid.warp_by_scalar()
    warped.plot()
    pv.plot(grid, show_edges=True, screenshot=screenshot)

def plot_eigenvalue(lmb_list:list,dt:float,save_name:str,offscreen:bool):
    """
    plots the eigenvalue

    Args:
        lmb_list (list):list of eigenvalues 
        dt (float):float 
        save_name (str):name to save string if offscreen 
        offscreen (bool): whether to save plot 
    """

    plt.plot(np.arange(len(lmb_list))*dt,np.array(lmb_list))
    plt.title("Eigenvalue")
    plt.xlabel("t")
    if offscreen:
        plt.savefig(f"graphics/{save_name}.png")
        plt.close()
    else:
        plt.show()

def colormap(grid:pv.UnstructuredGrid, delta:float=0.1,key:str="c")->ListedColormap:
    """
    creates a colormap with fixed band around 0
    Args:
        grid (pv.UnstructuredGrid): grid with point data   
        delta (float, optional): width of 0 level set. Defaults to 0.1.

    Returns:
        ListedColormap: created colormap 
    """

    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    grey = np.array([189 / 256, 189 / 256, 189 / 256, 1.0])
    
    delta = delta 
    #mapping = np.linspace(grid.point_data[key].min(),grid.point_data[key].max(),256)
    mapping = np.linspace(-1,1,256)
    newcolors = np.array(cmaplist)
    index0 = np.logical_and(mapping >= -delta,mapping<=delta)
    if newcolors[index0,:].shape[0] > 0:
        newcolors [index0,:] = np.vstack([grey for i in range(newcolors[index0,:].shape[0])])
    cmap = ListedColormap(newcolors)

    return cmap

def create_background_plotter3D(V0:FunctionSpace,
                                save_name:str,offscreen:bool=False,plot_3d:bool=False):
    """
    creates backgroundplotter

    Args:
        V0 (FunctionSpace): on which we plot 
        save_name (str): name to save figure 
        offscreen (bool, optional): if  saving. Defaults to False.

    Returns:
        plotter: backgroundplotter
        grid: grid
    """


    topology, cell_types, x = plot.create_vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.open_movie(f"graphics/{save_name}.mp4",framerate=3, quality=10)

    if not plot_3d:
        plotter.camera.position = (0.5,0.5,3)
        plotter.camera.focal_point = (0.5, 0.49, -1)
        plotter.camera_set = True

    return plotter,grid


def update_background_plotter3D(t:float,u:Function,dofs0:np.ndarray,
    plotter:pv.Plotter,grid:pv.UnstructuredGrid,
    title:str="concentration",plotting_3d:bool=True,
    delta_diffusive:float=0.05):
    """
    updates background plotter

    Args:
        t (float): _description_
        u (Function): _description_
        dofs0 (np.ndarray): _description_
        plotter (pv.Plotter): _description_
        grid (pv.UnstructuredGrid): _description_
        title (str, optional): _description_. Defaults to "concentration".
        plotting_3d (bool, optional): _description_. Defaults to True.
    """


    plotter.clear()
    grid.point_data[" "] = u.x.array[dofs0].real

    #cmap,scalars = colormap(grid)
    cmap = colormap(grid,delta=delta_diffusive)

    if plotting_3d: 
        grid_warped = grid.warp_by_scalar()
    else:
        grid_warped = grid
    plotter.add_mesh(
        grid_warped,
        lighting=False,
        show_edges=False,
        cmap = cmap,
        clim = [-1,1],
        show_scalar_bar=False
        #scalars = scalars
        #scalars = "values" 
        )
    #plotter.add_title(title)
    #plotter.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
    plotter.write_frame()



def get_test_values(eps:float, x:ufl.SpatialCoordinate,
                     initial_condition:callable):
    """

    returns mu, \laplace \mu and d2/du2 W(u), so that by the 
    initial condition is the solution
    Args:
        eps (float): epsilon surface parameter 
        x (ufl.SpatialCoordinate): spatial coordinate of mesh 
        initial_condition (callable): initial condition 

    Returns:
        _type_: _description_
    """

    
    u_sol_automatic = initial_condition(x,"ufl")
    #u_sol_automatic_p3 =  (u_sol_automatic)**3
    #u_sol_automatic_p2 = (u_sol_automatic)**2
    #u_sol_x = u_sol_automatic
    u_sol_xx = div(grad(u_sol_automatic))
    mu_test = (-1)*eps*u_sol_xx +eps**(-1)*((u_sol_automatic)**3-u_sol_automatic) 
    #mu_test_x = mu_test.dx(0)
    mu_test_xx = div(grad(mu_test))
    d2W_test = (3*(u_sol_automatic)**2-1)

    mu_test_x = grad(mu_test)
    #mu_test = (-1)*eps*ufl.pi*ufl.sin(ufl.pi*x[0]) \
    #        -eps**(-1)/(ufl.pi**3)*(ufl.sin(ufl.pi*x[0]))**3\
    #        +eps**(-1)/ufl.pi*(ufl.sin(ufl.pi*x[0]))

    #    

    #mu_test_xx = (ufl.pi*eps)**(-1) * ufl.sin(ufl.pi*x[0])\
    #        *(3*ufl.sin(ufl.pi*x[0])**2- 6*ufl.cos(ufl.pi*x[0])**2+ufl.pi**4*eps**2-ufl.pi**2)

    #d2W_test = 3/(ufl.pi**2)*(ufl.sin(ufl.pi*x[0]))**2-1

    return mu_test,mu_test_xx,d2W_test


def compute_solution(eps:float,dt:float,theta:float,t:float,T:float,
            initial_condition:callable,msh:Mesh,name:str,N:int,
            test:bool=False,comp_eigenvalue:bool=True,
            comp_discrete_lap:bool=True,
            offscreen:bool=False,plotting_3d:bool=True
            ,rtol_newton:float=1e-8, d:int=2,
            method:str=None,
            shift_numbers:list=None,
            alpha_penalty:float=None,r_tol_accept:float=None,
            maxiter:int=None,tol:float=None,p:float=np.inf,
            delta_diffusive:float=0.05,
            comp_residum:bool=True,
            msh_larger:Mesh=None,
            reload_time:float = -1,
            reload_time_res:float = -1):
    """


    Args:
        eps (float): epsilon 
        dt (float): time step 
        theta (float): discretization scheme 
        t (float): start time 
        T (float):end time 
        initial_condition (callable):initial condition 
        msh (Mesh): mesh 
        name (str): name to save 
        N (int): number of nodes on one side 
        test (bool, optional): if test to reconstruct solution. Defaults to False.
        comp_eigenvalue (bool, optional): if compunting eigenvalue. Defaults to True.
        comp_discrete_lap (bool, optional): if computing the dsicrete laplacian. Defaults to True.
        offscreen (bool, optional): if plot offscreen. Defaults to False.
        plotting_3d (bool, optional): plotting 3d or surface plot. Defaults to True.
        rtol_newton (float, optional): tolarance for newton solver. Defaults to 1e-8.
        d (int, optional): dimension, compatible with mesh implemented for 1 or 2. Defaults to 2.
        msh_list (list[Mesh], optional): list of meshes for eigenvalue. Defaults to None.
        method (str, optional): eigenvalue method. Defaults to None.
        shift_numbers (list, optional): list of shift values used for eigenvalue. Defaults to None.
        maxiter (int, optional):maxiter for eigenvalue problem. Defaults to None.
        tol (float, optional): tolerance for eigensolver. Defaults to None.
        p (float, optional): if factor in front of eigenvalue. Defaults to np.inf.
        alpha_penalty (float, optional): penalty for for eigenvalue. Defaults to None.
        r_tol_accept (float, optional): tolerance accept eigenvalue  . Defaults to None.
        delta_diffusive (float, optional): size of diffusive interface for plots. Defaults to 0.05.
        comp_residum (bool, optional): if computing residuals. Defaults to True.
        msh_larger (Mesh, optional): larger mesh for residuals. Defaults to None.
        reload_time (float, optional): reload time for solution. Defaults to -1.
        reload_time_res (float, optional): reload time for residuals. Defaults to -1.
    Args:

    """



    P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
    ME = FunctionSpace(msh, P1 * P1)

    # Trial and test functions of the space `ME` are now defined:
    q, v = ufl.TestFunctions(ME)



    u = Function(ME)  # current solution
    u0 = Function(ME)  # solution from previous converged step


    # Split mixed functions
    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)


    # Zero u
    u.x.array[:] = 0.0

    # Interpolate initial condition
    # sub extracts a sub  function (in this case c)
    u.sub(0).interpolate(lambda x :initial_condition(x,"np"))
    #TODO: understand
    u.x.scatter_forward()


    # Compute the chemical potential df/dc
    c = ufl.variable(c)
    f = ((c**2-1)**2)/4
    dfdc = ufl.diff(f, c)
    df2dc = ufl.diff(dfdc, c)

    # compute
    c0 = ufl.variable(c0)
    f0 = ((c0**2-1)**2)/4
    dfdc0 = ufl.diff(f0, c0)
    df2dc0 = ufl.diff(dfdc0, c0)



    # mu_(n+theta)
    mu_mid = (1.0 - theta) * mu0 + theta * mu

    # Weak statement of the equations
    F0 = inner(c, q) * dx - inner(c0, q) * dx \
        + dt*(eps**(-1))* inner(grad(mu_mid), grad(q)) * dx 

    if theta <1:
        F0 += dt*(1-theta)*(eps**(-3))*inner(df2dc0*mu0,q) *dx 

    if theta > 0:
        F0 += dt*theta*(eps**(-3))*inner(df2dc*mu,q) *dx


    F1 = inner(mu, v) * dx \
        - eps * inner(grad(c), grad(v)) * dx\
        - (eps**(-1))*inner(dfdc, v) * dx 

    F = F0 + F1

    # if reconstruct solution
    if test:
        x = ufl.SpatialCoordinate(msh)

        mu_test, mu_test_xx, d2W_test = get_test_values(eps, x, initial_condition)

        n = FacetNormal(msh)
        F0 +=  -dt*(eps**(-1))* (dot(grad(mu_test),n)* q) * ds 
    
        F_test = dt*eps**(-1)*(inner((mu_test_xx-eps**(-2)*d2W_test*mu_test),q)) *dx 
        F = F0 + F1 + F_test
        c_sol = initial_condition(x,"ufl")

    # Create nonlinear problem and Newton solver
    problem = NonlinearProblem(F, u)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = rtol_newton

    # We can customize the linear solver used inside the NewtonSolver by
    # modifying the PETSc options
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()


    # Get the sub-space for c and the corresponding dofs0 in the mixed space
    V0, dofs0 = ME.sub(0).collapse()
    V1, dofs1 = ME.sub(1).collapse()

    if have_pyvista:
        plotter,grid = create_background_plotter3D(V0,offscreen=offscreen,
                                save_name=f"cahn_hilliard_willmore_{name}_{eps}_N_{N}"
                                ,plot_3d=plotting_3d)
        update_background_plotter3D(0,u,dofs0,plotter,grid,
            plotting_3d=plotting_3d,delta_diffusive=delta_diffusive)

    c = u.sub(0)
    u0.x.array[:] = u.x.array
    if d == 1:
        print(f"h : {1/N} ")
    elif d == 2:
        print(f"h = {1/N*np.sqrt(2)}")
    print(f"dt = {dt}")

    if test:
        error = comp_norm(c_sol -c,msh,MPI.SUM)
        print(f"Error I_h u_0 - u_0 in L2: {error}")
        error = msh.comm.allreduce(assemble_scalar(form(dot(grad(c),n)**2 *ds)), MPI.SUM)**(1/2)
        print(f"Error partial_n c_sol = 0: {error}")



    n_steps = int(np.round(T/dt))
    if comp_eigenvalue:
        lmb_list = np.zeros(n_steps+1)
        lmb_list[0] = compute_eigenvalue(eps=eps,msh=msh,u=c,
                       maxiter=maxiter,tol=tol,r_tol_accept=r_tol_accept,method=method,
                       alpha_penalty=alpha_penalty,shift_numbers=shift_numbers,p=p)[0].real


    if comp_residum:
        eta_space_list = np.zeros(n_steps)
        eta_time_list  = np.zeros(n_steps)
        nu_space_list  = np.zeros(n_steps)
        nu_time_list   = np.zeros(n_steps)

    i = 0
    if comp_discrete_lap:
        discrete_lap_list = np.zeros(n_steps)
    while (t<=T-dt/2):

        if t<= reload_time+dt/2:
            with open(f"save_solution/solution_{N}_eps_{eps}_{dt}_{t}.pkl", 'rb') as f:
                u.x.array[:] = pickle.load(f)["u"][:]
            print(f"Step {int(np.round(t/dt))}")

        ######################## server
        #if t<= reload_time-dt/2:
        #    pass
        #elif t<= reload_time+dt/2:
        #    with open(f"save_solution/solution_c_{N}_eps_{eps}_{dt}.pkl", 'rb') as f:
        #        u.x.array[:] = pickle.load(f)["u"][:]
        #    with open(f"save_solution/solution_c0_{N}_eps_{eps}_{dt}.pkl", 'rb') as f:
        #        u.x.array[:] = pickle.load(f)["u"][:]
        #    ############################ server

        #    print(f"Step {int(np.round(t/dt))}")
        ######################## server

        else:
            r = solver.solve(u)
            u_sol_dict = {"u":u.x.array,"t":t,"N":N,"eps":eps}
            with open(f"save_solution/solution_{N}_eps_{eps}_{dt}_{t}.pkl",'wb') as f:
                pickle.dump(u_sol_dict,f)
            print(f"Step {int(np.round(t/dt))}: num iterations: {r[0]}")

            ######################## for server
            #with open(f"save_solution/solution_c_{N}_eps_{eps}_{dt}.pkl",'wb') as f:
            #    pickle.dump(u_sol_dict,f)
            #u_sol_dict = {"u":u0.x.array,"t":t,"N":N,"eps":eps}
            #with open(f"save_solution/solution_c0_{N}_eps_{eps}_{dt}.pkl",'wb') as f:
            #    pickle.dump(u_sol_dict,f)
            #################################
            if comp_discrete_lap:


                if t <= reload_time_res+dt/2:
                    with open(f'graphics/plot_residuals/discrete_lap_{name}_eps_{eps}_N_{N}_dt_{dt}.pkl', 'rb') as f:
                            discrete_lap_data = pickle.load(f)
                    
                
                else:
                    discrete_lap_data = {"discrete_lap_list":np.array(discrete_lap_list),
                                        "eps":eps,
                                        "N":N,
                                        "dt":dt}
                    with open(f'graphics/plot_residuals/discrete_lap_{name}_eps_{eps}_N_{N}_dt_{dt}.pkl', 'wb') as f:
                            pickle.dump(discrete_lap_data, f)
    
                    if abs(t)<dt/2:
                        discrete_lap = comp_discrete_laplace(c0,V0)
                        discrete_lap_list[0] = eps**2 *np.max(discrete_lap.x.array[:])
                    discrete_lap = comp_discrete_laplace(c,V0)
                    discrete_lap_list[i] = eps**2 *np.max(discrete_lap.x.array[:])
        

        if comp_residum:
            if t <= reload_time_res+dt/2:
                with open(f'graphics/plot_residuals/reisudal_eps_{eps}_N_{N}_dt_{dt}.pkl', 'rb') as f:
                        res_dict = pickle.load(f)
                        eta_space_list = res_dict["eta_space_list"]
                        eta_time_list = res_dict["eta_time_list"]
                        nu_space_list = res_dict["nu_space_list"]
                        nu_time_list = res_dict["nu_time_list"]

            else:
                eta_space,eta_time,nu_space,nu_time = compute_residum(u=u,u0=u0,msh=msh,
                        msh_larger=msh_larger,eps=eps,dt=dt,t=t,V0=V0,dofs0=dofs0,dofs1=dofs1)
                eta_space_list[i]=eta_space
                eta_time_list[i]=eta_time
                nu_space_list[i]=nu_space
                nu_time_list[i]=nu_time
                residual_data = {"eta_space_list":eta_space_list,
                                "eta_time_list":eta_time_list,
                                "nu_space_list":nu_space_list,
                                "nu_time_list":nu_time_list,
                                }
                with open(f'graphics/plot_residuals/reisudal_eps_{eps}_N_{N}_dt_{dt}.pkl', 'wb') as f:
                        pickle.dump(residual_data,f)


        if test:
            error = comp_norm(c-c_sol,msh,MPI.SUM)
            print(f"Error c -c_sol = {error}")

        u0.x.array[:] = u.x.array

        if comp_eigenvalue:

            if t  <= reload_time_res+dt/2: 
                with open(f'graphics/plot_eigenvalue/eigenvalue_{name}_eps_{eps}_N_{N}_dt_{dt}.pkl', 'rb') as f:
                    lmb_list = pickle.load(f)["lmb_list"]


            else:
                lmb_list[i] = compute_eigenvalue(eps=eps,msh=msh,
                            u=c,method=method,
                        maxiter=maxiter,tol=tol,r_tol_accept=r_tol_accept,alpha_penalty=alpha_penalty,p=p)[0].real
                print(f"Eigenvalue: {lmb_list[-1]}")

                eigenvalue_data = {"lmb_list": lmb_list,
                                "eps":eps,
                                "N":N,
                                "T": T,
                                "dt":dt}
                with open(f'graphics/plot_eigenvalue/eigenvalue_{name}_eps_{eps}_N_{N}_dt_{dt}.pkl', 'wb') as f:
                    pickle.dump(eigenvalue_data, f)



        # Update the plot window
        if have_pyvista:
            update_background_plotter3D(t,u,dofs0,plotter,grid,
            plotting_3d=plotting_3d,delta_diffusive=delta_diffusive)

        
        t += dt
        i += 1

    if have_pyvista:
        plotter.close()


        
    




if __name__ == "__main__":



    # Test d = 1. We need \partial_n u =0
    def initial_condition_test_1d(x:np.ndarray,mp:str):
        if mp == "np":
            return  (-1)/np.pi*np.cos(np.pi*x[0])
        elif mp == "ufl":
            return  (-1)/ufl.pi*ufl.cos(ufl.pi*x[0])

    test_1d = {
            "initial_condition" :initial_condition_test_1d, 
            "name"  : "test_sin_1d",   
            "eps"   : 0.5,  # surface parameter,
            "d"     : 1,
            "dt"    : 1e-4,
            "theta" : 1, # time stepping family, e.g. theta:1 -> backward Euler,
                # theta" :0.5 -> Crank-Nicholson
            "N"      :  64,
            "rtol_newton": 1e-9,
            "comp_eigenvalue": False,
            "t": 0.0,
            "test": True,
    }
    test_1d["T"] = test_1d["dt"] * 4



    # Test d = 2. We need \partial_n u =0
    def initial_condition_test_2d(x:np.ndarray,mp:str):
            return 0.5*x[0]**2 - 1/3*x[0]**3+ 0.5*x[1]**2 - 1/3*x[1]**3

    test_2d = {
            "initial_condition" :initial_condition_test_2d, 
            "name"  : "test_2d",   
            "eps"   : 0.5,  # surface parameter,
            "d"     : 2,
            "dt"    : 1e-3,
            "theta" : 1, # time stepping family, e.g. theta:1 -> backward Euler,
                # theta" :0.5 -> Crank-Nicholson
            "N"      :  64,
            "rtol_newton": 1e-9,
            "comp_eigenvalue": False,
            "t": 0.0,
            "test": True,
            "offscreen":False,
            "plotting_3d":True
    }
    test_2d["T"] = test_2d["dt"] * 50


    def initial_condition_tanh_1d(x:np.ndarray,mp:str,eps:float):
        if mp == "np":
            return  np.tanh(eps**(-1) * (x[0]-0.5))
        elif mp == "ufl":
            return  ufl.tanh(eps**(-1) * (x[0]-0.5))

    tanh_1d = {
            "name"  : "test_sin_1d",   
            "eps"   : 0.2,  # surface parameter,
            "d"     : 1,
            "dt"    : 1e-7,
            "theta" : 1, # time stepping family, e.g. theta:1 -> backward Euler,
                # theta" :0.5 -> Crank-Nicholson
            "N"      : 400, 
            "rtol_newton": 1e-9,
            "comp_eigenvalue": False,
            "t": 0.0,
            "test": False,
            "offscreen":False,
            "plotting_3d":False
    }
    tanh_1d["initial_condition"] = lambda x,mp:initial_condition_tanh_1d(x,mp,tanh_1d["eps"]) 
    tanh_1d["T"] = 20 * tanh_1d["dt"]


    # 2 circles
    def initial_condition_2circ_2d(x:np.ndarray,mp:str,eps:float):
        dist_to_center1 = (1/8-((x[0]-1/2)**2 + (x[1]-1/4)**2)**(1/2))
        dist_to_center2 = (1/8-((x[0]-1/2)**2 + (x[1]-3/4)**2)**(1/2))
        if mp == "np":
            return  (dist_to_center1>=dist_to_center2)\
                    *np.tanh(eps**(-1)*(np.sqrt(2))**(-1)* (1/8-((x[0]-1/2)**2 \
                    + (x[1]-1/4)**2)**(1/2)))+\
                    (dist_to_center1<dist_to_center2)*\
                    np.tanh(eps**(-1)*(np.sqrt(2))**(-1)* (1/8-((x[0]-1/2)**2 \
                    + (x[1]-3/4)**2)**(1/2)))
        elif mp == "ufl":
            return  (dist_to_center1>=dist_to_center2)\
                    *ufl.tanh(eps**(-1)*(np.sqrt(2))**(-1)* (1/8-((x[0]-1/2)**2 \
                    + (x[1]-1/4)**2)**(1/2)))+\
                    (dist_to_center1<dist_to_center2)*\
                    ufl.tanh(eps**(-1)*(np.sqrt(2))**(-1)* (1/8-((x[0]-1/2)**2 \
                    + (x[1]-3/4)**2)**(1/2)))

    case_2_circ = {
            "name"  : "2circ",   
            "eps"   : 1.4**(-3)*0.2,  # surface parameter,
            "d"     : 2,
            "dt"    : 5e-6,
            "theta" : 1, # time stepping family, e.g. theta:1 -> backward Euler,
                # theta" :0.5 -> Crank-Nicholson
            "N"      :  17,
            "rtol_newton": 1e-9,
            "comp_eigenvalue": True,
            "t": 0.0,
            "test": False,
            "offscreen":False,
            "plotting_3d":True,
            "delta_diffusive":0.03,
            "comp_residum":True,
            #"reload_time":6.5e-5
            "reload_time":-1
    }
    #case_2_circ["T"] = case_2_circ["dt"] *100 
    case_2_circ["T"] = case_2_circ["dt"] *30
    case_2_circ["initial_condition"]\
    = lambda x,mp:initial_condition_2circ_2d(x,mp,case_2_circ["eps"])

    ##################################################################
    ##################################################################
    param_dict = case_2_circ
    ##################################################################
    ##################################################################



    msh_larger_factor = 2

    if param_dict["d"] == 1 :
        msh = create_interval(MPI.COMM_WORLD,param_dict["N"],[0,1])
        msh_larger = create_interval(MPI.COMM_WORLD,msh_larger_factor*param_dict["N"],[0,1])
    elif param_dict["d"] == 2:
        msh = create_unit_square(MPI.COMM_WORLD, param_dict["N"], param_dict["N"], CellType.triangle)
        msh_larger = create_unit_square(MPI.COMM_WORLD, msh_larger_factor*param_dict["N"],
                                       msh_larger_factor* param_dict["N"], CellType.triangle)
    param_dict["msh"] = msh
    param_dict["msh_larger"] = msh_larger

    #eigsolver
    N_list = [90,100,128]
    param_dict["method"] = "ciarlet_raviart"
    param_dict["shift_numbers"] = [-4*param_dict["eps"]**(-i) for i in reversed(range(4,5))]
    param_dict["shift_numbers"] += [4*param_dict["eps"]**(-i) for i in range(1,5)]
    param_dict["alpha_penalty"] = 50
    param_dict["r_tol_accept"] = 1e-5
    param_dict["tol"] = 1e-5
    param_dict["maxiter"] = 10000

    compute_solution(**param_dict)