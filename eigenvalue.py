import numpy as np

import ufl
from dolfinx import  plot
from dolfinx.fem.petsc import (assemble_matrix, LinearProblem)
from dolfinx.fem import Function, FunctionSpace,form,assemble_scalar
from dolfinx.mesh import create_unit_square,create_interval,Mesh
from ufl import ds,dS,dx, grad, inner,dot,div,jump,avg,FacetNormal,CellDiameter

from mpi4py import MPI
from slepc4py import SLEPc
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  

import pyvista as pv





def compute_eigenvalue(eps:float, msh:Mesh,
                u:Function,  p:float=np.inf,
                maxiter:int=10,tol:float=1e-14, tol_ksp:float=1e-14,
                shift_numbers:list[float]=[100,-100],r_tol_accept:float=1e-7,
                alpha_penalty:float=50,u_lap:Function=None,method:str="ciarlet_raviart"):
     """
     computes the solution to the generalized eigenvalue problem

     Args:
          eps (float): espilon value 
          msh (Mesh): mesh 
          V (FunctionSpace): function space 
          u (callable): Linear Element function 
          maxiter (int, optional): maxiter*A.shape. Defaults to 10.
          p (int, optional): factor of norm of laplacian, such that
            (1-\epsilon^p) * \norm{\lap}. Defaults to np.inf.
          tol (float, optional):relative tolerance of eigesolver. Defaults to 1e-14.
          tol_ksp (float, optional): tolerance of linear equation solver. Defaults to 1e-14.
          shift_numbers (list[float], optional): shift numbers. Defaults to [100,-100].
          r_tol_accept (float, optional): test the residual and accept with this tolerance. Defaults to 1e-7.
          alpha_penalty (float, optional): penalty for penalty method. Defaults to 50.
          u_lap (Function, optional): laplcian of u, otherwise will be computed. Defaults to None.
          method (str, optional): penalty or ciarlet_raviart Defaults to "ciarlet_raviart".

     Returns:
          lmb (float): smallest real eigenvalue  
          v (np.ndarray): corresponding eigenvalue


     """

     if method == "penalty":

          alpha = ScalarType(alpha_penalty)


          P2 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 2)
          V = FunctionSpace(msh,P2)

          

          e = ufl.TrialFunction(V)
          v = ufl.TestFunction(V)
          h = CellDiameter(msh)
          n = FacetNormal(msh)
          h_avg = (h('+') + h('-')) / 2.0

          eps = ScalarType(eps)
          p = ScalarType(p)


          def hess ( v ) :
               return grad ( grad ( v ) )
          F2 =  (1-eps**p)*inner(hess((e)),hess (v)) * dx \
               -(1-eps**p)*inner(avg(dot(hess((e))* n, n)), jump(grad(v), n)) * dS \
               -(1-eps**p)*inner(dot(grad ((e)), n), dot(hess (v)* n, n)) * ds \
               -(1-eps**p)*inner(jump(grad ((e)), n), avg(dot(hess(v)* n, n)))* dS \
               -(1-eps**p)*inner(dot(hess((e))* n, n), dot(grad(v), n)) * ds \
               +(1-eps**p)*alpha/ h_avg* inner (jump(grad((e)), n), jump(grad(v), n)) * dS \
               +(1-eps**p)*alpha / h * inner(dot(grad((e)), n), dot(grad(v), n)) * ds                   
          
          F1  = form( inner(e,v) * dx) 


          c = ufl.variable(u)
          W = (c**2-1)**2/4
          dWdc = ufl.diff(W, c)
          dW2dc = ufl.diff(dWdc, c)
          dW3dc = ufl.diff(dW2dc, c)
          dW4dc = ufl.diff(dW3dc, c)
          
          f = dWdc*dW2dc
          dFdc = ufl.diff(f,c)
          
          
          F3 = +2/(eps**(2))*dW2dc*inner(grad(e),grad(v))* dx\
              +1/(eps**(2))*dW3dc*inner(grad(e),grad(u))*v* dx \
              +1/(eps**(2))*dW3dc*inner(grad(v),grad(u))*e* dx

          
          F4 =  1/(eps**(4)) * inner(dFdc*e,v) * dx
          
          if u_lap != None:
               F5 = -1/(eps**2)* dW3dc*inner(u_lap*e,v)*dx
          else:
               F5 = +1/(eps**(2))* dW4dc*inner(grad(u),grad(u))*e*v *dx\
                    +1/(eps**(2))* dW3dc*inner(grad(u),grad(e))*v *dx\
                    +1/(eps**(2))* dW3dc*inner(grad(u),grad(v))*e *dx\
                    -1/(eps**(2))* dW3dc*inner(grad(u),n)*e*v * ds\

          #Test u=1
          #F2345 = form (F2 +  -2/(eps**(2))*inner(div(grad(e)),v) * dx+
          #                    -2/(eps**(2))*inner(div(grad(v)),e) * dx\
          #                    +(4/eps**(4))* inner(e,v) *dx )
          #Test u=0
          #F2345 = form (F2 +  +2/(eps**(2))*inner(div(grad(e)),v) * dx+
          #                    +(1/eps**(4))* inner(e,v) *dx )

          F2345 = form(F2+F3 + F4 + F5)


          A = assemble_matrix(F1)
          A.assemble()

          B2 = assemble_matrix(F2345)
          B2.assemble()



          lmb = np.inf
          eigenvector = PETSc.Vec().createSeq(A.size[0])

          for shift in shift_numbers:

               # Setup the eigensolver
               eigsolver = SLEPc.EPS().create(msh.comm)
               ###############################
               eigsolver.setOperators(B2,A)
               eigsolver.setProblemType( SLEPc.EPS.ProblemType.GHIEP )
               #eigsolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

               ####################
               eigsolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
               #eigsolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
               eigsolver.setTarget(shift)
               
               st = eigsolver.getST()

               # Set shift-and-invert transformation
               st.setType(SLEPc.ST.Type.SINVERT)
               st.setShift(shift)
               ksp = st.getKSP()
               ksp.setTolerances(rtol=tol_ksp)#,maxits=10000)
               ##############################

               #eigsolver.setTrueResidual(True)
               tol = tol
               max_it = maxiter
               eigsolver.setTolerances(tol=tol,max_it=max_it)
               #eigsolver.setPurify(False)

               eigsolver.setConvergenceTest(SLEPc.EPS.Conv.ABS)

               # Solve the eigensystem
               eigsolver.setUp()
               eigsolver.solve()
               #eigsolver.view()
               eigsolver.errorView()
               #it = eigsolver.getIterationNumber()
               #print(f"Number of iterations {it}")

               nconv = eigsolver.getConverged()
               if nconv <=0:
                    print("No eigenvalues converged")
               else:
                    for i in range(nconv):
                         eig = eigsolver.getEigenpair(i)
                         eig_vec = PETSc.Vec().createSeq(A.size[1])
                         eigsolver.getEigenvector(i,eig_vec)

                         res = eigsolver.computeError(i)

                         Av = PETSc.Vec().createSeq(A.size[0])
                         A.mult(eig_vec,Av)

                         Bv = PETSc.Vec().createSeq(B2.size[0])
                         B2.mult(eig_vec,Bv)

                         res_real = (eig.real*Av-Bv).norm(PETSc.NormType.INFINITY)
                         res = eigsolver.computeError(i)


                         if (res_real <= r_tol_accept and eig.real < lmb.real)\
                              and np.abs(eig-eig.conjugate()) <= 1e-10:
                              lmb = eig
                              eigenvector = eig_vec.copy()


               tol_given, maxit_given = eigsolver.getTolerances()
               #print(f"Stopping condition: tol={tol_given}, maxit={maxit_given}" )
               #print(f"Reason for Conv {eigsolver.getConvergedReason()}")
               eigsolver.destroy()


          Av = PETSc.Vec().createSeq(A.size[0])
          A.mult(eigenvector,Av)

          Bv = PETSc.Vec().createSeq(B2.size[0])
          B2.mult(eigenvector,Bv)

          res_real = (lmb.real*Av-Bv).norm(PETSc.NormType.INFINITY)

          A.destroy()
          B2.destroy()


     elif method == "ciarlet_raviart":

          P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
          V = FunctionSpace(msh,P1*P1)

          e_lap,e = ufl.TrialFunctions(V)
          q,v = ufl.TestFunctions(V)
          
          n = FacetNormal(msh)
          h = CellDiameter(msh)

          c = ufl.variable(u)
          W = (c**2-1)**2/4
          dWdc = ufl.diff(W, c)
          dW2dc = ufl.diff(dWdc, c)
          dW3dc = ufl.diff(dW2dc, c)
          dW4dc = ufl.diff(dW3dc, c)
          
          f = dWdc*dW2dc
          dFdc = ufl.diff(f,c)

          
          F3 = +2/(eps**(2))*dW2dc*inner(grad(e),grad(v))* dx\
              +1/(eps**(2))*dW3dc*inner(grad(e),grad(u))*v* dx \
              +1/(eps**(2))*dW3dc*inner(grad(v),grad(u))*e* dx
          

          
          F4 =  1/(eps**(4)) * inner(dFdc*e,v) * dx
          
          if u_lap != None:
               F5 = -1/(eps**2)* dW3dc*inner(u_lap*e,v)*dx
          else:
               F5 = +1/(eps**(2))* dW4dc*inner(grad(u),grad(u))*e*v *dx\
                    +1/(eps**(2))* dW3dc*inner(grad(u),grad(e))*v *dx\
                    +1/(eps**(2))* dW3dc*inner(grad(u),grad(v))*e *dx\
                    -1/(eps**(2))*dW3dc*inner(grad(u),n)*e*v * ds\
               


          # e = lap e_lap
          h_avg = (h('+') + h('-')) / 2.0
          F2 = -(1-eps**p)*inner(e_lap,q)*dx +(1-eps**p)*inner(grad(e),grad(q))*dx \
               +(1-eps**p)*inner(grad(e_lap),grad(v))*dx

          F1 = form(inner(e,v)*dx)


          #Test u=1
          #F2345 = form (F2 +  -2/(eps**(2))*inner(div(grad(e)),v) * dx+
          #                    -2/(eps**(2))*inner(div(grad(v)),e) * dx\
          #                    +(4/eps**(4))* inner(e,v) *dx )
          #Test u=0
          #F2345 = form (F2 +  +2/(eps**(2))*inner(div(grad(e)),v) * dx+
          #                    +(1/eps**(4))* inner(e,v) *dx )

          F2345 = form(F2+F3 + F4 + F5)


          A = assemble_matrix(F1)
          A.assemble()

          B2 = assemble_matrix(F2345)
          B2.assemble()


          lmb = np.inf
          eigenvector = PETSc.Vec().createSeq(A.size[0])

          for shift in shift_numbers:

               # Setup the eigensolver
               eigsolver = SLEPc.EPS().create(msh.comm)
               ###############################
               eigsolver.setOperators(B2,A)
               eigsolver.setProblemType( SLEPc.EPS.ProblemType.GHIEP )
               #eigsolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

               #####################
               eigsolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
               #eigsolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
               eigsolver.setTarget(shift)
               
               st = eigsolver.getST()

               # Set shift-and-invert transformation
               st.setType(SLEPc.ST.Type.SINVERT)
               st.setShift(shift)
               ksp = st.getKSP()
               ksp.setTolerances(rtol=tol_ksp)#,maxits=10000)
               ###############################

               #eigsolver.setTrueResidual(True)
               tol = tol
               max_it = maxiter
               eigsolver.setTolerances(tol=tol,max_it=max_it)
               #eigsolver.setPurify(False)

               #eigsolver.setDimensions(nev=1)
               eigsolver.setConvergenceTest(SLEPc.EPS.Conv.ABS)

               # Solve the eigensystem
               eigsolver.setUp()
               eigsolver.solve()
               #eigsolver.view()
               #eigsolver.errorView()
               it = eigsolver.getIterationNumber()
               #print(f"Number of iterations {it}")

               nconv = eigsolver.getConverged()
               if nconv <=0:
                    print("No eigenvalues converged")
               else:
                    for i in range(nconv):
                         eig = eigsolver.getEigenpair(i)
                         eig_vec = PETSc.Vec().createSeq(A.size[1])
                         eigsolver.getEigenvector(i,eig_vec)

                         res = eigsolver.computeError(i)

                         Av = PETSc.Vec().createSeq(A.size[0])
                         A.mult(eig_vec,Av)

                         Bv = PETSc.Vec().createSeq(B2.size[0])
                         B2.mult(eig_vec,Bv)

                         res_real = (eig.real*Av-Bv).norm(PETSc.NormType.INFINITY)
                         res = eigsolver.computeError(i)

                         if (res_real <= r_tol_accept and eig.real < lmb.real)\
                              and np.abs(eig-eig.conjugate()) <= 1e-10:
                              lmb = eig
                              eigenvector = eig_vec.copy()


               tol_given, maxit_given = eigsolver.getTolerances()
               #print(f"Stopping condition: tol={tol_given}, maxit={maxit_given}" )
               #print(f"Reason for Conv {eigsolver.getConvergedReason()}")
               eigsolver.destroy()


          Av = PETSc.Vec().createSeq(A.size[0])
          A.mult(eigenvector,Av)

          Bv = PETSc.Vec().createSeq(B2.size[0])
          B2.mult(eigenvector,Bv)

          res_real = (lmb.real*Av-Bv).norm(PETSc.NormType.INFINITY)

          A.destroy()
          B2.destroy()

     return lmb,eigenvector


if __name__ == "__main__":

     u_0c = {
          "N": 512,
          "eps": 0.05,
          "u_value": 0,
          "u_const": True,
          "p":np.inf,
          "name_picture":"0",
          }
     u_0c["name"] = f"eigenvector{u_0c['u_value']}_N_{u_0c['N']}_eps_{u_0c['eps']}"

     u_1c = {
          "N":700, 
          "eps": 0.1,
          "u_value": 1,
          "u_const": True,
          "p":np.inf,
          "name_picture":"1",
          }
     u_1c["name"] = f"eigenvector{u_1c['u_value']}_N_{u_1c['N']}_eps_{u_1c['eps']}"

     u_tanh = {
          "N":1024, 
          "eps": 0.07,
          "u_const": False,
          "p":np.inf,
          "name_picture":"tanh",
          }
     u_tanh["u_value"] = lambda x: np.tanh(u_tanh["eps"]**(-1) * (x[0]-0.5))
     u_tanh["name"] = f"eigenvector{u_tanh['u_value']}_N_{u_tanh['N']}_eps_{u_tanh['eps']}"

     u_sin = {
          "N": 128,
          "eps": 0.1,
          "p":np.inf,
          "u_value":  lambda x: np.sin(6*np.pi*((0.5-x[0])**2+(0.5-x[1])**2)**(0.5)),
          "u_const": False,
          "name_picture":"sin",
          }
     u_sin ["name"] = f"eigenvector_N_{u_sin['N']}_eps_{u_sin['eps']}"


     u_sin_high = {
          "N": 32,
          "eps": 0.1,
          "u_value":  lambda x: np.sin(100*np.pi*((0.5-x[0])**2+(0.5-x[1])**2)**(0.5)),
          "u_const": False,
          "p":np.inf,
          "name_picture":"sin high frequenz",
          }
     u_sin_high ["name"] = f"eigenvector_N_{u_sin_high['N']}_eps_{u_sin_high['eps']}"

     
     u_circ = {
          "N": 400,
          "eps": 0.08,
          "u_const": False,
          "p":np.inf,
          "name_picture":"circle",
          }
     u_circ ["name"] = f"eigenvector_circ_N{u_circ['N']}_eps_{u_circ['eps']}"
     u_circ["u_value"] = lambda x : np.tanh((u_circ["eps"])**(-1)*(1+1/4)**(-1)* (1/4-((x[0]-1/2)**2 + (x[1]-1/2)**2)**(1/2)))



     def initial_condition_2circ_2d(x:np.ndarray,eps:float):
        dist_to_center1 = (1/10-((x[0]-1/2)**2 + (x[1]-1/4)**2)**(1/2))
        dist_to_center2 = (1/10-((x[0]-1/2)**2 + (x[1]-3/4)**2)**(1/2))
        return  (dist_to_center1>=dist_to_center2)\
                *np.tanh(eps**(-1)*(1+1/8)**(-1)* (1/8-((x[0]-1/2)**2 + (x[1]-1/4)**2)**(1/2)))+\
                (dist_to_center1<dist_to_center2)*\
                np.tanh(eps**(-1)*(1+1/8)**(-1)* (1/8-((x[0]-1/2)**2 + (x[1]-3/4)**2)**(1/2)))
     u_2circ = {
          "N": 100,
          "eps": 0.1,
          "u_const": False,
          "p":np.inf,
          "name_picture":"2 circles",
          }
     u_2circ ["name"] = f"eigenvector_2circ_N{u_2circ['N']}_eps_{u_2circ['eps']}"
     u_2circ["u_value"] = lambda x:initial_condition_2circ_2d(x,u_2circ["eps"])


     ##########################################################
     param_dict = u_2circ
     ##########################################################
     save_plot = False
     #method = "penalty"
     method = "ciarlet_raviart"
     d = 2

     N = param_dict["N"]
     if d == 1:
          msh = create_interval(MPI.COMM_WORLD,N,[0,1]) 
          alpha_penalty = 50


     elif d == 2:
          msh = create_unit_square(MPI.COMM_WORLD,N,N)
          alpha_penalty =  50

     param_dict["alpha_penalty"] = alpha_penalty


     P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
     V = FunctionSpace(msh,P1 )
     u = Function(V)


     if param_dict["u_const"]:
          u.x.set(param_dict['u_value'])
     else:
          u.interpolate(param_dict['u_value'])
     param_dict.pop("u_const",None)
     param_dict.pop("u_value",None)
     param_dict.pop("N",None)
     param_dict["u"] = u

     param_dict["r_tol_accept"] = 1e-2
     param_dict["shift_numbers"] = [-4*param_dict["eps"]**(-i) for i in reversed(range(4,5))]
     #param_dict["shift_numbers"] += [4*param_dict["eps"]**(-i) for i in range(1,5)]



     param_dict["msh"] = msh
     param_dict["maxiter"] = 100000
     param_dict["tol"] = 1e-8
     param_dict["tol_ksp"] = 1e-14
     param_dict["method"] = method


     param_dict["msh"] = msh
     name_picture = param_dict["name_picture"]
     name = param_dict["name"]
     param_dict.pop("name",None)
     param_dict.pop("name_picture",None)
     

     lmb, v = compute_eigenvalue(**param_dict)
     print(f"The smallest eigenvalues are : {lmb}")
     if method == "penalty":

          P2 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 2)
          V = FunctionSpace(msh,P2)

          # plot eigenvalue
          plotter = pv.Plotter(off_screen=save_plot)
          topology, cell_types, x = plot.create_vtk_mesh(V)
          grid = pv.UnstructuredGrid(topology, cell_types, x)

          grid.point_data["e"] = v.array[:] 
          grid.set_active_scalars("e")

          warped = grid.warp_by_scalar()
          plotter.add_mesh(warped)
          plotter.add_text(f"Lambda = ${lmb:1.2e},\
                         N = {N}, eps = {param_dict['eps']}, u = {name_picture}")
          plotter.show(screenshot=f"graphics/{name}.png") 


     elif method == "ciarlet_raviart":
          ME = FunctionSpace(msh,P1*P1)
          V0, dofs = ME.sub(0).collapse()

          # plot eigenvalue
          plotter = pv.Plotter(off_screen=save_plot)
          topology, cell_types, x = plot.create_vtk_mesh(V)
          grid = pv.UnstructuredGrid(topology, cell_types, x)

          grid.point_data["e"] = v.array[dofs] 
          grid.set_active_scalars("e")

          warped = grid.warp_by_scalar()
          plotter.add_mesh(warped)
          plotter.add_text(f"Lambda = ${lmb:1.2e},\
                         N = {N}, eps = {param_dict['eps']}, u = {name_picture}")
          plotter.show(screenshot=f"graphics/{name}.png") 
