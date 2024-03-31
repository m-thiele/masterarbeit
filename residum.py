from typing import Union
from dolfinx import log, plot
from dolfinx.fem import Function, FunctionSpace,form,assemble_scalar
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, create_interval,Mesh
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx,ds,dS, grad, inner, FacetNormal,CellDiameter, dot, div, jump,avg
from dolfinx.io import XDMFFile
from matplotlib.colors import ListedColormap

import ufl
import numpy as np


from eigenvalue import compute_eigenvalue

from mpi4py import MPI
from petsc4py import PETSc
#imageio.plugins.ffmpeg.download()




def compute_form(form_comp:form,msh:Mesh)->float:
    """ computes a form on a given mesh

    Args:
        form_comp (form): form to compute
        msh (Mesh):msh 

    Returns:
        float: value
    """


    comp_form = msh.comm.allreduce(assemble_scalar(form_comp), op=MPI.SUM)
    return comp_form


def eta_pois_h2(fh:Function,sh:Function,msh:Mesh)-> float:
    """ computes the a posteriori error estimator with 
    right hand side fh, function sh and mesh msh

    Args:
        fh (Function): right side
        sh (Function): discrete solution 
        msh (Mesh): mesh


    Returns:
        float: value
    """

    h = CellDiameter(msh)
    h_avg = (h('+') + h('-')) / 2.0
    n = FacetNormal(msh)

    eta_pois_form1 = form(
        h**4*fh**2*dx
    )
    eta_pois_form2 = form(
        h_avg**3*jump(grad(sh),n)**2*dS
        +h**3*dot(grad(sh),n)**2*ds
    )

    eta_f = compute_form(eta_pois_form1,msh)**(1/2)
    eta_jump = compute_form(eta_pois_form2,msh)**(1/2)
    eta_pois = eta_f+eta_jump
    return eta_pois


def comp_discrete_laplace(ch:Function,V0:FunctionSpace)-> Function:
    """ solves for the discrete laplacian 

    Args:
        ch (Function): function of which laplacian
        V0 (FunctionSpace): P1 function space subordianted to mesh

    Returns:
        Function: discrete laplacian
    """

    v = ufl.TrialFunction(V0)
    q = ufl.TestFunction(V0)
    a = inner(v,q) * dx
    b = inner(grad(ch),grad(q)) * dx
    problem = LinearProblem(a, b, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    discrete_lap = problem.solve()
    return discrete_lap


def comp_m0(eps:float,c0:Function,dfdc0:Function,V0:FunctionSpace)->Function:
    """
    solve (mu^0,q) = (grad u^0,grad q) + (W'(u^0),q)

    Args:
        eps (float): epsilon
        c0 (Function): initial function
        dfdc0 (Function): derivative of f
        V0 (FunctionSpace): function space

    Returns:
        Function: mu^0
    """

    v = ufl.TrialFunction(V0)
    q = ufl.TestFunction(V0)
    a = inner(v,q) * dx
    b = eps*inner(grad(c0),grad(q)) * dx + eps**(-1) *inner(dfdc0,q)*dx
    problem = LinearProblem(a, b, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    mu0 = problem.solve()
    return mu0




def comp_projection(c:Function,V0:FunctionSpace)-> Function:
    """
    computes projection

    Args:
        c (Function): function
        V0 (FunctionSpace): function space

    Returns:
        Function: projection
    """

    v = ufl.TrialFunction(V0)
    q = ufl.TestFunction(V0)
    a = inner(v,q) * dx
    b = inner(c,q) * dx
    problem = LinearProblem(a, b, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    c_proj = problem.solve()
    return c_proj


def comp_hatu(discrete_lap:Function,msh:Mesh)->Function:
    """ computes space recosntruction

    Args:
        discrete_lap (Function): discrete laplacian
        msh (Mesh): mesh

    Returns:
        Function : hatu
    """


    P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
    V = FunctionSpace(msh,P1)
    v = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    a = inner(grad(v),grad(q)) * dx
    rhs = Function(V)
    rhs.interpolate(discrete_lap)
    b = inner(rhs,q) * dx
    problem = LinearProblem(a, b, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    hatu = problem.solve()
    return hatu

def compute_residum(u:Function,u0:Function,msh:Mesh,msh_larger:Mesh,eps:float,dt:float,t:float,V0:FunctionSpace,dofs0,dofs1):
    """

    Args:
        u (Function): _description_
        u0 (Function): _description_
        msh (Mesh): _description_
        eps (float): _description_
        dt (float): _description_
    """

    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)
    uhk_linfty = np.max(u.x.array[dofs0])
    uhk0_linfty = np.max(u0.x.array[dofs0])

    c = ufl.variable(c)
    f = ((c**2-1)**2)/4
    dfdc = ufl.diff(f, c)
    df2dc = ufl.diff(dfdc, c)

    # compute
    c0 = ufl.variable(c0)
    f0 = ((c0**2-1)**2)/4
    dfdc0 = ufl.diff(f0, c0)
    df2dc0 = ufl.diff(dfdc0, c0)

    
    h = CellDiameter(msh)
    n = FacetNormal(msh)
    h_avg = (h('+') + h('-')) / 2.0



    # eta
    # eta_dis_space
    eta_dis_space_form1 = form(
        eps**2*(1/dt)**2*(h**4)*(c-c0)**2 *dx
    )

    eta_dis_space_form2 = form(
        +(h_avg**3)*jump(grad(mu),n)**2 *dS
        +(h**3)*dot(grad(mu),n)**2 *ds
    )

    f2_mu_proj = comp_projection(df2dc*mu,V0)
    eta_dis_space_form3 = form(
        eps**(-2)*(h**4)*(f2_mu_proj)**2 *dx
    )

    eta_dis_space = compute_form(eta_dis_space_form1,msh)**(1/2)\
                    + compute_form(eta_dis_space_form2,msh)**(1/2)\
                    + compute_form(eta_dis_space_form3,msh)**(1/2)
    

    # eta_u_space
    discre_lap_c = comp_discrete_laplace(c,V0)
    discre_lap_c0 = comp_discrete_laplace(c0,V0)
    discre_lap_mu = comp_discrete_laplace(mu,V0)



    #hatu = comp_hatu(discre_lap_c,msh_larger)
    #hatu.x.array[:] += -compute_form(form(hatu*dx),msh_larger) +compute_form(form(c*dx),msh)
    #hatu0 = comp_hatu(discre_lap_c0,msh_larger)
    #hatu0.x.array[:] += -compute_form(form(hatu0*dx),msh_larger) +compute_form(form(c0*dx),msh)
    #u_linfty = max(np.max(np.abs(hatu.x.array)),np.max(np.abs(hatu0.x.array)))
    u_linfty = max(uhk_linfty,uhk0_linfty)
    rho_k = max(u_linfty,uhk_linfty,uhk0_linfty)


    eta_pois_c = eta_pois_h2(discre_lap_c,c,msh)
    eta_pois_c0 = eta_pois_h2(discre_lap_c0,c0,msh)
    eta_pois_dt_c = eta_pois_h2( 1/dt*(discre_lap_c-discre_lap_c0),1/dt*(c-c0),msh)


    mu_l3 = compute_form(form(abs(mu)**3*dx),msh)**(1/3)

    eta_pois_mu = eta_pois_h2(discre_lap_mu,mu,msh)
    eta_u_space =    eps*eta_pois_dt_c +eps**(-2)* mu_l3* 6*rho_k * (eta_pois_c+eta_pois_c0)



    # eta_mu_space
    eta_mu_space = (eps**(-2)*(max(3*u_linfty**2-1,1))+1)*eta_pois_mu 


    #eta_projection
    eta_proj = compute_form(form((f2_mu_proj-df2dc*mu)**2*dx),msh_larger)**(1/2)

    # eta_time
    eta_time = eps**(-2)*(mu_l3)*compute_form(form((c-c0)**2*dx),msh)**(1/2)

    eta_space = eta_dis_space + eta_u_space + eta_mu_space + eta_proj


    # nu 
    #  nu_u_space
    max_df2dc = max(np.abs(3*rho_k**2-1),1)
    nu_u_space = eps**(-1)*max_df2dc*(eta_pois_c+eta_pois_c0)


    #  nu_mu_space
    nu_mu_space = eta_pois_mu


    #  nu_proj
    df2dc_proj = comp_projection(dfdc,V0)
    df2dc0_proj = comp_projection(dfdc,V0)
    nu_proj = eps**(-1)*(compute_form(form((df2dc_proj-dfdc)**2*dx),msh_larger)**(1/2) +\
                compute_form(form((df2dc0_proj-dfdc)**2*dx),msh_larger)**(1/2))


    #  nu_time
    #nu_time = eps * compute_form(form((discre_lap_c-discre_lap_c0)**2*dx),msh)**(1/2) +eps**(-1)*max_df2dc*compute_form(form((u-u0)**2*dx),msh)**(1/2)
    if t-dt/2<=0:
        mu0 = comp_m0(eps,c0,dfdc0,V0)
        print("".center(80,"-"))
        print(f"mu-mu0 {compute_form(form((mu-mu0)**2*dx),msh)**(1/2)}")
        print(f"mu0 {compute_form(form((mu0)**2*dx),msh)**(1/2)}")
        print(f"dicrete lap {compute_form(form((discre_lap_c0)**2*dx),msh)**(1/2)}")
        print(f"mu {compute_form(form((mu)**2*dx),msh)**(1/2)}")
        print("".center(80,"-"))
    nu_time = compute_form(form((mu-mu0)**2*dx),msh)**(1/2)+eps**(-1)*max_df2dc*compute_form( form((c-c0)**2*dx),msh)**(1/2) + eps**(-1)*compute_form( form( ((c**3-c)-(c0**3-c0))**2*dx),msh)**(1/2)

    

    nu_space = nu_u_space+nu_mu_space+nu_proj


    print(f"eta_space {eta_space}")
    print(f"eta_time {eta_time}")
    print(f"nu_space {nu_space}")
    print(f"nu_time {nu_time}")
    return eta_space,eta_time,nu_space,nu_time



