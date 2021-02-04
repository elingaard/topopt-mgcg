import os
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from FEA import FEmesh,LinearElasticity,SparseMGCG
from time import perf_counter

def build_filter(rmin,size):
    """
    Build a filter of the form:
    (sum_{i in N_i} w(x_i)*y)/(sum_{i in N_i} w(x_i))
    where N_i denotes the neighbourhood of element i

    Args:
    rmin: filter radius
    size(tuple): tuple with array size in x- and y direction

    Returns:
    kernel(NxN array): filter kernel where M is dependent on the filter radius
    Nsum(NxM array): sum within each neighbourhood in the filter
    """

    nely,nelx = size
    cone_kernel_1d = np.arange(-np.ceil(rmin)+1,np.ceil(rmin))
    [dy,dx] = np.meshgrid(cone_kernel_1d,cone_kernel_1d)
    cone_kernel_2d = np.maximum(0,rmin-np.sqrt(dx**2+dy**2))
    Nsum = convolve(np.ones((nely,nelx)),cone_kernel_2d,mode='constant',cval=0)

    return cone_kernel_2d, Nsum

def run(args):
    mesh = FEmesh(args.nelx,args.nely)
    fea = LinearElasticity(mesh,penal=args.penal)
    # BCs and load (Michell cantilever)
    sBlockY = 1/10
    ele_start = int((mesh.nelx-1)*mesh.nely+mesh.nely//2 - np.round(mesh.nely*sBlockY/2+1e-9))
    ele_end = int((mesh.nelx-1)*mesh.nely+mesh.nely//2 + np.round(mesh.nely*sBlockY/2+1e-9))
    ele_vec = np.arange(ele_start,ele_end)[:,np.newaxis]
    face_vals = 2*np.ones(ele_vec.shape).astype(np.int)
    face_mag = np.tile([0,-1],(ele_vec.size,1))
    fea.insert_wall_boundary(wall_pos=0,wall_ax='y',bound_dir='xy')
    fea.insert_face_forces(ele_vec,face_vals,face_mag)
    if args.solver=="mgcg":
        mgcg = SparseMGCG(mesh.nelx,mesh.nely,n_levels=args.nl,njac=1)

    filt_kernel, Nsum_filt = build_filter(args.rmin,size=(mesh.nely,mesh.nelx))

    # initialize design variables
    x = np.ones((mesh.nely,mesh.nelx))*args.volfrac
    beta = 1
    if args.ft_type=='sensitivity' or args.ft_type=='density':
        xPhys = x.copy()
    elif args.ft_type=='heaviside':
        xTilde = x.copy()
        xPhys = 1-np.exp(-beta*xTilde)+xTilde*np.exp(-beta)

    # initialize plotting
    plt.ion()
    fig,ax = plt.subplots(1,1,figsize=(12,6))
    im = ax.imshow(-xPhys, cmap='gray',interpolation='none',vmin=-1,vmax=0)
    ax.axis("off")
    fig.show()

    # optimization loop
    it = 0
    it_beta = 0
    change = 1
    move_limit = 0.2
    while change>0.05 and it<args.max_iter:
        t_start = perf_counter()
        # solve finite element problem
        if args.solver=="chol":
            u = fea.solve_system(xPhys,sparse=True,unit_load=True)
        elif args.solver=="mgcg":
            Kstiff = fea.stiffness_matrix_assembly(xPhys)
            load = fea.load/np.sum(np.abs(fea.load))
            Ndiag = np.ones(mesh.ndof)
            Ndiag[fea.fixed_dofs] = 0
            Null = sp.diags(Ndiag)
            Ieye = sp.eye(mesh.ndof)
            Kstiff = Null.T.dot(Kstiff).dot(Null) - (Null-Ieye)
            u = mgcg.solve(Kstiff,load.squeeze(-1),rtol=1e-10,conv_criterion='disp',verbose=False)
        # calculate element-wise compliance
        ce = np.sum(np.matmul(u[mesh.edofMat],fea.KE)*u[mesh.edofMat],axis=1)
        ce = ce.reshape((mesh.nely,mesh.nelx),order='F')
        # calculate objective and sensitivities
        obj = np.sum(fea.Emin+xPhys**2*(fea.Emax-fea.Emin)*ce)
        dc = -fea.penal*(fea.Emax-fea.Emin)*xPhys**(fea.penal-1)*ce-1e-9
        dv = np.ones((mesh.nely,mesh.nelx))
        # filtering + modification of sensitivities
        if args.ft_type=='sensitivity':
            dc = convolve(dc*xPhys,filt_kernel,mode='constant',cval=0)/Nsum_filt/np.maximum(1e-3,xPhys)
        elif args.ft_type=='density':
            dc = convolve(dc/Nsum_filt,filt_kernel,mode='constant',cval=0)
            dv = convolve(dv/Nsum_filt,filt_kernel,mode='constant',cval=0)
        elif args.ft_type=='heaviside':
            dx = beta*np.exp(-beta*xTilde)+np.exp(-beta)
            dc = convolve((dc*dx)/Nsum_filt,filt_kernel,mode='constant',cval=0)
            dv = convolve((dv*dx)/Nsum_filt,filt_kernel,mode='constant',cval=0)
        # use bisectioning scheme to ensure volume constraint
        l1=1e-9; l2=1e9;
        while (l2-l1)/(l1+l2)>1e-3:
            lmid=0.5*(l2+l1)
            B_K = np.sqrt(-dc/dv/lmid)
            # generate new design using optimality criteria
            fixpoint = np.minimum(1.0,np.minimum(x+move_limit,x*B_K))
            xnew = np.maximum(0.0, np.maximum(x-move_limit, fixpoint))
            if args.ft_type=='sensitivity':
                xPhys = xnew.copy()
            elif args.ft_type=='density':
                xPhys = convolve(xnew,filt_kernel,mode='constant',cval=0)/Nsum_filt
            elif args.ft_type=='heaviside':
                xTilde = convolve(xnew,filt_kernel,mode='constant',cval=0)/Nsum_filt
                xPhys = 1-np.exp(-beta*xTilde)+xTilde*np.exp(-beta)
            if np.sum(xPhys)>(args.volfrac*mesh.nele):
                l1 = lmid
            else:
                l2 = lmid
        # calculate the change in design
        change = np.max(np.abs(xnew-x))
        x = xnew.copy()
        # update heaviside regularization parameters
        if args.ft_type=='heaviside' and beta<512 and (it_beta>=50 or change<=0.05):
            beta*=2
            it_beta = 0
            change = 1
        # print optimization process
        it+=1
        it_beta+=1

        t_end = perf_counter()
        if args.verbose==True:
            progress_str = "it.:{0}, obj.:{1:.3f}, Vol.:{2:.3f}, chng.:{3:.3f}, beta.:{4}, it. beta.:{5}, t[s].:{6:.3f}".format(\
                        it,obj,np.mean(xPhys),change,beta,it_beta,t_end-t_start)
            print(progress_str)
        im.set_array(-xPhys)
        fig.canvas.draw()
    fig.show()
    raw_input("Press any key...")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nelx',type=int,default=480)
    parser.add_argument('--nely',type=int,default=240)
    parser.add_argument('--volfrac',type=float,default=0.4)
    parser.add_argument('--penal',type=float,default=3.0)
    parser.add_argument('--rmin',type=float,default=1.2)
    parser.add_argument('--ft_type',type=str,default='heaviside') # 'sensitivity','density','heaviside'
    parser.add_argument('--nl',type=int,default=3)
    parser.add_argument('--solver',type=str,default='mgcg') # 'mgcg','chol'
    parser.add_argument('--max_iter',type=int,default=1000)
    parser.add_argument('--verbose',type=bool,default=True)
    args = parser.parse_args()
    run(args)
