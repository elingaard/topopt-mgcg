# topopt-mgcg
Topology optimization code utilizing a Multi-Grid Conjugate Gradient method to solve large-scale problems. Optimization loop is inspired by the top99 code from https://www.topopt.mek.dtu.dk/Apps-and-software/A-99-line-topology-optimization-code-written-in-MATLAB, while separate classes have been created to handle the finite element meshing and analysis.

To run the code type `python topopt_mgcg` in the terminal. By default a 480x240 discretization of the Michell cantilever problem is solved. The following additional arguments can be parsed through the terminal

```python
parser.add_argument('--nelx',type=int,default=480)
parser.add_argument('--nely',type=int,default=240)
parser.add_argument('--volfrac',type=float,default=0.4) # volume fraction
parser.add_argument('--penal',type=float,default=3.0) # SIMP penalization
parser.add_argument('--rmin',type=float,default=1.2) # filter radius indicating the minimum feature size
parser.add_argument('--ft_type',type=str,default='heaviside') # filter type: 'sensitivity','density','heaviside'
parser.add_argument('--nl',type=int,default=3) # number of levels in the MGCG solver
parser.add_argument('--solver',type=str,default='mgcg') # 'mgcg','chol'
parser.add_argument('--max_iter',type=int,default=1000)
parser.add_argument('--verbose',type=bool,default=True)
```
