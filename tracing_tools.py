import numpy as np
#import pytools as pt
import os,sys

import scipy 
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import matplotlib.colors


def _interp_scalar(interp, position):
    """Call a RegularGridInterpolator and guaranteed return a Python float.

    Handles position shapes like (3,), [x,y,z], or (1,3). Raises TypeError
    if the interpolator returns more than one value.
    """
    pos = np.asarray(position, dtype=float)
    if pos.ndim == 2 and pos.shape[0] == 1:
        pos = pos[0]
    # ensure 1D position vector
    pos = pos.ravel()
    val = interp(pos)
    arr = np.asarray(val)
    if arr.size != 1:
        raise TypeError(f"Interpolator returned array of shape {arr.shape}; expected scalar")
    return float(arr.item())


def extract_fields_to_binary(file):
    file = sys.argv[1]
    print("Coverting file ",file)
    f=pt.vlsvfile.VlasiatorReader(file)
    time=np.float32(f.read_parameter("time"))
    extents=np.float32(f.get_fsgrid_mesh_extent())
    size=np.float32(f.get_fsgrid_mesh_size())
    b=np.float32(f.read_fsgrid_variable("fg_b"))
    e=np.float32(f.read_fsgrid_variable("fg_e"))
    nx,ny,nz,nc=np.shape(e)
    outfile = open(file+".bin", 'wb')
    outfile.write(time)
    outfile.write(extents)
    outfile.write(size)
    outfile.write(b.flatten())
    outfile.write(e.flatten())
    outfile.close()
    
    
#### functions for particle tracing in the given fields

RE=6371000
e_charge=1.6*1e-19
mass_ion=1836*9.1*1e-31
qom=e_charge/mass_ion

def get_axes(extends,size):
    xx=np.linspace(extends[0],extends[3],int(size[0]))
    yy=np.linspace(extends[1],extends[4],int(size[1]))
    zz=np.linspace(extends[2],extends[5],int(size[2]))
    return xx,yy,zz

def get_extend_and_size(tstep):
    path='/Users/ivanzait/Downloads/analysator/'
    filename='bulk1.000'+ str(tstep)+'.vlsv.reduced.bin' ### !!!!!
    with open(path+filename,mode='rb') as f:
        data=np.fromfile(f, dtype=np.float32)
    extends=data[1:7]
    size=data[7:10]
    return extends,size


def get_B(position,tstep):
    e,b,extends,time,size = load_fields(tstep)
    xx,yy,zz=get_axes(extends,size)
    bx=b[:,:,:,0]
    by=b[:,:,:,1]
    bz=b[:,:,:,2]
    bx_int = RegularGridInterpolator((xx,yy,zz), bx, bounds_error=False, fill_value=np.nan)
    by_int = RegularGridInterpolator((xx,yy,zz), by, bounds_error=False, fill_value=np.nan)
    bz_int = RegularGridInterpolator((xx,yy,zz), bz, bounds_error=False, fill_value=np.nan)
    b_int=[_interp_scalar(bx_int, position), _interp_scalar(by_int, position), _interp_scalar(bz_int, position)]
    return b_int


def get_E(position,tstep):
    e,b,extends,time,size = load_fields(tstep)
    xx,yy,zz=get_axes(extends,size)
    ex=e[:,:,:,0]
    ey=e[:,:,:,1]
    ez=e[:,:,:,2]
    ex_int = RegularGridInterpolator((xx,yy,zz), ex, bounds_error=False, fill_value=np.nan)
    ey_int = RegularGridInterpolator((xx,yy,zz), ey, bounds_error=False, fill_value=np.nan)
    ez_int = RegularGridInterpolator((xx,yy,zz), ez, bounds_error=False, fill_value=np.nan)
    e_int=[_interp_scalar(ex_int, position), _interp_scalar(ey_int, position), _interp_scalar(ez_int, position)]
    return e_int

def int_E(e,position,extends,size):
    #e,b,extends,time,size = load_fields(tstep)
    xx,yy,zz=get_axes(extends,size)
    ex=e[:,:,:,0]
    ey=e[:,:,:,1]
    ez=e[:,:,:,2]
    ex_int = RegularGridInterpolator((xx,yy,zz), ex, bounds_error=False, fill_value=np.nan)
    ey_int = RegularGridInterpolator((xx,yy,zz), ey, bounds_error=False, fill_value=np.nan)
    ez_int = RegularGridInterpolator((xx,yy,zz), ez, bounds_error=False, fill_value=np.nan)
    e_int=(_interp_scalar(ex_int, position), _interp_scalar(ey_int, position), _interp_scalar(ez_int, position))
    return e_int


def load_fields(tstep, filepath='/Users/ivanzait/Desktop/tracer_vlsv/data_fields/'):

    file = filepath + 'bulk1.000'+ str(tstep)+'.vlsv.reduced.bin'
    with open(file,mode='rb') as f:
        data=np.fromfile(f, dtype=np.float32)

    time=data[0]
    extends=data[1:7]
    size=data[7:10]
    b_and_e=data[10::]

    b_1d=b_and_e[0:len(b_and_e)//2]
    e_1d=b_and_e[len(b_and_e)//2::]

    b = b_1d.reshape((int(size[0]),int(size[1]),int(size[2]),3))
    e = e_1d.reshape((int(size[0]),int(size[1]),int(size[2]),3))

    return e,b,extends,time,size


def int_field(field,position,extends):
    #e,b,extends,time,size = load_fields(tstep)
    size=[field.shape[0],field.shape[1],field.shape[2]]
    xx,yy,zz=get_axes(extends,size)
    field_x_int = RegularGridInterpolator((xx,yy,zz), field[:,:,:,0], bounds_error=False, fill_value=np.nan)
    field_y_int = RegularGridInterpolator((xx,yy,zz), field[:,:,:,1], bounds_error=False, fill_value=np.nan)
    field_z_int = RegularGridInterpolator((xx,yy,zz), field[:,:,:,2], bounds_error=False, fill_value=np.nan)
    fx = _interp_scalar(field_x_int, position)
    fy = _interp_scalar(field_y_int, position)
    fz = _interp_scalar(field_z_int, position)
    return fx,fy,fz



def lorentz_force_static(U, t, e, b, extends):

    x,y,z=U[0],U[1],U[2]
    vx,vy,vz=U[3],U[4],U[5]    
    position=[x,y,z]
    ex,ey,ez=e[0],e[1],e[2]
    
    bx,by,bz=b[0],b[1],b[2]

    u=U*0
    u[0]=vx ## Vx
    u[1]=vy ## Vy
    u[2]=vz ## Vz
    u[3]=qom*(ex+vy*bz-vz*by) ## Vx
    u[4]=qom*(ey-vx*bz+vz*bx) ## Vx
    u[5]=qom*(ez+vx*by-vy*bx) ## Vx

    return u



def lorentz_force(U, t, e, b, extends):

    x,y,z=U[0],U[1],U[2]
    vx,vy,vz=U[3],U[4],U[5]    
    position=[x,y,z]
    
    e = int_field(e,position,extends)
    b = int_field(b,position,extends)
    
    ex,ey,ez=e[0],e[1],e[2]
    bx,by,bz=b[0],b[1],b[2]    

    u=U*0
    u[0]=vx ## Vx
    u[1]=vy ## Vy
    u[2]=vz ## Vz
    u[3]=qom*(ex+vy*bz-vz*by) ## Vx
    u[4]=qom*(ey-vx*bz+vz*bx) ## Vx
    u[5]=qom*(ez+vx*by-vy*bx) ## Vx

    return u





def plot_trace(ax, store):
    x,y,z, = store[0],store[1],store[2]
    vx,vy,vz, = store[3],store[4],store[5]
    energy = np.sqrt(vx**2+vy**2+vz**2)
    # ax = plt.axes(projection='3d')
    ax.scatter(x/RE,y/RE,z/RE,c=cm.jet(np.abs(energy)/energy.max()),edgecolor='none')



def plot_trace_with_b_eq_box_manual(ax,store,b,extends,limits):
    
    
    
    size=[b.shape[0],b.shape[1],b.shape[2]]
    xx,yy,zz=get_axes(extends,size)
    

    yyy,xxx = np.meshgrid(yy,xx)
    xxx,yyy=xxx/RE,yyy/RE

    x_left,x_right,y_left,y_right=limits[0],limits[1],limits[2],limits[3]


    x_ind_left= int(abs((xx[0]-x_left)//1e+6) ) 
    x_ind_right= int(abs((xx[0]-x_right)//1e+6) )
    y_ind_left= int(abs((yy[0]-y_left)//1e+6) )
    y_ind_right= int(abs((yy[0]-y_right)//1e+6) )
    xxx=xxx[x_ind_left:x_ind_right,y_ind_left:y_ind_right]
    yyy=yyy[x_ind_left:x_ind_right,y_ind_left:y_ind_right]


    cmap = plt.cm.jet
    bx=b[x_ind_left:x_ind_right,y_ind_left:y_ind_right,int(b.shape[2]//2),0]
    norm = matplotlib.colors.Normalize(vmin=bx.min(), vmax=bx.max() )
    colors = cmap(norm(bx))
    ax.plot_surface(xxx,yyy,np.zeros_like(xxx), cstride=1, rstride=1, facecolors=colors, shade=False)


    ###### plot trajectory ######
    x,y,z, = store[0],store[1],store[2]
    vx,vy,vz, = store[3],store[4],store[5]
    energy = np.sqrt(vx**2+vy**2+vz**2)
    ax.scatter(x/RE,y/RE,z/RE,c=cm.jet(np.abs(energy)/energy.max()),edgecolor='none',s=3)
    #############################





    