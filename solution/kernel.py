"""Independent implementation: exponential/SOC conic proximal solves."""
import numpy as np
import scipy.sparse as sp
import clarabel
from scipy.linalg import null_space
from scipy.optimize import minimize


def geometry(p):
    nx, ny = p['nx'], p['ny']
    ox, oy = p['origin']; dx, dy = p['spacing']
    R = np.zeros((len(p['rays']), nx*ny))
    for r, ray in enumerate(p['rays']):
        a = np.array(ray[:2], float); d = np.array(ray[2:], float)-a
        cuts = [0., 1.]
        for axis, base, step, count in [(0,ox,dx,nx),(1,oy,dy,ny)]:
            if d[axis] != 0:
                cuts.extend((base+j*step-a[axis])/d[axis] for j in range(count+1))
        cuts = sorted(set(max(0.,min(1.,float(t))) for t in cuts))
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            mid = a+(lo+hi)*.5*d
            ix=int(np.floor((mid[0]-ox)/dx)); iy=int(np.floor((mid[1]-oy)/dy))
            if 0 <= ix < nx and 0 <= iy < ny:
                R[r,iy*nx+ix] += (hi-lo)*np.linalg.norm(d)
    A = (np.array(p['efficiency'])*np.exp(-R@np.array(p['attenuation'])))[:,None]*(R@np.array(p['psf']))
    return R,A


def differences(p):
    nx,ny=p['nx'],p['ny']; n=nx*ny
    D=np.zeros((n,2,n))
    for y in range(ny):
        for x in range(nx):
            i=y*nx+x
            if x+1<nx: D[i,0,i]=-1/p['spacing'][0]; D[i,0,i+1]=1/p['spacing'][0]
            if y+1<ny: D[i,1,i]=-1/p['spacing'][1]; D[i,1,i+nx]=1/p['spacing'][1]
    return D


def prox(s,c,rho,alpha,w,D):
    n=len(s)
    if alpha == 0:
        return 2*c/(s+np.sqrt(s*s+4*rho*c))
    # Variables (u, log(u) hypograph, TV epigraph). All inequalities
    # represented as A*x + slack=b in exponential and Lorentz cones.
    rows=[]; rhs=[]; cones=[]
    for i in range(n):
        block=np.zeros((3,3*n)); block[0,n+i]=-1; block[2,i]=-1
        rows.append(block); rhs.extend([0.,1.,0.]); cones.append(clarabel.ExponentialConeT())
    for i in range(n):
        block=np.zeros((3,3*n)); block[0,2*n+i]=-1; block[1:,:n]=-D[i]
        rows.append(block); rhs.extend([0.,0.,0.]); cones.append(clarabel.SecondOrderConeT(3))
    settings=clarabel.DefaultSettings(); settings.verbose=False; settings.max_iter=250
    settings.tol_gap_abs=1e-12; settings.tol_gap_rel=1e-12; settings.tol_feas=1e-12
    settings.iterative_refinement_abstol=1e-13; settings.iterative_refinement_reltol=1e-13
    solver=clarabel.DefaultSolver(sp.diags(np.r_[np.full(n,rho),np.zeros(2*n)],format='csc'),
        np.r_[s,-c,alpha*w],sp.csc_matrix(np.vstack(rows)),np.array(rhs),cones,settings)
    result=solver.solve()
    if str(result.status) not in ('Solved','AlmostSolved'):
        raise ArithmeticError(str(result.status))
    u=np.array(result.x[:n])
    dual=np.array(result.z[3*n:]).reshape(n,3)[:,1:]
    dual=-dual/(alpha*w[:,None])
    for threshold in (2e-5,2e-7,2e-9,0.):
        refined=_refine(u,s,c,rho,alpha,w,D,threshold)
        if _certificate(refined,s,c,rho,alpha,w,D,dual):
            return refined
    # The interior-point solution remains a valid approximate minimizer
    # when a reliable nonsmooth face cannot be identified.
    return u


def _certificate(u,s,c,rho,alpha,w,D,dual):
    n=len(u); norms=np.linalg.norm(D@u,axis=1)
    active=norms<1e-11
    g=s-c/u+rho*u
    for i in np.flatnonzero(~active):
        g+=alpha*w[i]*D[i].T@(D[i]@u)/norms[i]
    ids=np.flatnonzero(active)
    if len(ids):
        B=np.concatenate([alpha*w[i]*D[i].T for i in ids],axis=1)
        def fun(p):
            r=g+B@p
            return float(r@r)/2,B.T@r
        def balls(p): return 1-np.sum(p.reshape(-1,2)**2,axis=1)
        def jac(p):
            out=np.zeros((len(ids),2*len(ids)))
            for k in range(len(ids)): out[k,2*k:2*k+2]=-2*p[2*k:2*k+2]
            return out
        res=minimize(fun,dual[ids].ravel(),jac=True,method='SLSQP',
            constraints={'type':'ineq','fun':balls,'jac':jac},
            options={'ftol':1e-25,'maxiter':200})
        if np.min(balls(res.x)) < -1e-8: return False
        g+=B@res.x
    return np.max(abs(g)) <= 2e-10*max(1.,np.max(s),np.max(c/u))


def _refine(u,s,c,rho,alpha,w,D,threshold):
    n=len(u)
    # Refine on the conic solver's identified face. This removes the
    # square-root primal uncertainty of a tiny interior-point duality gap.
    norms=np.linalg.norm(D@u,axis=1)
    active=norms <= threshold
    constraints=D[active].reshape(-1,n)
    Z=null_space(constraints) if constraints.size else np.eye(n)
    a=Z.T@u
    def smooth(a):
        v=Z@a
        if np.min(v)<=0: return np.inf,None,None
        g=s-c/v+rho*v; H=np.diag(c/v**2+rho)
        value=s@v-c@np.log(v)+rho*(v@v)/2
        for i in np.flatnonzero(~active):
            d=D[i]@v; norm=np.linalg.norm(d)
            if norm==0: return np.inf,None,None
            unit=d/norm; scale=alpha*w[i]
            value+=scale*norm; g+=scale*D[i].T@unit
            H+=scale/norm*D[i].T@(np.eye(2)-np.outer(unit,unit))@D[i]
        return value,Z.T@g,Z.T@H@Z
    for _ in range(30):
        value,g,H=smooth(a)
        if np.max(abs(g))<2e-14: break
        delta=np.linalg.solve(H,-g)
        step=1.
        while step>1e-8:
            val2,_,_=smooth(a+step*delta)
            if val2 <= value+1e-4*step*(g@delta)+1e-13: break
            step*=.5
        a+=step*delta
    return Z@a


def reconstruction(p):
    _,A=geometry(p); s=A.sum(axis=0); b=np.array(p['background']); y=np.array(p['data'])
    x=np.array(p['initial'],float); D=differences(p); w=np.array(p['tv_pixel_weights'])
    tau=p['step_size']; hist=[x.copy()]
    for _ in range(p['iterations']):
        v=(1-tau)*x+tau*x*(A.T@(y/(A@x+b)))/s
        x=prox(s,s*v,tau*p['quadratic_weight'],tau*p['tv_weight'],w,D)
        hist.append(x.copy())
    q=A@x+b
    obj=np.sum(q-y*np.log(q))+.5*p['quadratic_weight']*(x@x)+p['tv_weight']*(w@np.linalg.norm(D@x,axis=1))
    return x,hist,float(obj)


def solve(p):
    if p['action']=='forward':
        R,A=geometry(p)
        return dict(ray_matrix=R.tolist(),system_matrix=A.tolist(),mean=(A@np.array(p['activity'])+p['background']).tolist(),adjoint=(A.T@np.array(p['cotangent'])).tolist())
    x,hist,obj=reconstruction(p)
    if p['action']=='reconstruct':
        return dict(image=x.tolist(),history=np.array(hist).tolist(),objective=obj)
    if p['iterations']==0:
        return dict(image=x.tolist(),response=p['direction']['initial'])
    # Positive-step extrapolation respects one-sided branch semantics.
    values=[]
    hmax=.004
    for key,direction in p['direction'].items():
        v=np.atleast_1d(p[key]); d=np.atleast_1d(direction)
        if np.any(d<0): hmax=min(hmax,float(np.min(-.25*v[d<0]/d[d<0])))
    # Nearby distinct TV pixels can merge at a finite perturbation even
    # though the derivative at the nominal point is smooth. Resolve the
    # smallest nonzero nominal spatial scale before sampling the path.
    direction_scale=max(1.,max(float(np.max(abs(np.asarray(d)))) for d in p['direction'].values()))
    D=differences(p)
    for xx in hist[1:]:
        norms=np.linalg.norm(D@xx,axis=1)
        nz=norms[norms>1e-10]
        if len(nz): hmax=min(hmax,.01*float(np.min(nz))/direction_scale)
    previous=None; best=None; best_change=np.inf
    for level in range(8):
        h=hmax/(2**level)
        shifted=dict(p)
        for key,direction in p['direction'].items():
            value=np.array(p[key])+h*np.array(direction)
            shifted[key]=value.tolist()
        xx,_,_=reconstruction(shifted)
        values.append((xx-x)/h)
        if len(values)<3: continue
        d1=2*values[-2]-values[-3]; d2=2*values[-1]-values[-2]
        response=(4*d2-d1)/3
        if previous is not None:
            change=np.max(abs(response-previous)/(1+abs(response)))
            if change<best_change: best=response; best_change=change
            if change<2e-7: break
        previous=response
    if best is not None: response=best
    return dict(image=x.tolist(),response=response.tolist())
