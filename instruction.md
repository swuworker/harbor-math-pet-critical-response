# PET reconstruction in mirror coordinates

Implement the mathematical PET reconstruction and response contract below.

Implement `solve(request)` in `/app/src/kernel.py`. For the specified output fields, use only JSON-compatible finite real numbers, lists and dictionaries. The assessed writable source boundary will be `/app/src`. There are three actions: `forward`, `reconstruct`, and `directional`.

## Request fields and result types

All fields below are top-level fields of `request`, except the explicitly nested `direction` dictionary. Let n=nx*ny and m=len(rays). A scalar output is a finite real number; an array output is a JSON list with the stated dimensions. `solve` returns a dictionary. Arrays follow the flattening convention below. No solver diagnostics are required.

| Action | Required fields |
|---|---|
| All actions | `action` (one of the three strings), `nx`, `ny`, `origin` (2), `spacing` (2), `rays` (m-by-4), `psf` (n-by-n), `attenuation` (n), `efficiency` (m), `background` (m) |
| `forward` | Also `activity` (n), `cotangent` (m) |
| `reconstruct` | Also `data` (m), `initial` (n), `step_size` (scalar), `iterations` (integer), `tv_weight` (scalar), `quadratic_weight` (scalar), `tv_pixel_weights` (n) |
| `directional` | Same fields as `reconstruct`, plus `direction` with exactly the six parameter directions specified below |

No optional field supplies an unstated model parameter. Invalid requests and extra-field handling are outside the assessed scope.

## Imaging model

The image consists of `nx * ny` axis-aligned rectangular pixels. The bottom-left corner is `origin=[ox,oy]`, the pixel spacing is `spacing=[dx,dy]` in millimetres, and index `iy*nx+ix` increases first in the positive x direction, then in the positive y direction. Pixel values are constant inside each pixel. All activity and data values use the supplied numerical calibration; do not introduce an additional normalization.

Each row of `rays` is `[sx,sy,tx,ty]`, the endpoints of one line segment. Both endpoints are outside the image rectangle. The segment crosses the rectangle and is not collinear with any grid boundary. Let `R[r,i]` be the length, in millimetres, of the part of this segment inside pixel i. Isolated grid-corner intersections contribute zero length. No additional pixel-area factor is used.

`psf` is a supplied n-by-n nonnegative row-stochastic matrix H, acting on image activity before projection. It need not be symmetric. `attenuation` is a nonnegative n-vector mu in inverse millimetres; `efficiency` is a positive m-vector e. For ray r, the physical mean from activity x is the integral of Hx along that ray, multiplied by e[r] and by the attenuation factor associated with the integral of mu along the entire segment through the image. Add the supplied m-vector `background` b after these operations. There is no other scatter, blur, or detector response in this task.

Write this affine mean map as q(x)=A x+b. The adjoint is with respect to the ordinary Euclidean dot products of the flattened numerical vectors.

For `forward`, additionally supply `activity` x and an arbitrary real m-vector `cotangent` z. Return:

- `ray_matrix`: R, shape m-by-n;
- `system_matrix`: A, shape m-by-n;
- `mean`: q(x), length m;
- `adjoint`: the adjoint of the linear part applied to z, length n.

## Reconstruction map

For the other actions supply `data` y, `initial` x0, `step_size` tau, `iterations` K, `tv_weight` lambda, `quadratic_weight` beta and `tv_pixel_weights` w.

The data term is

`f(x) = sum_r [q(x)[r] - y[r]*log(q(x)[r])]`.

The product with a zero y entry is zero. Logarithms are natural. There is no factor 1/2, averaging, or factorial term in f. Real nonnegative data are allowed, including the continuous perturbations used below; the task does not ask for random Poisson samples.

For each pixel i, D x has two components: the forward difference to the next pixel in the positive x direction divided by dx, and the forward difference to the next pixel in the positive y direction divided by dy. A component with no neighbour is zero. Define

`g(x) = (beta/2)*sum_i x[i]^2 + lambda*sum_i w[i]*norm2((D x)[i])`.

Set `s=A^T ones(m)` and define the potential on positive vectors by

`h(x)=sum_i s[i]*(x[i]*log(x[i])-x[i])`.

Values in logarithms are numerical values in the fixed activity reference unit. Let h* be its convex conjugate and `D_psi(a,b)=psi(a)-psi(b)-dot(grad psi(b),a-b)`. No closed form for h* is supplied.

One reconstruction step is defined in mirror coordinates. At x_k put `eta_k=grad h(x_k)` and choose the global minimizer eta of

`dot(grad(f composed with grad h*)(eta_k), eta-eta_k)`
`+ D_(h*)(eta,eta_k)/tau + g(grad h*(eta))`.

Then `x_(k+1)=grad h*(eta)`. The selected input domain makes the step well-defined with a unique positive output. Return the result of exactly K such mathematical steps. The inner minimization is defined by its exact minimizer; it is not defined by a fixed number of iterations of any particular numerical method. The task accepts any method meeting the accuracy contract.

For `reconstruct`, return:

- `image`: x_K, length n;
- `history`: x_0 through x_K, shape (K+1)-by-n;
- `objective`: f(x_K)+g(x_K).

When K=0, the reconstruction result is the supplied initial image, `history` contains only that image, and `objective` is evaluated at that image with the supplied parameters.

## Directional response

For `directional`, additionally supply `direction`, a dictionary containing directions for `attenuation`, `background`, `data`, `initial`, `tv_weight`, and `quadratic_weight`, with the same respective shapes. Each is a required field; a zero direction is explicitly represented by zeros.

Let p denote these six varying inputs and F(p) the final image of the reconstruction map just defined. Return `image=F(p)` and

`response = limit_(t down to 0) [F(p+t*direction)-F(p)]/t`.

The geometry, H, efficiency, w, tau and K stay fixed. In each perturbed problem all quantities defined from p belong to that perturbed problem. The requested object is this one-sided directional derivative. Inputs may lie at a transition where the ordinary two-sided Jacobian does not exist. Tested cases will have the stated finite one-sided limit.

For K=0, `directional.image` is `initial` and `directional.response` is `direction.initial`.

## Valid input domain and accuracy

The following bounds and error rules define assessed valid inputs.

- nx,ny are positive integers at most 6; n<=36; 1<=m<=64; 0<=K<=6; 0.1<=tau<=0.5. Spacings lie in [0.5,2] mm.
- H has nonnegative entries, row sums 1 and diagonal entries at least 0.2. Efficiencies lie in [0.2,2], mu in [0,2], b in [0,3], y in [0,100], and initial activity in [0.1,5]. Forward activity uses the same positive range; cotangents may have either sign.
- lambda,beta lie in [0,2] and w in [0.5,2]. Every column sensitivity s[i]>=1e-3, and each ray has positive in-image length. Requested reconstruction iterates lie in [1e-4,100]. These are input promises, not requested verification outputs.
- Direction entries have magnitude at most 2. For all sufficiently small positive t, perturbed physical parameters remain in the nonnegative/positive mathematical domain and the relevant means and sensitivities stay positive. At a zero nonnegative parameter its direction cannot be negative. Nominal sizing/calibration boxes above do not constrain an infinitesimal path that exits a box while remaining in this mathematical domain.
- Elementwise error rule: `abs(actual-expected)<=atol+rtol*abs(expected)`. Forward arrays: atol=1e-9, rtol=1e-7. Reconstructed images, history and objective (including `directional.image`): atol=2e-7, rtol=2e-5. The `directional.response` array: atol=2e-5, rtol=3e-4.
- Invalid inputs, patient data, full scanner reconstruction, image quality metrics, and stochastic sampling are outside scope. Arrays, scalars, units and meanings above define the assessed outputs; internal solver states and a particular choice of numerical algorithm are not assessed.

## Execution

Python 3.11, NumPy, SciPy and Clarabel are installed. Implement any algorithm satisfying the contract. The JSON command is `python /app/run_case.py`, reading one request from stdin and writing one result to stdout. All candidate implementation files belong in `/app/src`. Use one CPU and 2 GiB RAM. A scoring item may make up to three requests and gives their candidate processes a cumulative 10-second execution budget, including interpreter startup. Runtime internet is available. No particular solver, differentiation technique or intermediate state is required.

Each request must produce at most 8 MiB (8,388,608 bytes) on stdout. JSON integer tokens may contain at most 4096 digits excluding a leading minus sign. Write exactly one JSON object to stdout; diagnostic text belongs on stderr. The specified output fields are required; additional output keys are ignored. JSON object member names must be unique. Object field order, insignificant whitespace and the presence or absence of a trailing newline are not assessed. All output must be valid finite JSON.
