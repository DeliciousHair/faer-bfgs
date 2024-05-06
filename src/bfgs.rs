use anyhow::{anyhow, Result};
use faer::{
    linalg::matmul::matmul,
    mat::{As2D, As2DMut},
    prelude::*,
    Entity, Parallelism, RealField,
};

pub enum FTol {
    Low,
    Moderate,
    High,
    Exact,
}

pub trait ExtraOps: Entity {
    type Real: RealField;

    const EPSILON: Self;
    const MACHINE_EPSILON: Self;
    const INFINITY: Self;

    fn faer_powi(self, pow: i32) -> Self;
    fn faer_max(self, other: Self) -> Self;
}

impl ExtraOps for f32 {
    type Real = Self;

    const EPSILON: Self = f32::EPSILON;
    const MACHINE_EPSILON: Self = 2e-24;
    const INFINITY: Self = f32::INFINITY;

    fn faer_powi(self, pow: i32) -> Self {
        self.powi(pow)
    }

    fn faer_max(self, other: Self) -> Self {
        f32::max(self, other)
    }
}

impl ExtraOps for f64 {
    type Real = Self;

    const EPSILON: Self = f64::EPSILON;
    const MACHINE_EPSILON: Self = 2e-53;
    const INFINITY: Self = f64::INFINITY;

    fn faer_powi(self, pow: i32) -> Self {
        self.powi(pow)
    }

    fn faer_max(self, other: Self) -> Self {
        f64::max(self, other)
    }
}

pub trait RealMathOps: RealField + ExtraOps {}
impl<E: RealField + ExtraOps> RealMathOps for E {}

/// From the L-BFGS paper (Zhu et al. 1994), 1e7 is for "moderate accuracy",
/// 1e12 for "low accuracy," 10 for "high accuracy". If `FACTOR_EXACT = 0`
/// is is used, the algorithm will only stop if the value of f stops
/// improving completely.
fn stopping_accuracy<E: RealMathOps>(val: &FTol) -> E {
    match val {
        FTol::Low => E::faer_from_f64(1e12).faer_mul(E::MACHINE_EPSILON),
        FTol::Moderate => E::faer_from_f64(1e7).faer_mul(E::MACHINE_EPSILON),
        FTol::High => E::faer_from_f64(10_f64).faer_mul(E::MACHINE_EPSILON),
        FTol::Exact => E::faer_zero(),
    }
}

/// Naive line search to find a decent value for `epsilon`
fn line_search<E: RealMathOps, F: Fn(E) -> E>(f: F) -> Result<E> {
    let mut best_epsilon = E::faer_zero();
    let mut best_fval = E::INFINITY;

    let mut eps: E;
    let mut fval: E;

    let two = E::faer_from_f64(2_f64);

    for i in -20..20 {
        eps = two.faer_powi(i);
        fval = f(eps);
        if fval.lt(&best_fval) {
            best_epsilon = eps;
            best_fval = fval;
        }
    }

    if best_epsilon.eq(&E::faer_zero()) {
        Err(anyhow!("line search failed"))
    } else {
        Ok(best_epsilon)
    }
}

/// If the improvement in `f` is not too much bigger than the rounding error,
/// then call it a success. This is the first stopping criterion from Zhu et al.
fn stop<E: RealMathOps>(old_val: E, val: E, tol: &FTol) -> bool {
    let delta = old_val.faer_sub(val);
    let denom = [old_val.faer_abs(), val.faer_abs(), E::faer_one()]
        .into_iter()
        .reduce(E::faer_max)
        .unwrap();
    let rhs = stopping_accuracy(tol);

    delta.faer_div(denom).le(&rhs)
}

/// # bgfs
/// Returns a value of `x` that should minimize `f`. `f` must be convex and
/// twice-differentiable.
///
/// ## parameters
/// - `x0` is an initial guess for `x`. Often this is chosen randomly.
/// - `f` is the objective function
/// - `g` is the gradient of `f`
pub fn bfgs<E: RealMathOps, F: Fn(ColRef<E>) -> E, G: Fn(ColRef<E>) -> Col<E>>(
    x0: Col<E>,
    f: F,
    g: G,
    stop_tol: FTol,
) -> Result<Col<E>> {
    let mut x = x0;
    let mut f_x = f(x.as_ref());
    let mut g_x = g(x.as_ref());
    let n = x.nrows();

    assert!(g_x.nrows().eq(&x.nrows()));

    // Initialize the inverse approximate Hessian to the identity matrix
    let mut b_inv: Mat<E> = Mat::identity(n, n);

    let mut y: Col<E>;
    let mut s: Col<E> = Col::zeros(n);
    let mut binv_y: Col<E> = Col::zeros(n);

    let mut sy: E;
    let mut ss: Mat<E>;

    loop {
        // Find the search direction
        let mut search_dir: Col<E> = Col::zeros(n);
        matmul(
            search_dir.as_2d_mut(),
            b_inv.as_ref(),
            g_x.as_2d_ref(),
            None,
            E::faer_one().faer_neg(),
            Parallelism::Rayon(0),
        );

        // Find a good step size
        let epsilon = line_search(|e| {
            let mut arg: Col<E> = Col::zeros(n);
            zipped!(arg.as_mut(), search_dir.as_ref(), x.as_ref()).for_each(
                |unzipped!(mut arg, search_dir, x)| {
                    let d = search_dir.read();
                    let x = x.read();
                    arg.write((d.faer_mul(e)).faer_add(x));
                },
            );
            f(arg.as_ref())
        })?;

        // Save the old values
        let f_x_old = f_x;
        let g_x_old = g_x;

        // Take a step in the search direction
        zipped!(x.as_mut(), search_dir.as_ref()).for_each(|unzipped!(mut x, search_dir)| {
            let d = search_dir.read();
            let elem = x.read();
            x.write(elem.faer_add(epsilon.faer_mul(d)))
        });

        f_x = f(x.as_ref());
        g_x = g(x.as_ref());

        // Compute deltas between old and new
        y = g_x.as_ref() - g_x_old.as_ref();
        zipped!(s.as_mut(), search_dir.as_ref()).for_each(|unzipped!(mut s, search_dir)| {
            let d = search_dir.read();
            s.write(epsilon.faer_mul(d))
        });
        sy = s.as_ref().transpose() * y.as_ref();
        ss = s.as_ref() * s.as_ref().transpose();

        if stop(f_x_old, f_x, &stop_tol) {
            return Ok(x);
        }

        // Update the Hessian approximation
        matmul(
            binv_y.as_2d_mut(),
            b_inv.as_ref(),
            y.as_2d_ref(),
            None,
            E::faer_one(),
            Parallelism::Rayon(0),
        );

        let update_add = sy
            .faer_add(y.transpose() * binv_y.as_ref())
            .faer_mul(sy.faer_powi(-2));
        let mut to_add = Mat::<E>::zeros(n, n);
        zipped!(to_add.as_mut(), ss.as_ref()).for_each(|unzipped!(mut to_add, ss)| {
            let e = ss.read();
            to_add.write(e.faer_mul(update_add));
        });

        let to_sub: Mat<E> = Mat::from_fn(n, n, |i, j| unsafe {
            let p = binv_y.read_unchecked(i).faer_mul(s.read_unchecked(j));
            let pt = s.read_unchecked(i).faer_mul(binv_y.read_unchecked(j));
            p.faer_add(pt).faer_div(sy)
        });

        b_inv = b_inv + to_add - to_sub;
    }
}
