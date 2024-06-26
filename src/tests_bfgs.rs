#[cfg(test)]
mod tests {
    use std::f64;

    use crate::bfgs::{bfgs, FTol, RealMathOps};
    use faer::{col, Col, ColRef};

    fn l2_distance<E: RealMathOps>(x: ColRef<E>, y: ColRef<E>) -> E {
        (x.as_ref() - y.as_ref()).transpose() * (x.as_ref() - y.as_ref())
    }

    #[test]
    fn test_x_squared_ndim() {
        let x0 = Col::<f64>::from_fn(25, |i| (i + 1) as f64 / 3_f64);
        let f = |x: ColRef<f64>| x.as_ref().transpose() * x.as_ref();
        let g = |x: ColRef<f64>| {
            Col::<f64>::from_fn(x.nrows(), |i| unsafe { x.read_unchecked(i) * 2_f64 })
        };
        let x_min = bfgs(x0, f, g, FTol::Moderate).unwrap();

        for i in 0..x_min.nrows() {
            assert!(x_min.read(i).eq(&0_f64));
        }
    }

    #[test]
    fn test_begin_at_minimum() {
        let x0 = Col::<f64>::zeros(25);
        let f = |x: ColRef<f64>| x.as_ref().transpose() * x.as_ref();
        let g = |x: ColRef<f64>| {
            Col::<f64>::from_fn(x.nrows(), |i| unsafe { x.read_unchecked(i) * 2_f64 })
        };
        let x_min = bfgs(x0, f, g, FTol::Moderate).unwrap();

        for i in 0..x_min.nrows() {
            assert!(x_min.read(i).eq(&0_f64));
        }
    }

    // Negative test, function has a max but no (global) min. Note that starting
    // the algorithm at the maximum of `0` will not error which is not correct
    // but is aligned with the current behaviour of `scipy.optimize.minimize`
    // and the reference code this has been ported from, so just leaving this
    // alone for now.
    #[test]
    fn test_negative_x_squared() {
        let x0 = Col::<f64>::from_fn(25, |i| (i + 1) as f64 / 3_f64);
        let f = |x: ColRef<f64>| -x.as_ref().transpose() * x.as_ref();
        let g = |x: ColRef<f64>| {
            Col::<f64>::from_fn(x.nrows(), |i| unsafe { x.read_unchecked(i) * -2_f64 })
        };

        bfgs(x0, f, g, FTol::Moderate).expect_err("line search failed");
    }

    #[test]
    fn test_rosenbrock() {
        let x0 = Col::<f64>::zeros(2);
        let f = |e: ColRef<f64>| {
            let x = e.read(0);
            let y = e.read(1);

            (1_f64 - x).powi(2) + 100_f64 * (y - x.powi(2)).powi(2)
        };
        let g = |e: ColRef<f64>| {
            let x = e.read(0);
            let y = e.read(1);

            col![
                -400_f64 * (y - x.powi(2)) * x - 2_f64 * (1_f64 - x),
                200_f64 * (y - x.powi(2)),
            ]
        };
        let x_min = bfgs(x0, f, g, FTol::Moderate).unwrap();

        assert!(
            l2_distance(x_min.as_ref(), Col::<f64>::from_fn(2, |_| 1_f64).as_ref())
                .lt(&f64::EPSILON)
        );
    }

    #[test]
    /// From the docstring of `scipy.optimize.fsolve', we find a solution to the
    /// system of equations:
    /// `x0*cos(x1) = 4,  x1*x0 - x1 = 5`
    ///
    /// `x = [6.50409711, 0.90841421]`
    fn test_solve() {
        let x0 = Col::<f64>::from_fn(2, |_| 1_f64);
        let f = |x: ColRef<f64>| {
            let tmp = col![
                x.read(0) * x.read(1).cos() - 4_f64,
                x.read(0) * x.read(1) - x.read(1) - 5_f64,
            ];
            tmp.as_ref().transpose() * tmp.as_ref()
        };
        let g = |x: ColRef<f64>| {
            let jac = col![
                2_f64 * (x.read(0) * x.read(1).cos() - 4_f64) * (x.read(1).cos())
                    + 2_f64 * (x.read(0) * x.read(1) - x.read(1) - 5_f64) * x.read(1),
                2_f64 * (x.read(0) * x.read(1).cos() - 4_f64) * (-x.read(0) * x.read(1).sin())
                    + 2_f64 * (x.read(0) * x.read(1) - x.read(1) - 5_f64) * (x.read(0) - 1_f64),
            ];

            jac
        };
        let res = bfgs(x0, f, g, FTol::High).unwrap();
        let target = col![6.504_097_11_f64, 0.908_414_21_f64,];

        assert!(l2_distance(res.as_ref(), target.as_ref()).lt(&f64::EPSILON));
    }
}
