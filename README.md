# faer-bfgs

This package contains an implementation of [BFGS][wikipedia], an algorithm for
minimizing convex $C^2$ smooth functions.

This is a direct port of the existing [bfgs crate][src] using `faer` for all
linear algebra ops as well as handling of generic types. Both `f32` and `f64`
data is supported out of the box.

# WARNING
This was done as quick exercise mostly for my own edification and has copied
the same horrible line-search from the source crate. The implementation fails
quite readily as a result. Do not rely on this for anything at this point in
time.

## example
Minimize a globally convex function:

```rust
use faer::{col, Col, ColRef};
use faer_bfgs::bfgs::{bfgs, FTol};

fn main() {
    // choose an arbitrary starting value
    let x0 = col![8.888_f64, 1.234_f64, -37.42_f64];
    let target = Col::<f64>::zeros(3);

    // easy example:
    //   f(x) = <x,x>
    //   g(x) = 2x
    let f = |x: ColRef<f64>| x.as_ref().transpose() * x.as_ref();
    let g = |x: ColRef<f64>| Col::<f64>::from_fn(x.nrows(), |i| x.read(i) * 2_f64);
    let x_min = bfgs(x0, f, g, FTol::Moderate).unwrap();

    for i in 0..3 {
        assert!(x_min.read(i).eq(&target.read(i)));
    }
}
```

License: MIT/Apache-2.0

[wikipedia]: <https://en.wikipedia.org/w/index.php?title=BFGS_method>
[src]: <https://github.com/paulkernfeld/bfgs>
