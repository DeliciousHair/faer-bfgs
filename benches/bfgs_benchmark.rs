use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::prelude::*;
use faer_bfgs::bfgs::{bfgs, FTol};
use rand::distributions::{Distribution, Uniform};

fn generate_x(n: usize) -> Col<f64> {
    let mut rng = rand::thread_rng();
    let unf = Uniform::from(-10_000..10_000);

    Col::<f64>::from_fn(n, |_| unf.sample(&mut rng) as f64 / 100_f64)
}

fn easy_fouth_order() -> (
    impl Fn(ColRef<f64>) -> f64,
    impl Fn(ColRef<f64>) -> Col<f64>,
) {
    let f = |x: ColRef<f64>| {
        let tmp = Col::<f64>::from_fn(x.nrows(), |i| x.read(i).powi(2));
        tmp.as_ref().transpose() * tmp.as_ref()
    };
    let g = |x: ColRef<f64>| Col::<f64>::from_fn(x.nrows(), |i| 4_f64 * x.read(i).powi(3));

    (f, g)
}

pub fn solve_convex(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve easy convex");
    for n in (2..=100).step_by(10) {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let x = generate_x(n as usize);
                    let (f, g) = easy_fouth_order();
                    (x, f, g)
                },
                |(x0, f, g)| bfgs(x0, f, g, FTol::Moderate).unwrap(),
                criterion::BatchSize::SmallInput,
            );
        });
    }
}

criterion_group!(benches, solve_convex);
criterion_main!(benches);
