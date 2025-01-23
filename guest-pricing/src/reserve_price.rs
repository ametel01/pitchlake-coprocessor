use eth_rlp_types::BlockHeader;
use eyre::{anyhow as err, Result};
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use statrs::distribution::Binomial;
use std::f64::consts::PI;

use optimization::{NumericalDifferentiation, Func, GradientDescent, Minimizer};


use super::utils::hex_string_to_f64;

pub fn calculate_reserve_price(block_headers: Vec<BlockHeader>) -> Result<f64> {
    eprintln!("Starting calculate_reserve_price with {} block headers", block_headers.len());
    if block_headers.is_empty() {
        return Err(eyre::eyre!("No block headers provided."));
    }
    eprintln!("Calculating reserve price...");
    let mut data = Vec::new();
    for header in block_headers {
        let timestamp = i64::from_str_radix(
            header.timestamp.ok_or_else(|| err!("No timestamp in header"))?.strip_prefix("0x").unwrap(),
            16,
        )?;
        let base_fee = hex_string_to_f64(
            &header.base_fee_per_gas.ok_or_else(|| err!("No base fee in header"))?,
        )?;
        eprintln!("Processed block - timestamp: {}, base_fee: {}", timestamp, base_fee);
        data.push((timestamp * 1000, base_fee));
    }

    data.sort_by(|a, b| a.0.cmp(&b.0));
    eprintln!("Sorted data range: {} to {}", data[0].0, data.last().unwrap().0);
    
    let twap_7d = add_twap_7d(&data)?;
    eprintln!("Calculated TWAP 7d, last value: {}", twap_7d.last().unwrap());
    let strike = twap_7d.last().ok_or_else(|| err!("The series is empty"))?;
    eprintln!("Strike price: {}", strike);
    
    let num_paths = 4;
    let n_periods = 7;
    
    let fees: Vec<&f64> = data.iter().map(|x| &x.1).collect();
    let log_base_fee = compute_log_of_base_fees(&fees)?;
    eprintln!("Computed log base fees, length: {}", log_base_fee.len());
    
    let (slope, intercept, trend_values) = discover_trend(&log_base_fee)?;
    eprintln!("Discovered trend - slope: {}, intercept: {}", slope, intercept);
    
    let detrended_log_base_fee: DVector<f64> = DVector::from_iterator(
        log_base_fee.len(),
        log_base_fee.iter().zip(&trend_values).map(|(log_base_fee, trend)| log_base_fee - trend),
    );

    let (de_seasonalised_detrended_log_base_fee, season_param) = remove_seasonality(
        &detrended_log_base_fee,
        &data,
    )?;
    eprintln!("Seasonality parameters: {:?}", season_param);
    
    let (de_seasonalized_detrended_simulated_prices, params) = simulate_prices(
        &de_seasonalised_detrended_log_base_fee,
        n_periods,
        num_paths,
    )?;
    eprintln!("Simulated prices with parameters: {:?}", params);
    
    let period_start_timestamp = data[0].0;
    let period_end_timestamp = data.last().ok_or_else(|| err!("Missing end timestamp"))?.0;
    let total_hours = (period_end_timestamp - period_start_timestamp) / 3600 / 1000;
    
    let sim_hourly_times = DVector::from_iterator(
        n_periods,
        (0..n_periods).map(|i| total_hours as f64 + i as f64),
    );
    
    let c = season_matrix(sim_hourly_times);
    let season = &c * &season_param;
    let season_matrix = season.reshape_generic(nalgebra::Dyn(n_periods), nalgebra::Const::<1>);
    let season_matrix_shaped = DMatrix::from_fn(n_periods, num_paths, |row, _| season_matrix[(row, 0)]);

    let detrended_simulated_prices = &de_seasonalized_detrended_simulated_prices + &season_matrix_shaped;

    let log_twap_7d: Vec<f64> = twap_7d.iter().map(|x| x.ln()).collect();
    let returns: Vec<f64> = log_twap_7d.windows(2).map(|window| window[1] - window[0]).collect();

    let mu = 0.05 / 52.0;
    let sigma = standard_deviation(&returns) * f64::sqrt(24.0 * 7.0);
    let dt = 1.0 / 24.0;
    
    let mut stochastic_trend = DMatrix::zeros(n_periods, num_paths);
    let normal = Normal::new(0.0, sigma * f64::sqrt(dt))?;
        let mut rng = thread_rng();
    
    for i in 0..num_paths {
        let random_shocks: Vec<f64> = (0..n_periods).map(|_| normal.sample(&mut rng)).collect();
        let mut cumsum = 0.0;
        for j in 0..n_periods {
            cumsum += (mu - 0.5 * sigma.powi(2)) * dt + random_shocks[j];
            stochastic_trend[(j, i)] = cumsum;
        }
    }

    let final_trend_value = slope * (log_base_fee.len() - 1) as f64 + intercept;
        let mut simulated_log_prices = DMatrix::zeros(n_periods, num_paths);

    for i in 0..n_periods {
        let trend = final_trend_value;
        for j in 0..num_paths {
            simulated_log_prices[(i, j)] = detrended_simulated_prices[(i, j)] + trend + stochastic_trend[(i, j)];
        }
    }

    let simulated_prices = simulated_log_prices.map(f64::exp);
    let twap_start = n_periods.saturating_sub(24 * 7);
        
    let final_prices_twap = simulated_prices
        .rows(twap_start, n_periods - twap_start)
        .column_mean();

    let capped_price = (1.0 + 0.3) * strike;
    let payoffs = final_prices_twap.map(|price| (price.min(capped_price) - strike).max(0.0));
    let average_payoff = payoffs.mean();
    
    let reserve_price = f64::exp(-0.05) * average_payoff;
    eprintln!("Final reserve price calculated: {}", reserve_price);
    
    Ok(reserve_price)
}

fn simulate_prices(
    de_seasonalised_detrended_log_base_fee: &DVector<f64>,
    n_periods: usize,
    num_paths: usize,
) -> Result<(DMatrix<f64>, Vec<f64>)> {
    eprintln!("Starting price simulation with {} periods and {} paths", n_periods, num_paths);
    
    let dt = 1.0 / (365.0 * 24.0);

    // Prepare time series data
    let pt = DVector::from_row_slice(&de_seasonalised_detrended_log_base_fee.as_slice()[1..]);
    let pt_1 = DVector::from_row_slice(&de_seasonalised_detrended_log_base_fee.as_slice()[..de_seasonalised_detrended_log_base_fee.len()-1]);

    // Define and initialize the function for numerical differentiation
    let function = NumericalDifferentiation::new(Func(|x: &[f64]| neg_log_likelihood(x, &pt, &pt_1)));

    // Minimizer for the optimization problem
    let minimizer = GradientDescent::new().max_iterations(Some(2400));

    // Perform the minimization
    let var_pt = pt.iter().map(|&x| x * x).sum::<f64>() / pt.len() as f64;
        
    let solution = minimizer.minimize(
        &function,
        vec![-3.928e-02, 2.873e-04, 4.617e-02, var_pt, var_pt, 0.2],
    );
    eprintln!("Optimization completed - solution: {:?}", solution.position);

    // Extract the optimized parameters
    let params = &solution.position;
    
    let alpha = params[0] / dt;
    let kappa = (1.0 - params[1]) / dt;
    let mu_j = params[2];
    let sigma = (params[3] / dt).sqrt();
    let sigma_j = params[4].sqrt();
    let lambda_ = params[5] / dt;
    eprintln!("Model parameters - alpha: {}, kappa: {}, mu_j: {}, sigma: {}, sigma_j: {}, lambda: {}", 
              alpha, kappa, mu_j, sigma, sigma_j, lambda_);

            
    // RNG for stochastic processes
    let mut rng = thread_rng();

    // Simulate the Poisson process (jumps)
    let binom = Binomial::new(lambda_ * dt, 1)?;
    let mut jumps = DMatrix::zeros(n_periods, num_paths);
    for i in 0..n_periods {
        for j in 0..num_paths {
            jumps[(i, j)] = binom.sample(&mut rng) as f64;
        }
    }

    // Initialize simulated prices
    let mut simulated_prices = DMatrix::zeros(n_periods, num_paths);
    let initial_price = de_seasonalised_detrended_log_base_fee[de_seasonalised_detrended_log_base_fee.len() - 1];
        for j in 0..num_paths {
        simulated_prices[(0, j)] = initial_price;
    }
    
        // Generate standard normal variables
    let normal = Normal::new(0.0, 1.0).unwrap();
        let mut n1 = DMatrix::zeros(n_periods, num_paths);
        let mut n2 = DMatrix::zeros(n_periods, num_paths);
        for i in 0..n_periods {
        for j in 0..num_paths {
            n1[(i, j)] = normal.sample(&mut rng);
            n2[(i, j)] = normal.sample(&mut rng);
        }
    }
        // Simulate prices over time
    for i in 1..n_periods {
        for j in 0..num_paths {
            let prev_price = simulated_prices[(i-1, j)];
            let current_n1 = n1[(i, j)];
            let current_n2 = n2[(i, j)];
            let current_j = jumps[(i, j)];

            simulated_prices[(i, j)] = alpha * dt 
                + (1.0 - kappa * dt) * prev_price 
                + sigma * dt.sqrt() * current_n1 
                + current_j * (mu_j + sigma_j * current_n2);
        }
    }

        Ok((simulated_prices, params.to_vec()))
}

fn fit_linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
    if x.len() != y.len() {
        return Err(eyre::eyre!("Input arrays x and y must have the same length.").into());
    }

    let n = x.len();
    let x_vec = DVector::from_row_slice(x);
    let y_vec = DVector::from_row_slice(y);
    
    let mut design_matrix = DMatrix::zeros(n, 2);
    design_matrix.set_column(0, &DVector::from_element(n, 1.0));
    design_matrix.set_column(1, &x_vec);

    let solution = (&design_matrix.transpose() * &design_matrix)
        .try_inverse()
        .ok_or_else(|| eyre::eyre!("Singular matrix"))?
        * &design_matrix.transpose()
        * y_vec;

    Ok((solution[1], solution[0]))
}

fn predict(x: &[f64], slope: f64, intercept: f64) -> DVector<f64> {
    DVector::from_iterator(
        x.len(),
        x.iter().map(|&xi| slope * xi + intercept)
    )
}

fn discover_trend(log_base_fee: &[f64]) -> Result<(f64, f64, Vec<f64>)> {
    eprintln!("Discovering trend from {} log base fee values", log_base_fee.len());
    let time_index: Vec<f64> = (0..log_base_fee.len()).map(|i| i as f64).collect();
    let (slope, intercept) = fit_linear_regression(&time_index, log_base_fee)?;
    eprintln!("Linear regression results - slope: {}, intercept: {}", slope, intercept);
    let trend_values = predict(&time_index, slope, intercept);
    
    Ok((slope, intercept, trend_values.as_slice().to_vec()))
}

fn compute_log_of_base_fees(base_fees: &Vec<&f64>) -> Result<Vec<f64>> {
    Ok(base_fees.iter().map(|&x| x.ln()).collect())
}

fn remove_seasonality(
    detrended_log_base_fee: &DVector<f64>,
    data: &Vec<(i64, f64)>,
) -> Result<(DVector<f64>, DVector<f64>)> {
    eprintln!("Removing seasonality from {} data points", data.len());
    let start_timestamp = data.first().ok_or_else(|| err!("Missing start timestamp"))?.0;
    
    let t_series = DVector::from_iterator(
        data.len(),
        data.iter().map(|(timestamp, _)| (*timestamp - start_timestamp) as f64 / 1000.0 / 3600.0)
    );
    
    let c = season_matrix(t_series.clone());
    
    let epsilon = 1e-300;
    let season_param = lstsq::lstsq(&c, &detrended_log_base_fee, epsilon)
        .unwrap()
        .solution;
    
    let season = &c * &season_param;
    
    let de_seasonalised_detrended_log_base_fee = detrended_log_base_fee - season;
    
    eprintln!("Seasonality parameters: {:?}", season_param);
    Ok((de_seasonalised_detrended_log_base_fee, season_param))
}

fn season_matrix(t: DVector<f64>) -> DMatrix<f64> {
    let n = t.len();
    let mut result = DMatrix::zeros(n, 12);
    
    for i in 0..n {
        let time = t[i];
        result[(i, 0)] = (2.0 * PI * time / 24.0).sin();
        result[(i, 1)] = (2.0 * PI * time / 24.0).cos();
        result[(i, 2)] = (4.0 * PI * time / 24.0).sin();
        result[(i, 3)] = (4.0 * PI * time / 24.0).cos();
        result[(i, 4)] = (8.0 * PI * time / 24.0).sin();
        result[(i, 5)] = (8.0 * PI * time / 24.0).cos();
        result[(i, 6)] = (2.0 * PI * time / (24.0 * 7.0)).sin();
        result[(i, 7)] = (2.0 * PI * time / (24.0 * 7.0)).cos();
        result[(i, 8)] = (4.0 * PI * time / (24.0 * 7.0)).sin();
        result[(i, 9)] = (4.0 * PI * time / (24.0 * 7.0)).cos();
        result[(i, 10)] = (8.0 * PI * time / (24.0 * 7.0)).sin();
        result[(i, 11)] = (8.0 * PI * time / (24.0 * 7.0)).cos();
    }
    
    result
}

fn standard_deviation(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

fn mrjpdf(params: &[f64], pt: &DVector<f64>, pt_1: &DVector<f64>) -> DVector<f64> {
    let (a, phi, mu_j, sigma_sq, sigma_sq_j, lambda) = (
        params[0], params[1], params[2], params[3], params[4], params[5],
    );

    let diff1 = pt - (DVector::from_element(pt.len(), a) + phi * pt_1 + DVector::from_element(pt.len(), mu_j));
    let diff2 = pt - (DVector::from_element(pt.len(), a) + phi * pt_1);

    let term1 = lambda * (-diff1.map(|x| x.powi(2)) / (2.0 * (sigma_sq + sigma_sq_j)))
        .map(f64::exp)
        / ((2.0 * std::f64::consts::PI * (sigma_sq + sigma_sq_j)).sqrt());

    let term2 = (1.0 - lambda) * (-diff2.map(|x| x.powi(2)) / (2.0 * sigma_sq))
        .map(f64::exp)
        / ((2.0 * std::f64::consts::PI * sigma_sq).sqrt());

    term1 + term2
}

fn neg_log_likelihood(params: &[f64], pt: &DVector<f64>, pt_1: &DVector<f64>) -> f64 {
    let pdf_vals = mrjpdf(params, pt, pt_1);
    -(pdf_vals.map(|x| x + 1e-10).map(f64::ln).sum())
}

fn add_twap_7d(data: &Vec<(i64, f64)>) -> Result<Vec<f64>> {
    eprintln!("Calculating TWAP 7d for {} data points", data.len());
    let required_window_size = 24 * 7;
    let n = data.len();
    
    if n < required_window_size {
        return Err(err!(
            "Insufficient data: At least {} data points are required, but only {} provided.",
            required_window_size,
            n
        ));
    }

    let values = DVector::from_iterator(n, data.iter().map(|&(_, value)| value));
    let mut twap_values = Vec::with_capacity(n);

    for i in 0..n {
        let window_start = if i >= required_window_size { i - required_window_size + 1 } else { 0 };
        let window_mean = values.rows(window_start, i - window_start + 1).mean();
        twap_values.push(window_mean);
    }

    eprintln!("TWAP 7d calculation completed with {} values", twap_values.len());
    Ok(twap_values)
}

