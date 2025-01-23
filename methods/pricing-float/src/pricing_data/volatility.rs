use eyre::Result;

// Returns volatility as BPS (i.e., 5001 means VOL=50.01%)
pub fn calculate_volatility(base_fees: Vec<Option<f64>>) -> Result<f64> {
    // Calculate log returns
    let mut returns: Vec<f64> = Vec::new();
    for i in 1..base_fees.len() {
        if let (Some(basefee_current), Some(basefee_previous)) = 
            (base_fees[i], base_fees[i - 1])
        {
            // If the previous base fee is zero, skip to the next iteration
            if basefee_previous == 0.0 {
                continue;
            }

            // Calculate log return and add it to the returns vector
            returns.push((basefee_current / basefee_previous).ln());
        }
    }

    // If there are no returns the volatility is 0
    if returns.is_empty() {
        return Ok(0f64);
    }

    // Calculate average returns
    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

    // Calculate variance of average returns
    let variance: f64 = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    // Square root the variance to get the volatility, translate to BPS (integer)
    Ok((variance.sqrt() * 10_000.0).round())
}
