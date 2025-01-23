use eth_rlp_types::BlockHeader;
use eyre::Result;

use super::utils::{hex_string_to_fixed, Fixed};

// Calculates natural logarithm using LN_2 constant
// Uses the identity: ln(x) = ln(2) * log2(x)
fn natural_log(x: Fixed) -> Result<Fixed> {
    if x <= Fixed::ZERO {
        return Err(eyre::eyre!("Cannot take logarithm of non-positive number"));
    }

    // Find the highest power of 2 less than x
    let mut power = 0i32;
    let mut val = x;
    while val >= Fixed::from_num(2) {
        val = val / Fixed::from_num(2);
        power += 1;
    }
    while val < Fixed::from_num(1) {
        val = val * Fixed::from_num(2);
        power -= 1;
    }

    // Now val is in [1, 2)
    // Use the fact that ln(x) = ln(2) * log2(x)
    let base_ln = Fixed::LN_2 * Fixed::from_num(power);

    // For the fractional part, use linear interpolation between known points
    let frac = val - Fixed::from_num(1); // Distance from 1
    let frac_contribution = frac * Fixed::LN_2; // Approximate ln using linear interpolation

    Ok(base_ln + frac_contribution)
}

// Returns volatility as BPS (i.e., 5001 means VOL=50.01%)
pub fn calculate_volatility(blocks: Vec<BlockHeader>) -> Result<Fixed> {
    // Calculate log returns
    let mut returns: Vec<Fixed> = Vec::new();
    for i in 1..blocks.len() {
        if let (Some(ref basefee_current), Some(ref basefee_previous)) =
            (&blocks[i].base_fee_per_gas, &blocks[i - 1].base_fee_per_gas)
        {
            // Convert base fees from hex string to Fixed
            let basefee_current = hex_string_to_fixed(basefee_current)?;
            let basefee_previous = hex_string_to_fixed(basefee_previous)?;

            // If the previous base fee is zero, skip to the next iteration
            if basefee_previous == Fixed::ZERO {
                continue;
            }

            // Calculate log return and add it to the returns vector
            let ratio = basefee_current / basefee_previous;
            if let Ok(ln_return) = natural_log(ratio) {
                returns.push(ln_return);
            }
        }
    }

    // If there are no returns the volatility is 0
    if returns.is_empty() {
        return Ok(Fixed::ZERO);
    }

    // Calculate average returns
    let sum = returns.iter().fold(Fixed::ZERO, |acc, &x| acc + x);
    let mean_return = sum / Fixed::from_num(returns.len());

    // Calculate variance of average returns
    let squared_diffs = returns.iter().map(|&r| {
        let diff = r - mean_return;
        diff * diff
    });
    let variance =
        squared_diffs.fold(Fixed::ZERO, |acc, x| acc + x) / Fixed::from_num(returns.len());

    // Square root the variance to get the volatility, translate to BPS (integer)
    Ok(variance.sqrt() * Fixed::from_num(10_000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_log() {
        let one = Fixed::from_num(1);
        let e = Fixed::from_num(2.718281828459045);

        assert!(natural_log(one).unwrap().abs() < Fixed::from_num(0.0001));
        assert!((natural_log(e).unwrap() - one).abs() < Fixed::from_num(0.1));
    }

    #[test]
    fn test_sqrt() {
        let four = Fixed::from_num(4);
        let nine = Fixed::from_num(9);

        assert_eq!(four.sqrt(), Fixed::from_num(2));
        assert_eq!(nine.sqrt(), Fixed::from_num(3));
    }

    #[test]
    fn test_zero_volatility() {
        let blocks = vec![];
        assert_eq!(calculate_volatility(blocks).unwrap(), Fixed::ZERO);
    }
}
