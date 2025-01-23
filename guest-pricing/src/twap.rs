use super::utils::Fixed;
use eyre::{anyhow, Result};

/// Calculates TWAP from a vector of fixed-point base fees
pub fn calculate_twap_fixed(base_fees: &[Option<Fixed>]) -> Result<Fixed> {
    if base_fees.is_empty() {
        return Err(anyhow!("The provided base fees are empty."));
    }

    let total_base_fee = base_fees
        .iter()
        .fold(Fixed::ZERO, |acc, fee| acc + fee.unwrap_or(Fixed::ZERO));

    let len: Fixed = Fixed::from_num(base_fees.len());
    let twap_result: Fixed = total_base_fee / len;

    Ok(twap_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_fixed_empty() {
        let base_fees = vec![];
        assert!(calculate_twap_fixed(&base_fees).is_err());
    }

    #[test]
    fn test_twap_fixed_with_values() {
        let base_fees = vec![
            Some(Fixed::from_num(100)),
            Some(Fixed::from_num(200)),
            Some(Fixed::from_num(300)),
        ];
        let twap = calculate_twap_fixed(&base_fees).unwrap();
        assert_eq!(twap, Fixed::from_num(200));
    }

    #[test]
    fn test_twap_fixed_with_none() {
        let base_fees = vec![Some(Fixed::from_num(100)), None, Some(Fixed::from_num(300))];
        let twap = calculate_twap_fixed(&base_fees).unwrap();
        assert_eq!(twap, Fixed::from_num(400) / Fixed::from_num(3));
    }
}
