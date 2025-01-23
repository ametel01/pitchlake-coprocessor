use super::utils::Fixed;
use eyre::Result;

fn natural_log(x: Fixed) -> Result<Fixed> {
    if x <= Fixed::ZERO {
        return Err(eyre::eyre!("Cannot take logarithm of non-positive number"));
    }
    let mut power = 0i32;
    let two = Fixed::from_num(2);
    let one = Fixed::from_num(1);
    let mut val = x;
    while val >= two {
        val = val / two;
        power += 1;
    }
    while val < one {
        val = val * two;
        power -= 1;
    }
    let base_ln = Fixed::LN_2 * Fixed::from_num(power);
    let frac = val - one;
    let frac_contribution = frac * Fixed::LN_2;
    Ok(base_ln + frac_contribution)
}

pub fn calculate_volatility_fixed(base_fees: &[Option<Fixed>]) -> Result<Fixed> {
    let mut count = 0;
    let mut sum = Fixed::ZERO;
    let mut sum_sq = Fixed::ZERO;
    for i in 1..base_fees.len() {
        if let (Some(curr), Some(prev)) = (base_fees[i], base_fees[i - 1]) {
            if prev == Fixed::ZERO {
                continue;
            }
            let ratio = curr / prev;
            if let Ok(r) = natural_log(ratio) {
                count += 1;
                sum += r;
                sum_sq += r * r;
            }
        }
    }
    if count == 0 {
        return Ok(Fixed::ZERO);
    }
    let mean = sum / Fixed::from_num(count);
    let variance = sum_sq / Fixed::from_num(count) - mean * mean;
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
        let base_fees = vec![];
        assert_eq!(calculate_volatility_fixed(&base_fees).unwrap(), Fixed::ZERO);
    }

    #[test]
    fn test_volatility_with_fixed_fees() {
        let base_fees = vec![
            Some(Fixed::from_num(100)),
            Some(Fixed::from_num(110)),
            Some(Fixed::from_num(90)),
            Some(Fixed::from_num(105)),
        ];
        let volatility = calculate_volatility_fixed(&base_fees).unwrap();
        assert!(volatility > Fixed::ZERO);
    }
}
