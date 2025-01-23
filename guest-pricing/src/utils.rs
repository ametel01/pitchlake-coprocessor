use fixed::{types::extra::U64, FixedI128};

// Define a type alias for our fixed-point number with 64 fractional bits
pub type Fixed = FixedI128<U64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_zero() {
        assert_eq!(Fixed::ZERO, Fixed::from_num(0));
    }

    #[test]
    fn test_fixed_arithmetic() {
        let a = Fixed::from_num(100);
        let b = Fixed::from_num(200);
        assert_eq!(a + b, Fixed::from_num(300));
        assert_eq!(b - a, Fixed::from_num(100));
        assert_eq!(a * Fixed::from_num(2), Fixed::from_num(200));
        assert_eq!(b / Fixed::from_num(2), Fixed::from_num(100));
    }
}
