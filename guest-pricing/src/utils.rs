use eth_rlp_types::BlockHeader;
use eyre::Result;
use fixed::{types::extra::U64, FixedI128};

// Define a type alias for our fixed-point number with 64 fractional bits
pub type Fixed = FixedI128<U64>;

pub fn hex_string_to_fixed(hex_str: &String) -> Result<Fixed> {
    let stripped = hex_str.trim_start_matches("0x");
    u128::from_str_radix(stripped, 16)
        .map(|value| Fixed::from_num(value))
        .map_err(|e| eyre::eyre!("Error converting hex string '{}' to Fixed: {}", hex_str, e))
}

/// Extracts base fee values from a vector of block headers
pub fn extract_base_fees(headers: &[BlockHeader]) -> Vec<String> {
    extract_header_values(headers, |header| {
        header
            .base_fee_per_gas
            .clone()
            .unwrap_or_else(|| "0x0".to_string())
    })
}

/// Generic function to extract values from block headers
pub fn extract_header_values<F, T>(headers: &[BlockHeader], extractor: F) -> Vec<T>
where
    F: Fn(&BlockHeader) -> T,
{
    headers.iter().map(extractor).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_string_to_fixed_zero_value() {
        let result = hex_string_to_fixed(&"0x0".to_string());

        assert_eq!(result.unwrap(), Fixed::ZERO);
    }

    #[test]
    fn test_hex_string_to_fixed_prefixed_value() {
        let result = hex_string_to_fixed(&"0x12345".to_string());

        assert_eq!(result.unwrap(), Fixed::from_num(74565));
    }

    #[test]
    fn test_hex_string_to_fixed_non_prefixed_value() {
        let result = hex_string_to_fixed(&"12345".to_string());

        assert_eq!(result.unwrap(), Fixed::from_num(74565));
    }

    #[test]
    fn test_hex_string_to_fixed_invalid_value() {
        let result = hex_string_to_fixed(&"shouldpanic".to_string());

        assert!(result.is_err(), "Expected an error, but got {:?}", result);
    }
}
