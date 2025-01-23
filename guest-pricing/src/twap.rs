use super::utils::{extract_base_fees, hex_string_to_fixed, Fixed};
use eth_rlp_types::BlockHeader;
use eyre::{anyhow, Result};

/// Calculates TWAP from a vector of hex strings representing base fees
pub fn calculate_twap(base_fees: Vec<String>) -> Result<Fixed> {
    if base_fees.is_empty() {
        return Err(anyhow!("The provided base fees are empty."));
    }

    let total_base_fee = base_fees
        .iter()
        .try_fold(Fixed::ZERO, |acc, fee| -> Result<Fixed> {
            let fee = hex_string_to_fixed(fee)?;
            Ok(acc + fee)
        })?;

    let len: Fixed = Fixed::from_num(base_fees.len());
    let twap_result: Fixed = total_base_fee / len;

    Ok(twap_result)
}

/// Calculates TWAP from a vector of block headers
pub fn calculate_twap_from_headers(headers: Vec<BlockHeader>) -> Result<Fixed> {
    let base_fees = extract_base_fees(&headers);
    calculate_twap(base_fees)
}
