use eyre::{anyhow, Result};

pub fn calculate_twap(base_fees: Vec<Option<f64>>) -> Result<f64> {
    if base_fees.is_empty() {
        return Err(anyhow!("The provided base fees are empty."));
    }

    let total_base_fee = base_fees.iter().try_fold(0.0, |acc, &fee| -> Result<f64> {
        let fee_value = fee.unwrap_or(0.0);
        Ok(acc + fee_value)
    })?;

    let twap_result = total_base_fee / base_fees.len() as f64;

    Ok(twap_result)
}
