use risc0_zkvm::guest::env;

mod pricing_data;


fn main() {
    // Read the base fees as Vec<Option<f64>>
    let base_fees: Vec<Option<f64>> = env::read();

    let volatility = pricing_data::volatility::calculate_volatility(base_fees.clone());
    let twap = pricing_data::twap::calculate_twap(base_fees);
    // let reserve_price = pricing_data::reserve_price::calculate_reserve_price(blocks.clone());

    // Commit the results, converting them to Option<f64>
    env::commit(&(volatility.ok(), twap.ok()));
}
