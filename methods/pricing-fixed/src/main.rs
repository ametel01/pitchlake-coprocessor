use risc0_zkvm::guest::env;
use guest_pricing::{twap::calculate_twap_fixed, volatility::calculate_volatility_fixed};
use fixed::{types::extra::U64, FixedI128};

type Fixed = FixedI128<U64>;

fn main() {
    let base_fees: Vec<Option<Fixed>> = env::read();

    let volatility = calculate_volatility_fixed(&base_fees);
    let twap = calculate_twap_fixed(&base_fees);

    env::commit(&(volatility.ok(), twap.ok()));
}
