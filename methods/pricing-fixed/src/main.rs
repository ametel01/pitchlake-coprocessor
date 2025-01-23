use risc0_zkvm::guest::env;
use eth_rlp_types::BlockHeader;
use guest_pricing::{twap::calculate_twap_from_headers, volatility::calculate_volatility};


fn main() {
    let blocks: Vec<BlockHeader> = env::read();

    let volatility = calculate_volatility(blocks.clone());
    let twap = calculate_twap_from_headers(blocks.clone());
    // let reserve_price = pricing_data::reserve_price::calculate_reserve_price(blocks.clone());

    env::commit(&(volatility.ok(), twap.ok()));
}
