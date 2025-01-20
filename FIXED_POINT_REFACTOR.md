# Fixed Point Arithmetic Refactoring Guide

This document outlines the libraries and functions that need to be refactored to move from floating-point to fixed-point arithmetic.

## Libraries Requiring Refactoring

### nalgebra
Core matrix operations that need fixed-point implementations:
- [ ] DMatrix/DVector basic operations
  - Addition
  - Subtraction
  - Multiplication
  - Division
- [ ] Statistical operations
  - `matrix.mean()`
  - `matrix.sum()`
- [ ] Linear algebra operations
  - `matrix.try_inverse()`
  - Matrix multiplication
  - Vector operations

### rand_distr
Random number generation and distributions:
- [ ] Normal distribution sampling
  - Implementation needed for fixed-point normal distribution
  - Consider using lookup tables for efficiency
- [ ] Binomial distribution sampling
  - Fixed-point implementation required
  - May need to adjust probability calculations

### statrs
Statistical computations:
- [ ] Binomial distribution calculations
  - Probability mass function
  - Cumulative distribution function
  - Parameter estimations

### Standard Library Math Operations
Basic mathematical operations requiring fixed-point implementations:
- [ ] Exponential function (`f64::exp()`)
- [ ] Natural logarithm (`f64::ln()`)
- [ ] Square root (`f64::sqrt()`)
- [ ] Trigonometric functions
  - `f64::sin()`
  - `f64::cos()`
- [ ] Mathematical constants
  - PI
  - e

## Implementation Strategy

### 1. Fixed-Point Type Selection
- Consider using the `fixed` crate
- Determine required precision (number of fractional bits)
- Define custom fixed-point type: 