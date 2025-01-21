# Fixed Point Arithmetic Refactoring Guide

This document outlines the libraries and functions that need to be refactored to move from floating-point to fixed-point arithmetic.

## Libraries Requiring Refactoring

## Original Python Repository
https://github.com/NethermindEth/pitchlake-pricing

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

# Refactoring Timeline and Complexity Analysis

## Quick Wins (1-2 days)
### Mathematical Constants
- [ ] Replace PI and e with fixed-point versions
- Complexity: Low
- Risk: Low
- Just needs careful precision selection

## Short Term (2-4 days)
### Basic Math Operations
- [ ] Basic arithmetic operations (add, subtract, multiply, divide)
- [ ] Square root implementation using Newton's method
- Complexity: Medium-Low
- Risk: Medium
- Well-documented algorithms available

## Medium Term (1-2 weeks)
### Matrix/Vector Basic Operations
- [ ] DMatrix/DVector basic operations
- [ ] Statistical operations (mean, sum)
- Complexity: Medium
- Risk: Medium
- Main challenge is handling precision in accumulations

### Standard Math Functions
- [ ] Exponential function
- [ ] Natural logarithm
- [ ] Basic trigonometric functions
- Complexity: Medium
- Risk: Medium-High
- Can use Taylor series or lookup tables

## Long Term (2-4 weeks)
### Distribution Implementations
- [ ] Normal distribution sampling
- [ ] Binomial distribution sampling
- Complexity: High
- Risk: High
- Requires careful statistical validation

### Complex Matrix Operations
- [ ] Matrix inverse
- [ ] Complex linear algebra operations
- Complexity: Very High
- Risk: High
- Precision critical for numerical stability

## Suggested Implementation Order:

1. Week 1:
   - Mathematical constants
   - Basic arithmetic operations
   - Square root implementation
   - Initial test framework

2. Week 2:
   - Basic matrix operations
   - Mean and sum implementations
   - Standard math functions
   - Expanded test coverage

3. Weeks 3-4:
   - Distribution implementations
   - Complex matrix operations
   - Performance optimization
   - Comprehensive testing

## Risk Factors:
- Precision requirements may extend timeline
- Performance optimization might be needed
- Edge cases could require rework
- Integration testing might reveal issues

## Recommendations:
1. Start with mathematical constants and basic operations
2. Build comprehensive tests early
3. Validate precision requirements before complex implementations
4. Consider parallel development of independent components 

# Two-Developer Implementation Timeline

## Best Case Scenario (3-4 weeks total)

### Week 1
Developer 1:
- Mathematical constants (0.5 day)
- Basic arithmetic operations (2 days)
- Basic matrix operations (2.5 days)

Developer 2:
- Test framework setup (1 day)
- Square root implementation (2 days)
- Mean/sum implementations (2 days)

### Week 2
Developer 1:
- Standard math functions (exp, ln) (3 days)
- Trigonometric functions (2 days)

Developer 2:
- Basic distribution groundwork (3 days)
- Simple matrix transformations (2 days)

### Week 3
Developer 1:
- Matrix inverse operations (5 days)

Developer 2:
- Distribution implementations (5 days)

### Week 4
Both Developers:
- Integration testing (2 days)
- Performance optimization (2 days)
- Bug fixes and refinements (1 day)

## Worst Case Scenario (6-8 weeks total)

### Week 1-2
Developer 1:
- Mathematical constants (1 day)
- Basic arithmetic operations (4 days)
- Basic matrix operations (5 days)
- Unexpected precision issues (4 days)

Developer 2:
- Test framework setup (2 days)
- Square root implementation (4 days)
- Mean/sum implementations (4 days)
- Initial testing reveals edge cases (4 days)

### Week 3-4
Developer 1:
- Standard math functions (exp, ln) (5 days)
- Trigonometric functions (5 days)
- Precision optimization (4 days)

Developer 2:
- Basic distribution groundwork (5 days)
- Simple matrix transformations (5 days)
- Performance issues resolution (4 days)

### Week 5-6
Developer 1:
- Matrix inverse operations (8 days)
- Numerical stability issues (2 days)

Developer 2:
- Distribution implementations (8 days)
- Statistical accuracy validation (2 days)

### Week 7-8
Both Developers:
- Integration testing (4 days)
- Performance optimization (4 days)
- Bug fixes and refinements (4 days)
- Documentation and final testing (3 days)

## Risk Factors Affecting Timeline

### Technical Risks
- Precision requirements more stringent than expected
- Performance bottlenecks in critical paths
- Numerical stability issues in matrix operations
- Edge cases in distribution sampling

### Process Risks
- Integration challenges between components
- Test coverage gaps
- Dependencies between developer tasks
- Learning curve with fixed-point arithmetic

### External Risks
- Third-party library limitations
- Hardware constraints
- Requirements changes
- Code review cycles

## Mitigation Strategies

1. Early Prototyping
   - Build quick proofs of concept for risky components
   - Validate precision requirements early
   - Test performance characteristics

2. Parallel Development
   - Independent components first
   - Clear interfaces between modules
   - Regular integration points

3. Testing Strategy
   - Continuous testing from day one
   - Automated precision validation
   - Performance benchmarking suite

4. Communication
   - Daily sync between developers
   - Regular code reviews
   - Documentation of decisions and trade-offs

## Critical Path Items
1. Matrix inverse operations
2. Distribution implementations
3. Integration testing
4. Performance optimization

These items should receive priority attention and additional resources if available. 

# Library-Based Implementation Timeline

## nalgebra Library Tasks (Developer 1)
### Week 1-2
- Basic Matrix/Vector Operations
  - Addition/Subtraction (2 days)
  - Multiplication/Division (3 days)
  - Mean/sum implementations (2 days)
  - Test framework for matrix operations (2 days)

### Week 3-4
- Advanced Matrix Operations
  - Matrix inverse operations (5 days)
  - Complex linear algebra operations (5 days)
  - Performance optimization (3 days)
  - Integration testing (2 days)

Total Estimated Time: 4 weeks
Risk Level: High (due to numerical stability requirements)

## rand_distr & statrs Libraries (Developer 2)
### Week 1-2
- Distribution Framework
  - Test framework setup (2 days)
  - Basic distribution groundwork (3 days)
  - Normal distribution sampling (4 days)
  - Initial testing (2 days)

### Week 3-4
- Advanced Distribution Implementation
  - Binomial distribution sampling (4 days)
  - Probability mass function (3 days)
  - Cumulative distribution function (3 days)
  - Parameter estimations (3 days)
  - Statistical validation (2 days)

Total Estimated Time: 4 weeks
Risk Level: Medium-High

## Standard Library Math Operations (Can be split between devs as needed)
### Week 1-2
- Basic Operations
  - Mathematical constants (PI, e) (1 day)
  - Basic arithmetic operations (2 days)
  - Square root implementation (2 days)

### Week 3-4
- Advanced Math Functions
  - Exponential function (3 days)
  - Natural logarithm (3 days)
  - Trigonometric functions (4 days)
  - Testing and validation (2 days)

Total Estimated Time: 2-4 weeks
Risk Level: Medium

## Parallel Development Benefits
1. Clear Ownership
   - Developer 1 focuses on matrix/linear algebra (nalgebra)
   - Developer 2 focuses on statistical distributions (rand_distr & statrs)
   - Standard library math can be divided based on availability

2. Reduced Dependencies
   - Matrix operations can progress independently of distributions
   - Each developer can maintain their own test suite
   - Integration points are well-defined

3. Risk Management
   - Issues in one library don't block progress in others
   - Parallel testing and validation
   - Independent performance optimization

## Integration Points
1. End of Week 2
   - Basic operations from all libraries
   - Initial test framework validation
   - Performance baseline establishment

2. End of Week 4
   - Complete feature integration
   - Cross-library testing
   - Performance optimization
   - Documentation review

## Library-Specific Risk Factors

### nalgebra
- Precision loss in matrix operations
- Performance in large matrix calculations
- Numerical stability in inverse operations

### rand_distr & statrs
- Statistical accuracy of distributions
- Performance of random number generation
- Edge cases in probability calculations

### Standard Library Math
- Precision requirements
- Performance of transcendental functions
- Accuracy of approximations

## Mitigation Strategies by Library

### nalgebra
1. Early precision testing
2. Incremental matrix size testing
3. Comparative testing with floating-point

### rand_distr & statrs
1. Statistical validation suite
2. Distribution property testing
3. Performance benchmarking

### Standard Library Math
1. Error bound analysis
2. Lookup table optimization
3. Algorithm selection testing

# Refactoring Timeline and Complexity Analysis

## Quick Wins (1-2 days)
### Mathematical Constants
- [ ] Replace PI and e with fixed-point versions
- Complexity: Low
- Risk: Low
- Just needs careful precision selection

## Short Term (2-4 days)
### Basic Math Operations
- [ ] Basic arithmetic operations (add, subtract, multiply, divide)
- [ ] Square root implementation using Newton's method
- Complexity: Medium-Low
- Risk: Medium
- Well-documented algorithms available

## Medium Term (1-2 weeks)
### Matrix/Vector Basic Operations
- [ ] DMatrix/DVector basic operations
- [ ] Statistical operations (mean, sum)
- Complexity: Medium
- Risk: Medium
- Main challenge is handling precision in accumulations

### Standard Math Functions
- [ ] Exponential function
- [ ] Natural logarithm
- [ ] Basic trigonometric functions
- Complexity: Medium
- Risk: Medium-High
- Can use Taylor series or lookup tables

## Long Term (2-4 weeks)
### Distribution Implementations
- [ ] Normal distribution sampling
- [ ] Binomial distribution sampling
- Complexity: High
- Risk: High
- Requires careful statistical validation

### Complex Matrix Operations
- [ ] Matrix inverse
- [ ] Complex linear algebra operations
- Complexity: Very High
- Risk: High
- Precision critical for numerical stability

## Suggested Implementation Order:

1. Week 1:
   - Mathematical constants
   - Basic arithmetic operations
   - Square root implementation
   - Initial test framework

2. Week 2:
   - Basic matrix operations
   - Mean and sum implementations
   - Standard math functions
   - Expanded test coverage

3. Weeks 3-4:
   - Distribution implementations
   - Complex matrix operations
   - Performance optimization
   - Comprehensive testing

## Risk Factors:
- Precision requirements may extend timeline
- Performance optimization might be needed
- Edge cases could require rework
- Integration testing might reveal issues

## Recommendations:
1. Start with mathematical constants and basic operations
2. Build comprehensive tests early
3. Validate precision requirements before complex implementations
4. Consider parallel development of independent components 

# Two-Developer Implementation Timeline

## Best Case Scenario (3-4 weeks total)

### Week 1
Developer 1:
- Mathematical constants (0.5 day)
- Basic arithmetic operations (2 days)
- Basic matrix operations (2.5 days)

Developer 2:
- Test framework setup (1 day)
- Square root implementation (2 days)
- Mean/sum implementations (2 days)

### Week 2
Developer 1:
- Standard math functions (exp, ln) (3 days)
- Trigonometric functions (2 days)

Developer 2:
- Basic distribution groundwork (3 days)
- Simple matrix transformations (2 days)

### Week 3
Developer 1:
- Matrix inverse operations (5 days)

Developer 2:
- Distribution implementations (5 days)

### Week 4
Both Developers:
- Integration testing (2 days)
- Performance optimization (2 days)
- Bug fixes and refinements (1 day)

## Worst Case Scenario (6-8 weeks total)

### Week 1-2
Developer 1:
- Mathematical constants (1 day)
- Basic arithmetic operations (4 days)
- Basic matrix operations (5 days)
- Unexpected precision issues (4 days)

Developer 2:
- Test framework setup (2 days)
- Square root implementation (4 days)
- Mean/sum implementations (4 days)
- Initial testing reveals edge cases (4 days)

### Week 3-4
Developer 1:
- Standard math functions (exp, ln) (5 days)
- Trigonometric functions (5 days)
- Precision optimization (4 days)

Developer 2:
- Basic distribution groundwork (5 days)
- Simple matrix transformations (5 days)
- Performance issues resolution (4 days)

### Week 5-6
Developer 1:
- Matrix inverse operations (8 days)
- Numerical stability issues (2 days)

Developer 2:
- Distribution implementations (8 days)
- Statistical accuracy validation (2 days)

### Week 7-8
Both Developers:
- Integration testing (4 days)
- Performance optimization (4 days)
- Bug fixes and refinements (4 days)
- Documentation and final testing (3 days)

## Risk Factors Affecting Timeline

### Technical Risks
- Precision requirements more stringent than expected
- Performance bottlenecks in critical paths
- Numerical stability issues in matrix operations
- Edge cases in distribution sampling

### Process Risks
- Integration challenges between components
- Test coverage gaps
- Dependencies between developer tasks
- Learning curve with fixed-point arithmetic

### External Risks
- Third-party library limitations
- Hardware constraints
- Requirements changes
- Code review cycles

## Mitigation Strategies

1. Early Prototyping
   - Build quick proofs of concept for risky components
   - Validate precision requirements early
   - Test performance characteristics

2. Parallel Development
   - Independent components first
   - Clear interfaces between modules
   - Regular integration points

3. Testing Strategy
   - Continuous testing from day one
   - Automated precision validation
   - Performance benchmarking suite

4. Communication
   - Daily sync between developers
   - Regular code reviews
   - Documentation of decisions and trade-offs

## Critical Path Items
1. Matrix inverse operations
2. Distribution implementations
3. Integration testing
4. Performance optimization

These items should receive priority attention and additional resources if available. 