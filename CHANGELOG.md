# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-06-26

### Fixed
- Fixed action format and efficiency-aware battery constraints: charging/discharging power now properly accounts for efficiency losses in SOC calculations
- MILP solver now perfectly matches environment execution with zero divergence in battery state trajectory
- Corrected space-to-charge constraint to account for charging efficiency (eta_cha)
- Corrected energy-available constraint to account for discharge efficiency (eta_dis)

## [0.1.1] - 2026-06-25

### Fixed
- Added missing pulp>=3.3.2 dependency for MILP solver

## [0.1.0] - 2026-06-24

### Added
- Initial release with physics-based battery simulation
- Battery energy storage system (BESS) model with commercial datasheet parameters
- Prosumer energy node implementation
- Gymnasium-compliant reinforcement learning environment
- Built-in MILP solver for benchmarking
- Comprehensive documentation and examples
