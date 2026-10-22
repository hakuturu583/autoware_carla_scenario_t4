//! T4 datasets of autoware_carla_scenario runs: the writer (`writer`), and,
//! with the `python` feature, its binding,
//! `autoware_carla_scenario_t4._core.T4Writer`.

pub mod carla;
pub mod geometry;
pub mod tables;
pub mod writer;

#[cfg(feature = "python")]
mod python;
