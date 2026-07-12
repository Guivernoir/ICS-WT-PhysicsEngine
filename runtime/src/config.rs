use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use thiserror::Error;

#[derive(Clone, Debug)]
pub struct Settings {
    pub bind_addr: SocketAddr,
    pub hmi_dir: PathBuf,
    pub modbus: ModbusSettings,
    pub simulation: SimulationSettings,
    pub tick_interval: Duration,
}

#[derive(Clone, Debug)]
pub struct ModbusSettings {
    pub enabled: bool,
    pub bind_addr: SocketAddr,
}

#[derive(Clone, Debug)]
pub struct SimulationSettings {
    pub python: String,
    pub module: String,
    pub python_path: PathBuf,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("{name} has invalid value {value:?}: {source}")]
    SocketAddress {
        name: &'static str,
        value: String,
        source: std::net::AddrParseError,
    },
    #[error("{name} has invalid integer value {value:?}: {source}")]
    Integer {
        name: &'static str,
        value: String,
        source: std::num::ParseIntError,
    },
    #[error(
        "{name} has invalid boolean value {value:?}; expected one of true/false, yes/no, on/off, or 1/0"
    )]
    Boolean { name: &'static str, value: String },
}

impl Settings {
    pub fn from_env() -> Result<Self, ConfigError> {
        Ok(Self {
            bind_addr: parse_socket("HS_RUNTIME_BIND_ADDR", "127.0.0.1:8088")?,
            hmi_dir: env::var("HS_RUNTIME_HMI_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| default_hmi_dir()),
            modbus: ModbusSettings {
                enabled: parse_bool("HS_RUNTIME_MODBUS_ENABLED", false)?,
                bind_addr: parse_socket("HS_RUNTIME_MODBUS_BIND_ADDR", "127.0.0.1:5502")?,
            },
            simulation: SimulationSettings {
                python: env::var("HS_RUNTIME_PYTHON").unwrap_or_else(|_| "python3".to_string()),
                module: env::var("HS_RUNTIME_SIM_MODULE")
                    .unwrap_or_else(|_| "hydrasim.simulation_worker".to_string()),
                python_path: env::var("HS_RUNTIME_PYTHONPATH")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| default_python_path()),
            },
            tick_interval: Duration::from_millis(parse_u64("HS_RUNTIME_TICK_MS", "1000")?),
        })
    }
}

fn parse_socket(name: &'static str, default: &str) -> Result<SocketAddr, ConfigError> {
    let value = env::var(name).unwrap_or_else(|_| default.to_string());
    value.parse().map_err(|source| ConfigError::SocketAddress {
        name,
        value,
        source,
    })
}

fn parse_u64(name: &'static str, default: &str) -> Result<u64, ConfigError> {
    let value = env::var(name).unwrap_or_else(|_| default.to_string());
    value.parse().map_err(|source| ConfigError::Integer {
        name,
        value,
        source,
    })
}

fn parse_bool(name: &'static str, default: bool) -> Result<bool, ConfigError> {
    let default_value = if default { "true" } else { "false" };
    let value = env::var(name).unwrap_or_else(|_| default_value.to_string());
    parse_bool_value(name, &value)
}

fn parse_bool_value(name: &'static str, value: &str) -> Result<bool, ConfigError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(ConfigError::Boolean {
            name,
            value: value.to_string(),
        }),
    }
}

fn default_python_path() -> PathBuf {
    project_root().join("src")
}

fn default_hmi_dir() -> PathBuf {
    project_root().join("hmi").join("build")
}

fn project_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_localhost_only() {
        assert_eq!(
            parse_socket("HS_TEST_BIND_ADDR", "127.0.0.1:8088")
                .map(|bind| bind.to_string())
                .ok(),
            Some("127.0.0.1:8088".to_string()),
        );
    }

    #[test]
    fn default_python_path_points_at_source_tree() {
        assert!(default_python_path().ends_with("src"));
    }

    #[test]
    fn default_hmi_dir_points_at_static_build() {
        assert!(default_hmi_dir().ends_with(Path::new("hmi/build")));
    }

    #[test]
    fn boolean_values_accept_operator_friendly_forms() {
        assert!(matches!(parse_bool_value("HS_TEST_BOOL", "true"), Ok(true)));
        assert!(matches!(parse_bool_value("HS_TEST_BOOL", "1"), Ok(true)));
        assert!(matches!(parse_bool_value("HS_TEST_BOOL", "yes"), Ok(true)));
        assert!(matches!(parse_bool_value("HS_TEST_BOOL", "on"), Ok(true)));

        assert!(matches!(
            parse_bool_value("HS_TEST_BOOL", "false"),
            Ok(false)
        ));
        assert!(matches!(parse_bool_value("HS_TEST_BOOL", "0"), Ok(false)));
        assert!(matches!(parse_bool_value("HS_TEST_BOOL", "no"), Ok(false)));
        assert!(matches!(parse_bool_value("HS_TEST_BOOL", "off"), Ok(false)));
    }

    #[test]
    fn boolean_values_reject_ambiguous_input() {
        assert!(parse_bool_value("HS_TEST_BOOL", "enabled").is_err());
    }
}
