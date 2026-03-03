use anyhow::Result;
use sysinfo::System;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, trace, warn};

use crate::proto::{GpuSample, TelemetrySample};

/// Per-GPU tracking state for energy deltas and derived power.
struct GpuState {
    energy_baseline_mj: u64,
    last_energy_mj: u64,
    last_ts: Option<std::time::Instant>,
}

pub struct NvmlFullSampler {
    #[cfg(not(target_os = "macos"))]
    devices: Vec<nvml_wrapper::Device<'static>>,
    gpu_states: Vec<GpuState>,
    sysinfo: parking_lot::Mutex<System>,
    hostname: String,
}

impl NvmlFullSampler {
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            anyhow::bail!("NvmlFullSampler requires NVIDIA GPUs (not available on macOS)");
        }

        #[cfg(not(target_os = "macos"))]
        {
            let nvml = nvml_wrapper::Nvml::init()
                .map_err(|e| anyhow::anyhow!("NVML init failed: {e}"))?;
            let nvml_static: &'static nvml_wrapper::Nvml = Box::leak(Box::new(nvml));

            let count = nvml_static.device_count()
                .map_err(|e| anyhow::anyhow!("device_count: {e}"))?;

            let indices = Self::parse_cuda_visible_devices(count);
            let mut devices = Vec::new();
            let mut gpu_states = Vec::new();

            for idx in indices {
                match nvml_static.device_by_index(idx) {
                    Ok(dev) => {
                        let baseline = dev.total_energy_consumption().unwrap_or(0);
                        let name = dev.name().unwrap_or_default();
                        debug!(idx, %name, "Attached GPU");
                        gpu_states.push(GpuState {
                            energy_baseline_mj: baseline,
                            last_energy_mj: baseline,
                            last_ts: None,
                        });
                        devices.push(dev);
                    }
                    Err(e) => warn!(idx, "Skipping GPU: {e}"),
                }
            }
            if devices.is_empty() {
                anyhow::bail!("No NVIDIA GPUs found");
            }

            let hostname = gethostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".into());

            Ok(Self {
                devices,
                gpu_states,
                sysinfo: parking_lot::Mutex::new(System::new_all()),
                hostname,
            })
        }
    }

    pub fn gpu_count(&self) -> usize {
        #[cfg(not(target_os = "macos"))]
        { self.devices.len() }
        #[cfg(target_os = "macos")]
        { 0 }
    }

    /// Collect a single telemetry snapshot across all GPUs.
    pub fn sample(&mut self) -> Result<TelemetrySample> {
        let now_instant = std::time::Instant::now();
        let ts_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;

        let mut gpus = Vec::with_capacity(self.gpu_count());

        #[cfg(not(target_os = "macos"))]
        for (i, device) in self.devices.iter().enumerate() {
            let state = &mut self.gpu_states[i];
            let dt_s = state.last_ts.map(|t| now_instant.duration_since(t).as_secs_f64()).unwrap_or(0.0);
            state.last_ts = Some(now_instant);

            // Utilization
            let (gpu_util, mem_util) = device.utilization_rates()
                .map(|u| (u.gpu as f64, u.memory as f64))
                .unwrap_or((-1.0, -1.0));

            // Temperature
            let temp = device
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .map(|t| t as f64)
                .unwrap_or(-1.0);

            // Clocks
            let clock_sm = device
                .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)
                .unwrap_or(0);
            let clock_mem = device
                .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)
                .unwrap_or(0);

            // Power limit
            let power_limit_mw = device.enforced_power_limit().unwrap_or(0);
            let power_limit_w = power_limit_mw as f64 / 1000.0;

            // Energy
            let energy_mj = device.total_energy_consumption().unwrap_or(state.last_energy_mj);
            let delta_from_baseline = if energy_mj >= state.energy_baseline_mj {
                energy_mj - state.energy_baseline_mj
            } else {
                (u64::MAX - state.energy_baseline_mj) + energy_mj
            };

            // Derive power from energy delta
            let power_draw_w = if dt_s > 0.0 {
                let delta_mj = if energy_mj >= state.last_energy_mj {
                    energy_mj - state.last_energy_mj
                } else {
                    (u64::MAX - state.last_energy_mj) + energy_mj
                };
                (delta_mj as f64 / 1000.0) / dt_s
            } else {
                device.power_usage().map(|mw| mw as f64 / 1000.0).unwrap_or(-1.0)
            };
            state.last_energy_mj = energy_mj;

            // Memory
            let (mem_used, mem_total) = device.memory_info()
                .map(|m| (m.used, m.total))
                .unwrap_or((0, 0));

            // PCIe throughput
            let pcie_tx = device
                .pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Send)
                .unwrap_or(0) as u64;
            let pcie_rx = device
                .pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Receive)
                .unwrap_or(0) as u64;

            // Throttle reasons
            let throttle = device.current_throttle_reasons()
                .map(|r| r.bits())
                .unwrap_or(0);

            // Fan speed
            let fan = device.fan_speed(0).unwrap_or(0);

            trace!(i, gpu_util, power_draw_w, energy_mj, "GPU sample");

            gpus.push(GpuSample {
                gpu_index: i as u32,
                gpu_utilization_pct: gpu_util,
                memory_utilization_pct: mem_util,
                temperature_c: temp,
                clock_sm_mhz: clock_sm,
                clock_mem_mhz: clock_mem,
                power_limit_w,
                total_energy_mj: energy_mj,
                memory_used_bytes: mem_used,
                memory_total_bytes: mem_total,
                pcie_tx_kbps: pcie_tx,
                pcie_rx_kbps: pcie_rx,
                throttle_reasons: throttle,
                fan_speed_pct: fan,
                power_draw_w,
                energy_since_baseline_j: delta_from_baseline as f64 / 1000.0,
            });
        }

        let cpu_mem_mb = {
            let mut sys = self.sysinfo.lock();
            sys.refresh_memory();
            sys.used_memory() as f64 / (1024.0 * 1024.0)
        };

        Ok(TelemetrySample {
            timestamp_nanos: ts_nanos,
            gpus,
            cpu_memory_used_mb: cpu_mem_mb,
            hostname: self.hostname.clone(),
        })
    }

    #[cfg(not(target_os = "macos"))]
    fn parse_cuda_visible_devices(total: u32) -> Vec<u32> {
        match std::env::var("CUDA_VISIBLE_DEVICES") {
            Ok(val) => {
                let trimmed = val.trim();
                if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
                    return Vec::new();
                }
                trimmed
                    .split(',')
                    .filter_map(|s| s.trim().parse::<u32>().ok())
                    .filter(|&i| i < total)
                    .collect()
            }
            Err(_) => (0..total).collect(),
        }
    }
}
