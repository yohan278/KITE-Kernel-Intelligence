use anyhow::Result;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::proto::TelemetrySample;

pub enum RecordFormat {
    Csv,
    Jsonl,
}

pub struct Recorder {
    writer: BufWriter<File>,
    format: RecordFormat,
    header_written: bool,
}

impl Recorder {
    pub fn new(path: &Path, format: RecordFormat) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            format,
            header_written: false,
        })
    }

    pub fn csv(path: &Path) -> Result<Self> {
        Self::new(path, RecordFormat::Csv)
    }

    pub fn jsonl(path: &Path) -> Result<Self> {
        Self::new(path, RecordFormat::Jsonl)
    }

    pub fn write_sample(&mut self, sample: &TelemetrySample) -> Result<()> {
        match self.format {
            RecordFormat::Csv => self.write_csv(sample),
            RecordFormat::Jsonl => self.write_jsonl(sample),
        }
    }

    fn write_csv(&mut self, sample: &TelemetrySample) -> Result<()> {
        if !self.header_written {
            writeln!(
                self.writer,
                "timestamp_nanos,gpu_index,gpu_util_pct,mem_util_pct,temperature_c,\
                 clock_sm_mhz,clock_mem_mhz,power_limit_w,total_energy_mj,\
                 memory_used_bytes,memory_total_bytes,pcie_tx_kbps,pcie_rx_kbps,\
                 throttle_reasons,fan_speed_pct,power_draw_w,energy_since_baseline_j,\
                 cpu_memory_used_mb"
            )?;
            self.header_written = true;
        }
        for gpu in &sample.gpus {
            writeln!(
                self.writer,
                "{},{},{:.2},{:.2},{:.1},{},{},{:.1},{},{},{},{},{},{},{},{:.3},{:.6},{:.1}",
                sample.timestamp_nanos,
                gpu.gpu_index,
                gpu.gpu_utilization_pct,
                gpu.memory_utilization_pct,
                gpu.temperature_c,
                gpu.clock_sm_mhz,
                gpu.clock_mem_mhz,
                gpu.power_limit_w,
                gpu.total_energy_mj,
                gpu.memory_used_bytes,
                gpu.memory_total_bytes,
                gpu.pcie_tx_kbps,
                gpu.pcie_rx_kbps,
                gpu.throttle_reasons,
                gpu.fan_speed_pct,
                gpu.power_draw_w,
                gpu.energy_since_baseline_j,
                sample.cpu_memory_used_mb,
            )?;
        }
        Ok(())
    }

    fn write_jsonl(&mut self, sample: &TelemetrySample) -> Result<()> {
        for gpu in &sample.gpus {
            let record = serde_json::json!({
                "timestamp_nanos": sample.timestamp_nanos,
                "hostname": sample.hostname,
                "gpu_index": gpu.gpu_index,
                "gpu_utilization_pct": gpu.gpu_utilization_pct,
                "memory_utilization_pct": gpu.memory_utilization_pct,
                "temperature_c": gpu.temperature_c,
                "clock_sm_mhz": gpu.clock_sm_mhz,
                "clock_mem_mhz": gpu.clock_mem_mhz,
                "power_limit_w": gpu.power_limit_w,
                "total_energy_mj": gpu.total_energy_mj,
                "memory_used_bytes": gpu.memory_used_bytes,
                "memory_total_bytes": gpu.memory_total_bytes,
                "pcie_tx_kbps": gpu.pcie_tx_kbps,
                "pcie_rx_kbps": gpu.pcie_rx_kbps,
                "throttle_reasons": gpu.throttle_reasons,
                "fan_speed_pct": gpu.fan_speed_pct,
                "power_draw_w": gpu.power_draw_w,
                "energy_since_baseline_j": gpu.energy_since_baseline_j,
                "cpu_memory_used_mb": sample.cpu_memory_used_mb,
            });
            serde_json::to_writer(&mut self.writer, &record)?;
            writeln!(self.writer)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Build output path from base dir and format.
pub fn output_path(base_dir: &Path, prefix: &str, format: &RecordFormat) -> PathBuf {
    let ext = match format {
        RecordFormat::Csv => "csv",
        RecordFormat::Jsonl => "jsonl",
    };
    base_dir.join(format!("{prefix}_telemetry.{ext}"))
}
