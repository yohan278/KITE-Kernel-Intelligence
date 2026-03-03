use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::info;

mod sampler;
mod stream;
mod recorder;

pub mod proto {
    tonic::include_proto!("h100_telemetry");
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum Mode {
    Stream,
    Record,
    Both,
}

#[derive(Parser, Debug)]
#[command(name = "h100-energy-sampler", about = "Full NVML telemetry sampler for H100 GPUs")]
struct Args {
    #[arg(long, default_value = "both")]
    mode: Mode,

    #[arg(long, default_value = "[::1]:50052")]
    grpc_addr: String,

    #[arg(long, default_value = "50")]
    interval_ms: u64,

    #[arg(long, default_value = "telemetry_output")]
    output_dir: PathBuf,

    #[arg(long, default_value = "h100")]
    prefix: String,

    /// Record in CSV (default) or JSONL.
    #[arg(long, default_value = "csv")]
    format: String,

    /// Maximum number of samples to record (0 = unlimited).
    #[arg(long, default_value = "0")]
    max_samples: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    info!(?args, "Starting h100-energy-sampler");

    let mut sampler = sampler::NvmlFullSampler::new()?;
    info!("Detected {} GPU(s)", sampler.gpu_count());

    match args.mode {
        Mode::Stream => {
            let addr: SocketAddr = args.grpc_addr.parse()?;
            stream::run_server(sampler, addr).await?;
        }
        Mode::Record => {
            run_recorder(&args, &mut sampler).await?;
        }
        Mode::Both => {
            let addr: SocketAddr = args.grpc_addr.parse()?;
            let sampler2 = sampler::NvmlFullSampler::new()?;
            let server_handle = tokio::spawn(async move {
                stream::run_server(sampler2, addr).await
            });

            let record_handle = tokio::spawn(async move {
                run_recorder(&args, &mut sampler).await
            });

            tokio::select! {
                res = server_handle => { res??; }
                res = record_handle => { res??; }
            }
        }
    }

    Ok(())
}

async fn run_recorder(args: &Args, sampler: &mut sampler::NvmlFullSampler) -> Result<()> {
    let fmt = match args.format.as_str() {
        "jsonl" => recorder::RecordFormat::Jsonl,
        _ => recorder::RecordFormat::Csv,
    };
    let path = recorder::output_path(&args.output_dir, &args.prefix, &fmt);
    let mut rec = recorder::Recorder::new(&path, fmt)?;
    info!("Recording to {}", path.display());

    let interval = std::time::Duration::from_millis(args.interval_ms);
    let mut ticker = tokio::time::interval(interval);
    let mut count: u64 = 0;

    loop {
        ticker.tick().await;
        let sample = sampler.sample()?;
        rec.write_sample(&sample)?;
        count += 1;

        if count % 100 == 0 {
            rec.flush()?;
        }
        if args.max_samples > 0 && count >= args.max_samples {
            info!("Reached max_samples={}, stopping", args.max_samples);
            break;
        }
    }

    rec.flush()?;
    Ok(())
}
