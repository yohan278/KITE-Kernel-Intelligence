use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::proto::h100_telemetry_service_server::{H100TelemetryService, H100TelemetryServiceServer};
use crate::proto::{HealthRequest, HealthResponse, StreamRequest, TelemetrySample};
use crate::sampler::NvmlFullSampler;

pub struct TelemetryServer {
    sampler: Arc<Mutex<NvmlFullSampler>>,
    gpu_count: u32,
}

impl TelemetryServer {
    pub fn new(sampler: NvmlFullSampler) -> Self {
        let gpu_count = sampler.gpu_count() as u32;
        Self {
            sampler: Arc::new(Mutex::new(sampler)),
            gpu_count,
        }
    }

    pub fn into_service(self) -> H100TelemetryServiceServer<Self> {
        H100TelemetryServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl H100TelemetryService for TelemetryServer {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            healthy: true,
            gpu_count: self.gpu_count,
            platform: "nvidia_h100".into(),
        }))
    }

    type StreamSamplesStream = ReceiverStream<std::result::Result<TelemetrySample, Status>>;

    async fn stream_samples(
        &self,
        request: Request<StreamRequest>,
    ) -> std::result::Result<Response<Self::StreamSamplesStream>, Status> {
        let interval_ms = request.into_inner().interval_ms;
        let interval = std::time::Duration::from_millis(if interval_ms == 0 { 50 } else { interval_ms as u64 });

        let sampler = self.sampler.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(256);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                let sample = {
                    let mut s = sampler.lock().await;
                    s.sample()
                };
                match sample {
                    Ok(s) => {
                        if tx.send(Ok(s)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Sample error: {e}");
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

pub async fn run_server(sampler: NvmlFullSampler, addr: std::net::SocketAddr) -> Result<()> {
    let server = TelemetryServer::new(sampler);
    info!("Starting gRPC telemetry server on {addr}");
    tonic::transport::Server::builder()
        .add_service(server.into_service())
        .serve(addr)
        .await?;
    Ok(())
}
