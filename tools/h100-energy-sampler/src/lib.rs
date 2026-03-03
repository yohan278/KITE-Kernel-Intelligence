pub mod sampler;
pub mod stream;
pub mod recorder;

pub mod proto {
    tonic::include_proto!("h100_telemetry");
}
