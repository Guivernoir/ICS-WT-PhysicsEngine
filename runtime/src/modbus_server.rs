use std::{future::Future, io, net::SocketAddr, pin::Pin};

use tokio::net::TcpListener;
use tokio_modbus::{
    ExceptionCode, Request, Response,
    server::{Service, tcp},
};
use tracing::{debug, info, warn};

use crate::{
    commands::{BinaryCommandKind, NumericCommand},
    runtime::{ModbusImage, RuntimeHandle},
};

#[derive(Clone)]
struct RuntimeModbusService {
    runtime: RuntimeHandle,
}

impl Service for RuntimeModbusService {
    type Request = Request<'static>;
    type Response = Response;
    type Exception = ExceptionCode;
    type Future = Pin<Box<dyn Future<Output = Result<Response, ExceptionCode>> + Send>>;

    fn call(&self, req: Self::Request) -> Self::Future {
        let runtime = self.runtime.clone();
        Box::pin(async move { handle_request(runtime, req).await })
    }
}

pub(crate) async fn serve(bind_addr: SocketAddr, runtime: RuntimeHandle) -> io::Result<()> {
    let listener = TcpListener::bind(bind_addr).await?;
    let server = tcp::Server::new(listener);
    let service = RuntimeModbusService { runtime };
    info!("HydraSim Rust Modbus server listening on {}", bind_addr);

    let on_connected = move |stream, socket_addr| {
        let service = service.clone();
        async move { tcp::accept_tcp_connection(stream, socket_addr, |_| Ok(Some(service.clone()))) }
    };

    server.serve(&on_connected, log_process_error).await
}

fn log_process_error(error: io::Error) {
    match error.kind() {
        io::ErrorKind::UnexpectedEof
        | io::ErrorKind::ConnectionReset
        | io::ErrorKind::ConnectionAborted
        | io::ErrorKind::BrokenPipe => {
            debug!("Modbus client disconnected before a complete frame was processed: {error}");
        }
        _ => {
            warn!("Modbus client sent an invalid or unsupported frame: {error}");
        }
    }
}

async fn handle_request(
    runtime: RuntimeHandle,
    request: Request<'static>,
) -> Result<Response, ExceptionCode> {
    match request {
        Request::ReadInputRegisters(address, count) => {
            let image = runtime
                .modbus_image()
                .await
                .map_err(|_| ExceptionCode::ServerDeviceFailure)?;
            read_words(&image.input_registers, address, count).map(Response::ReadInputRegisters)
        }
        Request::ReadHoldingRegisters(address, count) => {
            let image = runtime
                .modbus_image()
                .await
                .map_err(|_| ExceptionCode::ServerDeviceFailure)?;
            read_words(&image.holding_registers, address, count).map(Response::ReadHoldingRegisters)
        }
        Request::ReadCoils(address, count) => {
            let image = runtime
                .modbus_image()
                .await
                .map_err(|_| ExceptionCode::ServerDeviceFailure)?;
            read_coils(&image, address, count).map(Response::ReadCoils)
        }
        Request::WriteSingleCoil(address, enabled) => {
            let command = BinaryCommandKind::from_address(address)
                .ok_or(ExceptionCode::IllegalDataAddress)?;
            runtime
                .write_binary(command.id(), enabled)
                .await
                .map_err(|_| ExceptionCode::IllegalDataValue)?;
            Ok(Response::WriteSingleCoil(address, enabled))
        }
        Request::WriteMultipleRegisters(address, words) => {
            write_registers(&runtime, address, &words).await?;
            Ok(Response::WriteMultipleRegisters(
                address,
                words.len() as u16,
            ))
        }
        Request::WriteSingleRegister(address, word) => {
            let words = [word];
            write_registers(&runtime, address, &words).await?;
            Ok(Response::WriteSingleRegister(address, word))
        }
        _ => Err(ExceptionCode::IllegalFunction),
    }
}

async fn write_registers(
    runtime: &RuntimeHandle,
    address: u16,
    words: &[u16],
) -> Result<(), ExceptionCode> {
    if words.len() != 2 {
        return Err(ExceptionCode::IllegalDataValue);
    }
    let command = NumericCommand::from_address(address).ok_or(ExceptionCode::IllegalDataAddress)?;
    runtime
        .write_numeric(command.id(), words_to_f32(words))
        .await
        .map_err(|_| ExceptionCode::IllegalDataValue)
}

fn read_words(words: &[u16], address: u16, count: u16) -> Result<Vec<u16>, ExceptionCode> {
    let start = usize::from(address);
    let end = start + usize::from(count);
    words
        .get(start..end)
        .map(<[u16]>::to_vec)
        .ok_or(ExceptionCode::IllegalDataAddress)
}

fn read_coils(image: &ModbusImage, address: u16, count: u16) -> Result<Vec<bool>, ExceptionCode> {
    let start = usize::from(address);
    let end = start + usize::from(count);
    image
        .coils
        .get(start..end)
        .map(<[bool]>::to_vec)
        .ok_or(ExceptionCode::IllegalDataAddress)
}

fn words_to_f32(words: &[u16]) -> f64 {
    f64::from(f32::from_bits(
        (u32::from(words[0]) << 16) | u32::from(words[1]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_decode_big_endian_floats() {
        assert_eq!(words_to_f32(&[0x3f80, 0x0000]), 1.0);
    }
}
