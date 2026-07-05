"""Deterministic classic-PCAP export for scenario profiles."""

from __future__ import annotations

import ipaddress
import struct
from pathlib import Path

from .models import ModbusTransaction, NetworkNode, ScenarioProfile

ETHERTYPE_IPV4 = 0x0800
LINKTYPE_ETHERNET = 1
MODBUS_UNIT_ID = 1


def _checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for index in range(0, len(data), 2):
        total += (data[index] << 8) + data[index + 1]
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _mac_bytes(mac: str) -> bytes:
    parts = mac.split(":")
    if len(parts) != 6:
        raise ValueError(f"invalid MAC address: {mac}")
    return bytes(int(part, 16) for part in parts)


def _ipv4_bytes(ipv4: str) -> bytes:
    return ipaddress.IPv4Address(ipv4).packed


def _mbap(transaction: ModbusTransaction, pdu: bytes) -> bytes:
    length = 1 + len(pdu)
    return struct.pack(">HHHB", transaction.ordinal, 0, length, MODBUS_UNIT_ID) + pdu


def _request_pdu(transaction: ModbusTransaction) -> bytes:
    fc = transaction.function_code
    if fc in (3, 4):
        return struct.pack(">BHH", fc, transaction.address, transaction.quantity)
    if fc == 5:
        value = 0xFF00 if transaction.wire_values[0] else 0x0000
        return struct.pack(">BHH", fc, transaction.address, value)
    if fc == 6:
        return struct.pack(">BHH", fc, transaction.address, transaction.wire_values[0])
    if fc == 16:
        values = bytes().join(
            struct.pack(">H", value) for value in transaction.wire_values
        )
        return (
            struct.pack(
                ">BHHB", fc, transaction.address, transaction.quantity, len(values)
            )
            + values
        )
    raise ValueError(f"unsupported Modbus function code: {fc}")


def _response_pdu(transaction: ModbusTransaction) -> bytes:
    fc = transaction.function_code
    if transaction.response.startswith("exception_0x"):
        code = int(transaction.response.removeprefix("exception_0x"), 16)
        return struct.pack(">BB", fc | 0x80, code)
    if fc in (3, 4):
        byte_count = transaction.quantity * 2
        return struct.pack(">BB", fc, byte_count) + bytes(byte_count)
    if fc in (5, 6):
        return _request_pdu(transaction)
    if fc == 16:
        return struct.pack(">BHH", fc, transaction.address, transaction.quantity)
    raise ValueError(f"unsupported Modbus function code: {fc}")


def _ipv4_packet(
    source_ip: bytes,
    dest_ip: bytes,
    payload: bytes,
    packet_id: int,
) -> bytes:
    version_ihl = 0x45
    total_length = 20 + len(payload)
    header = struct.pack(
        ">BBHHHBBH4s4s",
        version_ihl,
        0,
        total_length,
        packet_id & 0xFFFF,
        0x4000,
        64,
        6,
        0,
        source_ip,
        dest_ip,
    )
    checksum = _checksum(header)
    return header[:10] + struct.pack(">H", checksum) + header[12:] + payload


def _tcp_segment(
    source_ip: bytes,
    dest_ip: bytes,
    source_port: int,
    dest_port: int,
    payload: bytes,
    seq: int,
    ack: int,
) -> bytes:
    offset_flags = (5 << 12) | 0x018
    header = struct.pack(
        ">HHIIHHHH",
        source_port,
        dest_port,
        seq,
        ack,
        offset_flags,
        8192,
        0,
        0,
    )
    pseudo = source_ip + dest_ip + struct.pack(">BBH", 0, 6, len(header) + len(payload))
    checksum = _checksum(pseudo + header + payload)
    return header[:16] + struct.pack(">H", checksum) + header[18:] + payload


def _ethernet_frame(
    source: NetworkNode, dest: NetworkNode, ipv4_packet: bytes
) -> bytes:
    return (
        _mac_bytes(dest.mac)
        + _mac_bytes(source.mac)
        + struct.pack(">H", ETHERTYPE_IPV4)
        + ipv4_packet
    )


def _frame(
    transaction: ModbusTransaction,
    source: NetworkNode,
    dest: NetworkNode,
    source_port: int,
    dest_port: int,
    payload: bytes,
    packet_id: int,
    seq: int,
    ack: int,
) -> bytes:
    source_ip = _ipv4_bytes(source.ipv4)
    dest_ip = _ipv4_bytes(dest.ipv4)
    tcp = _tcp_segment(source_ip, dest_ip, source_port, dest_port, payload, seq, ack)
    ipv4 = _ipv4_packet(source_ip, dest_ip, tcp, packet_id)
    return _ethernet_frame(source, dest, ipv4)


def _pcap_record(timestamp_us: int, frame: bytes) -> bytes:
    seconds = timestamp_us // 1_000_000
    micros = timestamp_us % 1_000_000
    header = struct.pack("<IIII", seconds, micros, len(frame), len(frame))
    return header + frame


def render_pcap_bytes(scenario: ScenarioProfile) -> bytes:
    nodes = {node.node_id: node for node in scenario.nodes}
    output = [struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET)]
    for transaction in scenario.transactions:
        actor = nodes[transaction.actor_id]
        server = nodes[transaction.server_id]
        server_port = server.modbus_port or 502
        client_port = 40_000 + transaction.ordinal
        request = _mbap(transaction, _request_pdu(transaction))
        response = _mbap(transaction, _response_pdu(transaction))
        request_ts = transaction.timestamp_ms * 1000
        response_ts = request_ts + 50_000
        request_seq = transaction.ordinal * 10_000
        response_seq = transaction.ordinal * 20_000
        request_frame = _frame(
            transaction,
            actor,
            server,
            client_port,
            server_port,
            request,
            transaction.ordinal * 2 - 1,
            request_seq,
            response_seq,
        )
        response_frame = _frame(
            transaction,
            server,
            actor,
            server_port,
            client_port,
            response,
            transaction.ordinal * 2,
            response_seq,
            request_seq + len(request),
        )
        output.append(_pcap_record(request_ts, request_frame))
        output.append(_pcap_record(response_ts, response_frame))
    return b"".join(output)


def write_pcap(path: str | Path, scenario: ScenarioProfile) -> None:
    Path(path).write_bytes(render_pcap_bytes(scenario))
