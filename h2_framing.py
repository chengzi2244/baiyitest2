"""HTTP/2-style framing demo (RFC 9113 section 4).

A frame is ``length:24 | type:8 | flags:8 | R:1 stream_id:31 | payload``.
``Frame`` is one message, ``FrameCodec`` encodes it and rejects illegal
layouts, and ``Endpoint`` exchanges frames on an asyncio stream. Header
blocks below are plain text, not HPACK. Run ``python h2_framing.py``.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass

PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
DATA, HEADERS, RST_STREAM = 0x0, 0x1, 0x3
SETTINGS, PUSH_PROMISE, PING = 0x4, 0x5, 0x6
GOAWAY, WINDOW_UPDATE, CONTINUATION = 0x7, 0x8, 0x9
END_STREAM, ACK, END_HEADERS = 0x1, 0x1, 0x4
_NAMES = {
    DATA: "DATA", HEADERS: "HEADERS", RST_STREAM: "RST_STREAM", SETTINGS: "SETTINGS",
    PUSH_PROMISE: "PUSH_PROMISE", PING: "PING", GOAWAY: "GOAWAY",
    WINDOW_UPDATE: "WINDOW_UPDATE", CONTINUATION: "CONTINUATION",
}
_SETTINGS = {
    1: "HEADER_TABLE_SIZE", 2: "ENABLE_PUSH", 3: "MAX_CONCURRENT_STREAMS",
    4: "INITIAL_WINDOW_SIZE", 5: "MAX_FRAME_SIZE", 6: "MAX_HEADER_LIST_SIZE",
}
_LIMITS = {2: (0, 1), 4: (0, 0x7FFFFFFF), 5: (16384, 0xFFFFFF)}
_CONN = frozenset({SETTINGS, PING, GOAWAY})
_STREAM = frozenset({DATA, HEADERS, RST_STREAM, PUSH_PROMISE, CONTINUATION})
_HEAD = struct.Struct(">IBI")


def hexdump(blob: bytes) -> str:
    """Render *blob* as address, hexadecimal, and ASCII columns."""
    rows = []
    for offset in range(0, len(blob), 16):
        chunk = blob[offset : offset + 16]
        words = " ".join(f"{byte:02x}" for byte in chunk)
        text = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
        rows.append(f"{offset:08x}  {words:<47}  |{text}|")
    return "\n".join(rows)


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame. ``stream_id`` is 31 bits; the reserved R bit is not stored.

    The class methods build the frames the demo sends. Every payload is opaque:
    this module does not compress header blocks.
    """

    type: int
    flags: int
    stream_id: int
    payload: bytes = b""

    def summary(self) -> str:
        """Name, stream, flags, and payload fields that are not an HPACK block."""
        names: list[str] = []
        if self.type in (DATA, HEADERS) and self.flags & END_STREAM:
            names.append("END_STREAM")
        if self.type in (HEADERS, CONTINUATION) and self.flags & END_HEADERS:
            names.append("END_HEADERS")
        if self.type in (SETTINGS, PING) and self.flags & ACK:
            names.append("ACK")
        return (
            f"{_NAMES.get(self.type, f'TYPE_{self.type}')} stream={self.stream_id} "
            f"len={len(self.payload)} flags={'|'.join(names) or '0'}{self._detail()}"
        )

    def _detail(self) -> str:
        payload = self.payload
        if self.type == SETTINGS and payload:
            pairs = []
            for offset in range(0, len(payload), 6):
                ident, value = struct.unpack_from(">HI", payload, offset)
                pairs.append(f"{_SETTINGS.get(ident, ident)}={value}")
            return " " + " ".join(pairs)
        if self.type == PING:
            return f" opaque={payload!r}"
        if self.type == WINDOW_UPDATE and len(payload) == 4:
            return f" increment={int.from_bytes(payload, 'big') & 0x7FFFFFFF}"
        if self.type == GOAWAY and len(payload) >= 8:
            last, code = struct.unpack_from(">II", payload)
            return f" last_stream={last & 0x7FFFFFFF} error={code} debug={payload[8:]!r}"
        return f" payload={payload!r}" if payload else ""

    @classmethod
    def settings(cls, values: tuple[tuple[int, int], ...] = (), *, ack: bool = False) -> Frame:
        """Return SETTINGS, or an empty ACK when *ack* is set."""
        if ack:
            return cls(SETTINGS, ACK, 0)
        body = b"".join(struct.pack(">HI", ident, value) for ident, value in values)
        return cls(SETTINGS, 0, 0, body)

    @classmethod
    def headers(cls, stream_id: int, block: bytes, *, end_stream: bool = False) -> Frame:
        """Return HEADERS with END_HEADERS. *block* is not compressed."""
        flags = END_HEADERS | (END_STREAM if end_stream else 0)
        return cls(HEADERS, flags, stream_id, block)

    @classmethod
    def data(cls, stream_id: int, payload: bytes, *, end_stream: bool = False) -> Frame:
        """Return DATA. *end_stream* marks the last chunk on the stream."""
        return cls(DATA, END_STREAM if end_stream else 0, stream_id, payload)

    @classmethod
    def ping(cls, opaque: bytes, *, ack: bool = False) -> Frame:
        """Return a PING. *opaque* is 8 bytes and is copied into the ACK."""
        return cls(PING, ACK if ack else 0, 0, opaque)

    @classmethod
    def window_update(cls, stream_id: int, increment: int) -> Frame:
        """Return a WINDOW_UPDATE. Stream 0 grows the connection window."""
        return cls(WINDOW_UPDATE, 0, stream_id, increment.to_bytes(4, "big"))

    @classmethod
    def goaway(cls, last_stream: int, error_code: int, debug: bytes = b"") -> Frame:
        """Return a GOAWAY. Error code 0 is NO_ERROR."""
        prefix = (last_stream & 0x7FFFFFFF).to_bytes(4, "big") + error_code.to_bytes(4, "big")
        return cls(GOAWAY, 0, 0, prefix + debug)


class FrameCodec:
    """Pack and unpack the 9-byte header.

    SETTINGS entries are 6 bytes, PING is 8, WINDOW_UPDATE and RST_STREAM are
    4, and GOAWAY and PUSH_PROMISE are at least 8 and 4. Unknown flags are
    preserved. Padding is not stripped.
    """

    def encode(self, frame: Frame) -> bytes:
        """Return the header plus the payload."""
        self.validate(frame)
        word = (len(frame.payload) << 8) | frame.type
        return _HEAD.pack(word, frame.flags, frame.stream_id) + frame.payload

    def decode(self, blob: bytes) -> Frame:
        """Parse one complete frame. A short buffer or a set R bit is an error."""
        if len(blob) < 9:
            raise ValueError("truncated frame header")
        word, flags, stream_word = _HEAD.unpack(blob[:9])
        if stream_word & 0x80000000:
            raise ValueError("reserved header bit is set")
        if len(blob) != 9 + (word >> 8):
            raise ValueError("length prefix does not match payload")
        frame = Frame(word & 0xFF, flags, stream_word, blob[9:])
        self.validate(frame)
        return frame

    def validate(self, frame: Frame) -> None:
        """Raise ``ValueError`` when *frame* cannot legally be sent."""
        if not 0 <= frame.type <= 0xFF or not 0 <= frame.flags <= 0xFF:
            raise ValueError("type or flags out of range")
        if not 0 <= frame.stream_id <= 0x7FFFFFFF or len(frame.payload) > 0xFFFFFF:
            raise ValueError("stream id or length out of range")
        payload = frame.payload
        if frame.type in _CONN and frame.stream_id:
            raise ValueError("connection frame must use stream 0")
        if frame.type in _STREAM and not frame.stream_id:
            raise ValueError("stream frame must use a non-zero stream")
        if frame.type == SETTINGS and ((frame.flags & ACK and payload) or len(payload) % 6):
            raise ValueError("SETTINGS ACK must be empty and the body a multiple of 6")
        if frame.type == SETTINGS:
            for offset in range(0, len(payload), 6):
                ident, value = struct.unpack_from(">HI", payload, offset)
                limits = _LIMITS.get(ident)
                if limits and not limits[0] <= value <= limits[1]:
                    raise ValueError(f"setting {ident} value {value} out of range")
        elif frame.type == PING and len(payload) != 8:
            raise ValueError("PING payload must be 8 bytes")
        elif frame.type == WINDOW_UPDATE:
            increment = int.from_bytes(payload, "big") if len(payload) == 4 else 0
            if len(payload) != 4 or increment == 0 or increment & 0x80000000:
                raise ValueError("WINDOW_UPDATE increment must be 1..2^31-1")
        elif frame.type == GOAWAY and (len(payload) < 8 or payload[0] & 0x80):
            raise ValueError("GOAWAY needs 8 bytes and a clear reserved bit")
        elif frame.type == RST_STREAM and len(payload) != 4:
            raise ValueError("RST_STREAM payload must be 4 bytes")
        elif frame.type == PUSH_PROMISE and len(payload) < 4:
            raise ValueError("PUSH_PROMISE payload must be at least 4 bytes")


class Endpoint:
    """One side of an asyncio connection. The client writes the 24-byte preface."""

    def __init__(
        self,
        name: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.name, self.reader, self.writer, self.codec = name, reader, writer, FrameCodec()

    def _show(self, arrow: str, detail: str, blob: bytes, headed: bool) -> None:
        print(f"\n[{self.name} {arrow}] {detail}")
        if headed:
            stream = int.from_bytes(blob[5:9], "big")
            print(
                f"  header={blob[:9].hex(' ')}  length={int.from_bytes(blob[:3], 'big')} "
                f"type={blob[3]} flags={blob[4]} R={stream >> 31} stream={stream & 0x7FFFFFFF}"
            )
        print(hexdump(blob))

    async def write_preface(self) -> None:
        """Send ``PRI * HTTP/2.0`` followed by the two blank lines and ``SM``."""
        self._show(">>", "connection preface", PREFACE, False)
        self.writer.write(PREFACE)
        await self.writer.drain()

    async def read_preface(self) -> None:
        """Read the preface and require the standard 24 bytes."""
        blob = await self.reader.readexactly(len(PREFACE))
        if blob != PREFACE:
            raise ValueError("connection preface mismatch")
        self._show("<<", "connection preface", blob, False)

    async def send(self, frame: Frame) -> None:
        """Encode *frame*, require a matching decode, and write the bytes."""
        blob = self.codec.encode(frame)
        if self.codec.decode(blob) != frame:
            raise ValueError("encode/decode round-trip failed")
        self._show(">>", frame.summary(), blob, True)
        self.writer.write(blob)
        await self.writer.drain()

    async def recv(self) -> Frame:
        """Read one frame. The 24-bit length is the payload size."""
        header = await self.reader.readexactly(9)
        payload = await self.reader.readexactly(int.from_bytes(header[:3], "big"))
        frame = self.codec.decode(header + payload)
        self._show("<<", frame.summary(), header + payload, True)
        return frame

    async def aclose(self) -> None:
        """Close the stream."""
        self.writer.close()
        await self.writer.wait_closed()


def _self_check() -> None:
    """Round-trip every frame type, then reject layouts a peer must not send."""
    codec = FrameCodec()
    good = (
        Frame.settings(((1, 4096), (2, 0), (5, 16384))), Frame.settings(ack=True),
        Frame.headers(1, b":method: GET\n"), Frame.data(1, b"hello", end_stream=True),
        Frame(RST_STREAM, 0, 1, (1).to_bytes(4, "big")),
        Frame(PUSH_PROMISE, END_HEADERS, 1, (2).to_bytes(4, "big") + b"b"),
        Frame.ping(b"PINGPONG"), Frame.goaway(1, 0, b"done"),
        Frame.window_update(0, 65535), Frame(CONTINUATION, END_HEADERS, 1, b"more"),
    )
    bad = (
        Frame.window_update(0, 0), Frame.data(0, b"x"), Frame.ping(b"short"),
        Frame(SETTINGS, ACK, 0, b"123456"), Frame(GOAWAY, 0, 0, b"short"),
        Frame(PING, 0, 3, b"PINGPONG"), Frame(PUSH_PROMISE, 0, 1, b"xx"),
        Frame(SETTINGS, 0, 0, b"abc"),
        Frame(SETTINGS, 0, 0, struct.pack(">HI", 5, 100)),
        Frame(0x100, 0, 1, b""),
        Frame(DATA, 0x100, 1, b"x"),
    )
    for frame in good:
        if codec.decode(codec.encode(frame)) != frame:
            raise AssertionError(frame.summary())
    for frame in bad:
        try:
            codec.encode(frame)
        except ValueError:
            continue
        raise AssertionError(frame.summary())
    raw = bytearray(codec.encode(Frame.ping(b"PINGPONG")))
    raw[5] |= 0x80
    for blob in (bytes(raw), bytes(raw) + b"\x00", bytes(raw[:8])):
        try:
            codec.decode(blob)
        except ValueError:
            continue
        raise AssertionError("malformed frame was accepted")


async def _exchange(
    port: int,
    ready: asyncio.Event,
    answered: asyncio.Event,
    closing: asyncio.Event,
) -> None:
    """Send a request and read the response. Used as the loopback client."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    peer = Endpoint("client", reader, writer)
    try:
        print("\n=== client writes the preface and request ===")
        await peer.write_preface()
        request = (
            Frame.settings(((1, 4096), (2, 0), (4, 65535), (5, 16384))),
            Frame.headers(1, b":method: GET\n:path: /\n:scheme: http\n"),
            Frame.data(1, b"hello", end_stream=True),
            Frame.ping(b"PINGPONG"),
        )
        for frame in request:
            await peer.send(frame)
        ready.set()
        await answered.wait()
        print("\n=== client reads the response ===")
        ack, window, headers, body, pong = [await peer.recv() for _ in range(5)]
        if (ack.type, window.payload, body.payload, pong.payload) != (
            SETTINGS, (65535).to_bytes(4, "big"), b"hello h2\n", b"PINGPONG",
        ):
            raise AssertionError("response payload mismatch")
        if not (ack.flags & ACK and headers.flags & END_HEADERS and pong.flags & ACK):
            raise AssertionError("response flags mismatch")
        print("\n=== client sends GOAWAY ===")
        await peer.send(Frame.goaway(1, 0, b"done"))
        closing.set()
        if (await peer.recv()).payload[8:] != b"ok":
            raise AssertionError("server GOAWAY mismatch")
    finally:
        await peer.aclose()


async def _serve(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    ready: asyncio.Event,
    answered: asyncio.Event,
    closing: asyncio.Event,
) -> None:
    """Read one request, answer it, and answer GOAWAY. Used as the loopback server."""
    peer = Endpoint("server", reader, writer)
    try:
        await ready.wait()
        print("\n=== server reads the preface and request ===")
        await peer.read_preface()
        settings, headers, data, ping = [await peer.recv() for _ in range(4)]
        if (data.payload, ping.payload) != (b"hello", b"PINGPONG"):
            raise AssertionError("unexpected request")
        request_ok = (
            settings.type == SETTINGS and headers.flags & END_HEADERS and data.flags & END_STREAM
        )
        if not request_ok:
            raise AssertionError("request flags mismatch")
        print("\n=== server writes the response ===")
        reply = (
            Frame.settings(ack=True),
            Frame.window_update(0, 65535),
            Frame.headers(1, b":status: 200\ncontent-type: text/plain\n"),
            Frame.data(1, b"hello h2\n", end_stream=True),
            Frame.ping(ping.payload, ack=True),
        )
        for frame in reply:
            await peer.send(frame)
        answered.set()
        await closing.wait()
        print("\n=== server reads GOAWAY and replies ===")
        if (await peer.recv()).payload[8:] != b"done":
            raise AssertionError("client GOAWAY mismatch")
        await peer.send(Frame.goaway(1, 0, b"ok"))
    finally:
        await peer.aclose()


async def main() -> None:
    """Check the codec, then print both sides of one loopback exchange."""
    print("HTTP/2-style framing: length:24 | type:8 | flags:8 | R:1 stream_id:31 | payload")
    _self_check()
    print("self-check passed: every frame type round-trips, illegal layouts are rejected")
    ready, answered, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _serve(reader, writer, ready, answered, closing)
        except Exception as exc:
            if not finished.done():
                finished.set_exception(exc)
        else:
            if not finished.done():
                finished.set_result(None)

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    if not server.sockets:
        raise RuntimeError("listener failed to bind")
    port = server.sockets[0].getsockname()[1]
    async with server:
        await asyncio.wait_for(
            asyncio.gather(_exchange(port, ready, answered, closing), finished),
            5,
        )
    print("\nexchange complete")


if __name__ == "__main__":
    asyncio.run(main())
