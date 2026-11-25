mod commands;

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};
use std::os::unix::io::AsRawFd;
use std::time;

use async_shutdown::ShutdownManager;
use bytes::{Buf, BytesMut};
use mio::net::UnixListener;
use pulseaudio::protocol::{self as pulse};

use super::buffer::PlaybackBuffer;
use super::{AudioChannels, SINK_SAMPLE_RATE, SinkSpec, log};
use crate::session::manager::SessionShutdownReason;

type Error = Box<dyn std::error::Error + Send + Sync>;

const LISTENER: mio::Token = mio::Token(0);
const CLOCK: mio::Token = mio::Token(1);

/// Cap on per-client pending outgoing data. A buffer exceeding this means the
/// client has stopped reading; the stuck client is dropped rather than letting
/// the buffer grow unboundedly, and the session survives.
const MAX_OUTGOING_BUFFER: usize = 64 * 1024 * 1024;

/// Clock tick rate. Determines audio frame size sent to the encoder.
/// For 5ms frames: 200 Hz; for 10ms frames: 100 Hz.
pub(crate) const DEFAULT_CLOCK_RATE_HZ: u32 = 200;

const SINK_NAME: &str = "moonshine";

const MAX_SINK_CHANNELS: usize = AudioChannels::Surround71 as usize;
const ZERO_VOL: [f32; MAX_SINK_CHANNELS] = [0.0; MAX_SINK_CHANNELS];

/// A buffer of interleaved f32 samples ready for Opus encoding.
pub(crate) struct AudioFrame {
	/// Interleaved f32 samples for the negotiated channel count.
	pub buf: Vec<f32>,

	/// Capture timestamp in milliseconds since process start.
	pub capture_ts_ms: u64,
}

/// One media generation's capture destination and frame layout.
pub(super) struct CaptureOutput {
	pub channels: AudioChannels,
	pub packet_duration_ms: u32,
	pub frame_tx: crossbeam_channel::Sender<AudioFrame>,
	pub frame_recycle_rx: crossbeam_channel::Receiver<AudioFrame>,
	pub stop: ShutdownManager<SessionShutdownReason>,
}

impl CaptureOutput {
	fn clock_rate_hz(&self) -> u32 {
		match self.packet_duration_ms {
			5 | 10 => 1000 / self.packet_duration_ms,
			0 => DEFAULT_CLOCK_RATE_HZ,
			duration => {
				tracing::warn!(
					"Unsupported packet_duration_ms {duration}, falling back to default {DEFAULT_CLOCK_RATE_HZ}Hz"
				);
				DEFAULT_CLOCK_RATE_HZ
			},
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
	Prebuffering(u64),
	Corked,
	Playing,
	Draining(u32),
}

struct PlaybackStream {
	stream_index: u32,
	state: StreamState,
	buffer_attr: pulse::stream::BufferAttr,
	buffer: PlaybackBuffer,
	volume: [f32; MAX_SINK_CHANNELS],
	muted: bool,
	/// Bytes consumed by the audio clock since the last REQUEST was sent.
	/// Grows on every successful drain. Zeroed atomically when a REQUEST is issued.
	/// Mirrors `pa_memblockq::missing` (signed to allow temporary negative values
	/// from flush/seek, though in practice it never goes negative here).
	missing: i64,

	/// Bytes we have sent a REQUEST for but have not yet received from the client.
	/// Grows when a REQUEST is sent. Shrinks when the client writes data.
	/// Mirrors `pa_memblockq::requested`.
	requested: usize,
	played_bytes: u64,
	write_offset: u64,
	read_offset: u64,
	opened_at: time::Instant,
}

struct Client {
	id: u32,
	socket: mio::net::UnixStream,
	protocol_version: u16,
	props: Option<pulse::Props>,
	incoming: BytesMut,
	outgoing: BytesMut,
	writable_registered: bool,
	playback_streams: BTreeMap<u32, PlaybackStream>,
}

/// Outcome of attempting to flush a client's outgoing buffer.
enum FlushResult {
	/// Buffer fully drained; socket is idle.
	Done,
	/// Socket send buffer is full; retry when the socket becomes writable.
	Blocked,
	/// Socket write failed (peer gone, etc.); the client should be dropped.
	Dead,
}

impl Client {
	fn stream_owner(&self, channel: u32) -> log::StreamOwner<'_> {
		log::StreamOwner {
			client_id: self.id,
			client_props: self.props.as_ref(),
			channel,
		}
	}

	fn log_stream_closed(&self, channel: u32, stream: &PlaybackStream, reason: &str) {
		log::pulse_stream_closed(
			self.stream_owner(channel),
			reason,
			stream.buffer.stream_spec(),
			stream.opened_at.elapsed(),
			stream.played_bytes,
		);
	}

	/// Write as much of the outgoing buffer to the socket as possible.
	///
	/// `WouldBlock` is not an error: the remainder stays buffered and the
	/// caller arms `WRITABLE` interest. A real write error only ever kills the
	/// offending client, never the server.
	fn flush(&mut self) -> FlushResult {
		loop {
			if self.outgoing.is_empty() {
				return FlushResult::Done;
			}
			match self.socket.write(&self.outgoing) {
				Ok(0) => return FlushResult::Dead,
				Ok(n) => self.outgoing.advance(n),
				Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return FlushResult::Blocked,
				Err(_) => return FlushResult::Dead,
			}
		}
	}
}

/// Adapter that lets the `pulseaudio` protocol writers append directly into a
/// client's outgoing buffer instead of hitting a nonblocking socket.
struct ClientWriter<'a>(&'a mut BytesMut);

impl std::io::Write for ClientWriter<'_> {
	fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
		self.0.extend_from_slice(buf);
		Ok(buf.len())
	}

	fn flush(&mut self) -> std::io::Result<()> {
		Ok(())
	}
}

struct ServerState {
	server_info: pulse::ServerInfo,
	clock_rate_hz: u32,
	sinks: Vec<pulse::SinkInfo>,
	default_format_info: pulse::FormatInfo,
	next_playback_channel_index: u32,
	next_stream_index: u32,
	sink_volume: [f32; MAX_SINK_CHANNELS],
	sink_muted: bool,
	sink_spec: SinkSpec,
}

impl ServerState {
	fn apply_sink_layout(&mut self) {
		let channels = self.sink_spec.channels;
		let sample_spec = self.sink_spec.as_pulse_spec();
		self.server_info.sample_spec = sample_spec;
		self.server_info.channel_map = channels.map();

		let sink = &mut self.sinks[0];
		sink.sample_spec = sample_spec;
		sink.channel_map = channels.map();
		sink.cvolume = commands::linear_to_cvolume(&self.sink_volume[..channels as usize]);
		sink.ports[0].name = channels.port_as_cstr().into();
		sink.formats[0] = self.default_format_info.clone();
	}
}

pub(crate) struct PulseServer {
	listener: UnixListener,
	poll: mio::Poll,
	clock: mio_timerfd::TimerFd,

	output: CaptureOutput,
	output_rx: crossbeam_channel::Receiver<CaptureOutput>,
	spare_frame: Option<AudioFrame>,

	clients: BTreeMap<mio::Token, Client>,
	server_state: ServerState,

	epoch: time::Instant,
}

/// Mirrors `pa_memblockq_pop_missing`.
///
/// Transfers accumulated demand (`missing`) into in-flight credit (`requested`)
/// and returns the number of bytes to include in the next REQUEST message.
/// Returns 0 if there is nothing to request yet.
///
/// The `in_prebuf` flag bypasses the `min_req` gate — when re-filling after
/// an underrun the server should not wait for a full `min_req` chunk to
/// accumulate before asking; any positive demand should be requested immediately.
fn pop_missing(missing: &mut i64, requested: &mut usize, min_req: usize, in_prebuf: bool) -> usize {
	if *missing <= 0 {
		return 0;
	}
	if (*missing as usize) < min_req && !in_prebuf {
		return 0;
	}
	let l = *missing as usize;
	*requested += l;
	*missing = 0;
	l
}

fn sink_format_info(channels: AudioChannels) -> pulse::FormatInfo {
	let mut props = pulse::Props::new();
	props.set(pulse::Prop::FormatChannels, channels.count_as_cstr());
	props.set(pulse::Prop::FormatChannelMap, channels.map_cstr());
	props.set(pulse::Prop::FormatSampleFormat, c"float32le");
	props.set(
		pulse::Prop::FormatRate,
		std::ffi::CString::new(SINK_SAMPLE_RATE.to_string()).unwrap(),
	);
	pulse::FormatInfo {
		encoding: pulse::FormatEncoding::Pcm,
		props,
	}
}

impl PulseServer {
	pub(super) fn spawn(
		listener: std::os::unix::net::UnixListener,
		output: CaptureOutput,
		stop: ShutdownManager<SessionShutdownReason>,
	) -> Result<crossbeam_channel::Sender<CaptureOutput>, Error> {
		listener.set_nonblocking(true)?;
		let listener = UnixListener::from_std(listener);
		let poll = mio::Poll::new()?;

		let clock_rate_hz = output.clock_rate_hz();
		let sink_channels = output.channels;
		let mut clock = mio_timerfd::TimerFd::new(mio_timerfd::ClockId::Monotonic)?;
		clock.set_timeout_interval(&time::Duration::from_nanos(1_000_000_000 / clock_rate_hz as u64))?;

		let sink_name = std::ffi::CString::new(SINK_NAME).unwrap();

		let (output_tx, output_rx) = crossbeam_channel::unbounded();

		let mut dummy_sink = pulse::SinkInfo::new_dummy(1);
		dummy_sink.name = sink_name.clone();
		dummy_sink.description = Some(std::ffi::CString::new("Moonshine virtual output").unwrap());
		dummy_sink.base_volume = pulse::Volume::NORM;

		let server_info = pulse::ServerInfo {
			server_name: Some(std::ffi::CString::new("Moonshine").unwrap()),
			server_version: Some(std::ffi::CString::new(env!("CARGO_PKG_VERSION")).unwrap()),
			host_name: Some(std::ffi::CString::new("moonshine").unwrap()),
			default_sink_name: Some(sink_name),
			default_source_name: Some(std::ffi::CString::new("").unwrap()),
			..Default::default()
		};

		dummy_sink.ports[0].port_type = pulse::port_info::PortType::Network;
		dummy_sink.ports[0].description = Some(std::ffi::CString::new("virtual output").unwrap());

		let mut server_state = ServerState {
			server_info,
			clock_rate_hz,
			sinks: vec![dummy_sink],
			default_format_info: sink_format_info(sink_channels),
			next_playback_channel_index: 0,
			next_stream_index: 0,
			sink_volume: [1.0; MAX_SINK_CHANNELS],
			sink_muted: false,
			sink_spec: SinkSpec::new(sink_channels),
		};
		server_state.apply_sink_layout();

		let server = Self {
			listener,
			poll,
			clock,
			output,
			output_rx,
			spare_frame: None,
			clients: BTreeMap::new(),
			server_state,
			epoch: time::Instant::now(),
		};

		let delay_stop = stop.delay_shutdown_token()?;
		std::thread::Builder::new()
			.name("pulse-server".to_string())
			.spawn(move || {
				let _delay_stop = delay_stop;
				let _stop_token = stop.trigger_shutdown_token(SessionShutdownReason::PulseServerStopped);
				if let Err(e) = server.run(stop) {
					tracing::error!("PulseServer error: {e}");
				}
			})?;

		Ok(output_tx)
	}

	fn run(mut self, stop: ShutdownManager<SessionShutdownReason>) -> Result<(), Error> {
		let mut next_client_token = 1024u64;

		self.poll.registry().register(
			&mut mio::unix::SourceFd(&self.clock.as_raw_fd()),
			CLOCK,
			mio::Interest::READABLE,
		)?;
		self.poll
			.registry()
			.register(&mut self.listener, LISTENER, mio::Interest::READABLE)?;

		let mut events = mio::Events::with_capacity(1024);

		loop {
			match self.poll.poll(&mut events, Some(time::Duration::from_secs(1))) {
				Ok(_) => (),
				Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
				Err(e) => return Err(e.into()),
			}

			if stop.is_shutdown_triggered() {
				return Ok(());
			}

			for event in events.iter() {
				match event.token() {
					CLOCK => {
						// A wakeup can race its own drain and find no expirations
						// left to read (EAGAIN); skip the tick, don't kill the session.
						match self.clock.read() {
							Ok(_) => self.clock_tick()?,
							Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => (),
							Err(e) => return Err(e.into()),
						}
					},
					LISTENER => loop {
						let (mut socket, _) = match self.listener.accept() {
							Ok(conn) => conn,
							Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
							Err(e) => return Err(e.into()),
						};
						let id = next_client_token as u32;
						let token = mio::Token(next_client_token as usize);
						next_client_token += 1;

						tracing::debug!("PulseAudio client connected (id={})", id);

						self.poll
							.registry()
							.register(&mut socket, token, mio::Interest::READABLE)?;

						self.clients.insert(
							token,
							Client {
								id,
								socket,
								protocol_version: pulse::MAX_VERSION,
								props: None,
								incoming: BytesMut::new(),
								outgoing: BytesMut::new(),
								writable_registered: false,
								playback_streams: BTreeMap::new(),
							},
						);
					},
					client_token if event.is_read_closed() => {
						self.remove_client(client_token);
					},
					client_token if event.is_readable() && self.clients.contains_key(&client_token) => {
						if let Err(e) = self.recv(client_token) {
							tracing::error!("PulseAudio client error: {:#}", e);
							self.remove_client(client_token);
						}
					},
					client_token if event.is_writable() => {
						self.flush_client(client_token);
					},
					_ => (),
				}
			}
		}
	}

	fn recv(&mut self, client_token: mio::Token) -> Result<(), Error> {
		let result = (|| -> Result<(), Error> {
			let client = self.clients.get_mut(&client_token).unwrap();

			let mut read_size = 8192;

			'read: loop {
				let off = client.incoming.len();
				client.incoming.resize(off + read_size, 0);
				let n = match client.socket.read(&mut client.incoming[off..]) {
					Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
						client.incoming.truncate(off);
						return Ok(());
					},
					v => v.map_err(|e| -> Error { format!("recv error: {e}").into() })?,
				};

				client.incoming.truncate(off + n);

				loop {
					if client.incoming.len() < pulse::DESCRIPTOR_SIZE {
						read_size = 8192;
						continue 'read;
					}

					let desc = pulse::read_descriptor(&mut Cursor::new(&client.incoming[..pulse::DESCRIPTOR_SIZE]))?;

					// Guard against excessively large payloads (max 4 MiB).
					const MAX_PAYLOAD_SIZE: u32 = 4 * 1024 * 1024;
					if desc.length > MAX_PAYLOAD_SIZE {
						return Err(format!("payload too large: {} bytes", desc.length).into());
					}

					if client.incoming.len() < (desc.length as usize + pulse::DESCRIPTOR_SIZE) {
						read_size = desc.length as usize + pulse::DESCRIPTOR_SIZE - client.incoming.len();
						continue 'read;
					}

					let _desc_bytes = client.incoming.split_to(pulse::DESCRIPTOR_SIZE);
					let payload = client.incoming.split_to(desc.length as usize).freeze();

					if desc.channel == u32::MAX {
						let (seq, cmd) =
							match pulse::Command::read_tag_prefixed(&mut Cursor::new(payload), client.protocol_version)
							{
								Err(pulse::ProtocolError::Unimplemented(seq, cmd)) => {
									tracing::warn!("Unimplemented PA command: {:?}", cmd);
									pulse::write_error(
										&mut ClientWriter(&mut client.outgoing),
										seq,
										&pulse::PulseError::NotImplemented,
									)?;
									continue;
								},
								v => v.map_err(|e| -> Error { format!("decoding command: {e}").into() })?,
							};

						match commands::handle_command(client, &mut self.server_state, seq, cmd) {
							Ok(()) => (),
							Err(e) => {
								let _ = pulse::write_error(
									&mut ClientWriter(&mut client.outgoing),
									seq,
									&pulse::PulseError::Internal,
								);
								return Err(e);
							},
						}
					} else {
						commands::handle_stream_write(client, desc, &payload)?;
					}
				}
			}
		})();

		if result.is_ok() {
			self.flush_client(client_token);
		}

		result
	}

	/// Flush a client's pending outgoing data, managing `WRITABLE` interest and
	/// dropping clients that are stuck or whose socket has failed.
	fn flush_client(&mut self, client_token: mio::Token) {
		let remove = {
			let client = match self.clients.get_mut(&client_token) {
				Some(client) => client,
				None => return,
			};

			if client.outgoing.len() > MAX_OUTGOING_BUFFER {
				tracing::warn!(
					"PulseAudio client {} is stuck (outgoing buffer {} B exceeds cap), dropping",
					client.id,
					MAX_OUTGOING_BUFFER,
				);
				true
			} else {
				match client.flush() {
					FlushResult::Done => {
						if client.writable_registered {
							client.writable_registered = false;
							let _ = self.poll.registry().reregister(
								&mut client.socket,
								client_token,
								mio::Interest::READABLE,
							);
						}
						false
					},
					FlushResult::Blocked => {
						if !client.writable_registered {
							client.writable_registered = true;
							let _ = self.poll.registry().reregister(
								&mut client.socket,
								client_token,
								mio::Interest::READABLE | mio::Interest::WRITABLE,
							);
						}
						false
					},
					FlushResult::Dead => true,
				}
			}
		};

		if remove {
			self.remove_client(client_token);
		}
	}

	fn remove_client(&mut self, client_token: mio::Token) {
		if let Some(mut client) = self.clients.remove(&client_token) {
			tracing::debug!("PulseAudio client disconnected (id={})", client.id);
			for (channel, stream) in &client.playback_streams {
				client.log_stream_closed(*channel, stream, "client disconnected");
			}
			let _ = self.poll.registry().deregister(&mut client.socket);
		}
	}

	fn reconfigure_output(&mut self, output: CaptureOutput) -> Result<(), Error> {
		let clock_rate_hz = output.clock_rate_hz();
		if self.server_state.clock_rate_hz != clock_rate_hz {
			self.clock
				.set_timeout_interval(&time::Duration::from_nanos(1_000_000_000 / clock_rate_hz as u64))?;
			self.server_state.clock_rate_hz = clock_rate_hz;
		}
		if self.server_state.sink_spec.channels != output.channels {
			self.server_state.sink_spec.channels = output.channels;
			self.server_state.default_format_info = sink_format_info(output.channels);
			self.server_state.apply_sink_layout();
			for client in self.clients.values_mut() {
				for stream in client.playback_streams.values_mut() {
					stream.buffer.set_sink_channels(output.channels);
				}
			}
		}
		self.output = output;
		Ok(())
	}

	fn clock_tick(&mut self) -> Result<(), Error> {
		// Only the Pulse thread switches output, at a clock boundary. The
		// clients, byte queues, request credits and stream state stay in place.
		if let Some(output) = self.output_rx.try_iter().last() {
			self.reconfigure_output(output)?;
		}
		if self.output.stop.is_shutdown_triggered() {
			return Ok(());
		}

		let mut done_draining = Vec::new();

		let capture_ts = self.epoch.elapsed().as_millis() as u64;
		let sink_channels = self.server_state.sink_spec.channels as usize;
		let num_frames = self.server_state.sink_spec.sample_rate / self.server_state.clock_rate_hz;
		let encode_len = num_frames as usize * sink_channels;

		let mut frame = match self.output.frame_recycle_rx.try_recv() {
			Ok(mut frame) => {
				frame.buf.resize(encode_len, 0.0);
				frame.buf.fill(0.0);
				frame
			},
			Err(crossbeam_channel::TryRecvError::Empty) => {
				if let Some(mut frame) = self.spare_frame.take() {
					frame.buf.resize(encode_len, 0.0);
					frame.buf.fill(0.0);
					frame
				} else {
					// Recycle pool temporarily exhausted; allocate a fresh frame rather
					// than waiting while the encoder may itself be waiting for audio.
					AudioFrame {
						buf: vec![0.0; encode_len],
						capture_ts_ms: 0,
					}
				}
			},
			Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(()),
		};

		for client in self.clients.values_mut() {
			done_draining.clear();
			for (id, stream) in client.playback_streams.iter_mut() {
				// ── Drain phase ─────────────────────────────────────────────────────
				// Only Playing and Draining streams consume audio data each tick.
				// Prebuffering and Corked streams do not drain.
				if matches!(stream.state, StreamState::Playing | StreamState::Draining(_)) {
					let buffer_len = stream.buffer.len_bytes();

					let drained = if stream.muted {
						stream
							.buffer
							.drain_and_mix(num_frames as usize, &mut frame.buf, &ZERO_VOL[..sink_channels])
					} else {
						stream.buffer.drain_and_mix(
							num_frames as usize,
							&mut frame.buf,
							&stream.volume[..sink_channels],
						)
					};

					if !drained {
						// Buffer underrun: notify client and (re-)enter prebuffering.
						tracing::warn!("Buffer underrun for stream {}", id);
						pulse::write_command_message(
							&mut ClientWriter(&mut client.outgoing),
							u32::MAX,
							&pulse::Command::Underflow(pulse::Underflow {
								channel: *id,
								offset: 0,
							}),
							client.protocol_version,
						)?;

						if stream.buffer_attr.pre_buffering > 0 && matches!(stream.state, StreamState::Playing) {
							stream.state = StreamState::Prebuffering(stream.buffer_attr.pre_buffering as u64);
							// Seed missing = pre_buffering so pop_missing fires immediately
							// in the REQUEST phase below (same tick, no one-tick delay).
							// Reset requested — any stale in-flight credit is discarded.
							stream.missing = stream.buffer_attr.pre_buffering as i64;
							stream.requested = 0;
						}
						// Fall through to REQUEST phase — do NOT continue.
					} else {
						// Successful drain: accumulate demand.
						// Mirrors pa_memblockq::read_index_changed → missing += delta.
						let read_len = buffer_len - stream.buffer.len_bytes();
						stream.read_offset += read_len as u64;
						stream.played_bytes += read_len as u64;
						stream.missing += read_len as i64;

						if matches!(stream.state, StreamState::Draining(_)) && stream.buffer.is_empty() {
							done_draining.push(*id);
						}
					}
				}

				// ── Request phase ────────────────────────────────────────────────────
				// Issue a REQUEST if enough demand has accumulated.
				// Draining streams are excluded — they are emptying, not refilling.
				if matches!(
					stream.state,
					StreamState::Playing | StreamState::Corked | StreamState::Prebuffering(_)
				) {
					let min_req = stream.buffer_attr.minimum_request_length as usize;
					let in_prebuf = matches!(stream.state, StreamState::Prebuffering(_));
					let req = pop_missing(&mut stream.missing, &mut stream.requested, min_req, in_prebuf);
					if req > 0 {
						pulse::write_command_message(
							&mut ClientWriter(&mut client.outgoing),
							u32::MAX,
							&pulse::Command::Request(pulse::Request {
								channel: *id,
								length: req as u32,
							}),
							client.protocol_version,
						)?;
					}
				}
			}

			for id in done_draining.iter() {
				let stream = client.playback_streams.remove(id).unwrap();
				client.log_stream_closed(*id, &stream, "drained");
				if let StreamState::Draining(drain_seq) = stream.state {
					pulse::write_ack_message(&mut ClientWriter(&mut client.outgoing), drain_seq)?;
				}
			}
		}

		// Flush pending output for all clients. WouldBlock is handled gracefully
		// (buffered + WRITABLE interest); a stuck client is dropped, never the
		// whole server.
		let tokens: Vec<mio::Token> = self.clients.keys().copied().collect();
		for token in tokens {
			self.flush_client(token);
		}

		// Apply sink-level volume.
		if self.server_state.sink_muted {
			frame.buf.fill(0.0);
		} else if self.server_state.sink_volume[..sink_channels].iter().any(|&v| v != 1.0) {
			let vol = &self.server_state.sink_volume[..sink_channels];
			for (i, sample) in frame.buf.iter_mut().enumerate() {
				*sample *= vol[i % vol.len()];
			}
		}

		frame.capture_ts_ms = capture_ts;
		match self.output.frame_tx.try_send(frame) {
			Ok(()) => {},
			Err(crossbeam_channel::TrySendError::Full(frame)) => {
				// Encoder is behind; stash the frame so we don't lose the allocation.
				self.spare_frame = Some(frame);
			},
			Err(crossbeam_channel::TrySendError::Disconnected(_)) => return Ok(()),
		}

		Ok(())
	}
}
