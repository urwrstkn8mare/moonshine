use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use async_shutdown::{DelayShutdownToken, ShutdownManager};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, watch};

use crate::gpu::GpuContext;
use crate::session::SessionKeysReceiver;
use crate::session::compositor::CompositorFrames;
use crate::session::compositor::frame::HdrModeState;
use crate::session::manager::SessionShutdownReason;
use crate::session::stream::{AvPingToken, StartSignal, StartWaiter};

mod gso_socket;
mod packetizer;
mod pipeline;
mod shard_batch;

pub use crate::session::compositor::frame::FrameColorSpace;
use gso_socket::UdpGsoSocket;
pub(crate) use pipeline::KcmpProbe;
use pipeline::{MAX_FRAMES_IN_FLIGHT, PacketBatch, VideoPipeline};

/// Configuration for the video stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VideoStreamConfig {
	/// Port to use for streaming video data.
	pub port: u16,

	/// What percentage of data packets should be parity packets.
	pub fec_percentage: u8,

	/// Whether to enable video stream encryption (AES-128-GCM).
	#[serde(default)]
	pub encrypt: bool,

	/// Whether to emit a WARN log when a single frame takes longer to encode and
	/// packetize than the frame budget.
	#[serde(default)]
	pub log_frame_spikes: bool,

	/// Upper bound for the client-requested video packet size, in bytes.
	///
	/// Moonlight's default packet size (1392 bytes) can exceed the path MTU
	/// over VPNs and tunnels, causing fragmented, dropped video. When non-zero,
	/// the client's `x-nv-video[0].packetSize` is clamped to this value; a
	/// smaller client request is honored. `0` disables the cap.
	///
	/// This is the stream's packet size, not a raw interface MTU: the on-wire
	/// UDP payload is `max_packet_size + 16` bytes. For example, to fit a
	/// 1420-byte WireGuard MTU over IPv4, use 1376 (1420 minus the 20-byte IP
	/// and 8-byte UDP headers and the 16-byte stream overhead).
	#[serde(default)]
	pub max_packet_size: usize,

	/// Size of the video socket's send buffer, in bytes.
	///
	/// Data in the send buffer has already left the pipeline, so it isn't
	/// covered by its frame dropping: when the host's own uplink is slower
	/// than the bitrate, up to this much video can queue in the kernel. The
	/// buffer only fills when that uplink is the bottleneck.
	///
	/// `0` sizes it to fit a keyframe burst (about eight average frames at the
	/// stream's bitrate and frame rate), keeping a larger host default. A
	/// non-zero value sets the size exactly, even below the host default: a
	/// smaller buffer lowers the worst-case queueing, at the cost of keyframes
	/// waiting for the socket to drain. The kernel caps it at
	/// `net.core.wmem_max`.
	#[serde(default)]
	pub send_buffer_size: usize,

	/// How the encoder holds the stream to its bitrate.
	#[serde(default)]
	pub rate_control: RateControlMode,
}

/// How the encoder holds the stream to its bitrate. Either way, no frame goes
/// over the per-frame budget (bitrate / frame rate).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RateControlMode {
	/// Constant bitrate. Some drivers (RADV) pad frames that need fewer bits
	/// with filler data up to the budget.
	#[default]
	Cbr,
	/// Variable bitrate capped at the bitrate: frames that need fewer bits are
	/// sent at their size, without filler. Needs a driver that supports VBR
	/// encoding.
	Vbr,
}

/// Smallest accepted packet size. Lower values would leave almost no room for
/// payload after the 16-byte NV video header, so a lower cap is ignored and a
/// smaller client request is rejected.
const MIN_PACKET_SIZE: usize = 200;
/// Largest accepted packet size, so one stream packet fits an IPv6 UDP datagram:
/// 65_535 - 40 (IPv6) - 8 (UDP) - 16 (stream overhead). The packetizer sizes its
/// buffers from the client's value, so an unbounded request could exhaust memory.
const MAX_PACKET_SIZE: usize = 65_471;

impl VideoStreamConfig {
	/// Whether a negotiated packet size is safe to packetize with.
	pub(crate) fn valid_packet_size(size: usize) -> bool {
		(MIN_PACKET_SIZE..=MAX_PACKET_SIZE).contains(&size)
	}

	/// Clamp a client-requested video packet size to `max_packet_size`.
	///
	/// A client only asks for what it believes it can receive, so a request
	/// smaller than the cap is honored; the cap only lowers oversized requests.
	/// A cap of `0` disables the limit.
	pub(crate) fn clamp_packet_size(&self, requested: usize) -> usize {
		if self.max_packet_size == 0 {
			return requested;
		}
		if self.max_packet_size < MIN_PACKET_SIZE {
			tracing::warn!(
				"max_packet_size {} is below the minimum of {MIN_PACKET_SIZE}, ignoring the cap.",
				self.max_packet_size
			);
			return requested;
		}
		requested.min(self.max_packet_size)
	}
}

impl Default for VideoStreamConfig {
	fn default() -> Self {
		Self {
			port: 47998,
			fec_percentage: 20,
			encrypt: false,
			log_frame_spikes: false,
			max_packet_size: 0,
			send_buffer_size: 0,
			rate_control: RateControlMode::default(),
		}
	}
}

/// The video bitrate that keeps a stream within `configured` bits per second,
/// as Sunshine computes it from `x-ml-video.configuredBitrateKbps` in
/// `cmd_announce()` (src/rtsp.cpp): take off the FEC parity (unless over 80%),
/// then the audio bitrate (at most a fifth of what's left), then 500 kb/s for
/// packet headers and control traffic (at most a tenth). The caps stop a low
/// bitrate from leaving almost nothing for video.
pub fn video_bitrate_within(configured: usize, fec_percentage: u8, audio_bitrate: usize) -> usize {
	let mut bitrate = configured as u64;
	if fec_percentage <= 80 {
		bitrate = bitrate * (100 - u64::from(fec_percentage)) / 100;
	}
	bitrate -= (audio_bitrate as u64).min(bitrate / 5);
	bitrate -= 500_000.min(bitrate / 10);
	bitrate as usize
}

/// Average frames a keyframe burst spans, FEC included, for sizing the
/// socket's send buffer when `send_buffer_size` is automatic.
const KEYFRAME_BURST_FRAMES: usize = 8;

/// Send buffer size that fits a keyframe burst at `bitrate` (bits per second)
/// and `fps`.
fn keyframe_burst_bytes(bitrate: usize, fps: u32) -> usize {
	let fps = usize::try_from(fps.max(1)).unwrap_or(usize::MAX);
	bitrate / 8 / fps * KEYFRAME_BURST_FRAMES
}

/// Statistics line format a `frame` line follows. Version 1 is [`FrameStats`]'s fields in
/// declaration order with their own names, every duration as whole microseconds, absent durations
/// omitted.
pub const STATS_VERSION: u8 = 1;

/// Writes a duration as whole microseconds, the unit every duration in this format uses.
fn serialize_micros<S: serde::Serializer>(duration: &std::time::Duration, serializer: S) -> Result<S::Ok, S::Error> {
	serializer.serialize_u64(u64::try_from(duration.as_micros()).unwrap_or(u64::MAX))
}

/// Writes an optional duration as whole microseconds. Pair with `skip_serializing_if` so absent
/// durations leave the record entirely.
fn serialize_micros_opt<S: serde::Serializer>(
	duration: &Option<std::time::Duration>,
	serializer: S,
) -> Result<S::Ok, S::Error> {
	match duration {
		Some(duration) => serialize_micros(duration, serializer),
		None => serializer.serialize_none(),
	}
}

/// Writes a point in time as microseconds since the Unix epoch.
fn serialize_unix_micros<S: serde::Serializer>(time: &std::time::SystemTime, serializer: S) -> Result<S::Ok, S::Error> {
	let since_epoch = time.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
	serialize_micros(&since_epoch, serializer)
}

/// Per-frame encoding statistics emitted by the video pipeline.
///
/// Sent in `StreamEvent::Frame`.
///
/// This struct's serde representation is the statistics socket's wire format: fields are written
/// with their own names in declaration order, durations as whole microseconds, and an absent duration is
/// omitted. A field added here is published on the socket by the same commit, so a field that must
/// stay internal needs `#[serde(skip)]`. `src/stats_socket.rs` locks the resulting format.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrameStats {
	/// The frame's number on the wire (its frame index), which the client sees too, so the two
	/// sides' records of a frame can be matched.
	pub frame_number: u32,
	/// When the frame was captured, by the wall clock.
	#[serde(serialize_with = "serialize_unix_micros")]
	pub captured_at: std::time::SystemTime,
	/// The color space of the captured content; `None` for a keyframe re-encoded without a new
	/// capture.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub content: Option<FrameColorSpace>,
	/// Time the frame spent waiting in the compositor's output channel.
	#[serde(serialize_with = "serialize_micros")]
	pub channel_wait: std::time::Duration,
	/// Time spent importing the DMA-BUF into Vulkan.
	#[serde(serialize_with = "serialize_micros")]
	pub import: std::time::Duration,
	/// Time spent on GPU color conversion.
	#[serde(serialize_with = "serialize_micros")]
	pub convert: std::time::Duration,
	/// Time spent submitting the frame to the asynchronous encoder.
	#[serde(serialize_with = "serialize_micros")]
	pub submit: std::time::Duration,
	/// Time between submit completion and the packet consumer awaiting the encode future.
	#[serde(serialize_with = "serialize_micros")]
	pub consumer_queue: std::time::Duration,
	/// Elapsed wall time from submit completion until the consumer obtains the encoded packet.
	/// Includes time queued before the consumer polls the encode future.
	#[serde(serialize_with = "serialize_micros")]
	pub encode_wait: std::time::Duration,
	/// GPU video encode command duration, when the video queue supports timestamps.
	/// This excludes conversion, queueing, and bitstream readback, and is part of `encode_wait`.
	#[serde(serialize_with = "serialize_micros_opt", skip_serializing_if = "Option::is_none")]
	pub gpu_encode: Option<std::time::Duration>,
	/// Pixelforge's wall time from encode submission until bitstream readback finishes.
	/// Available when the video queue supports timestamps. This includes GPU
	/// scheduling and readback, but not the consumer's post-completion scheduling, and is part of
	/// `encode_wait`.
	#[serde(serialize_with = "serialize_micros_opt", skip_serializing_if = "Option::is_none")]
	pub bitstream_ready: Option<std::time::Duration>,
	/// Time spent packetizing the encoded data.
	#[serde(serialize_with = "serialize_micros")]
	pub packetize: std::time::Duration,
	/// Time spent sending the packets over the channel.
	#[serde(serialize_with = "serialize_micros")]
	pub send: std::time::Duration,
	/// Total end-to-end latency for this frame.
	#[serde(serialize_with = "serialize_micros")]
	pub total: std::time::Duration,
	/// Number of bytes encoded for this frame.
	pub encoded_bytes: usize,
	/// Whether this frame is a key (IDR) frame.
	pub is_key_frame: bool,
}

/// A video stream's start, frames and end, in that order.
///
/// Receivable through `SessionManager::bench_stats_receiver()`.
#[derive(Clone, Debug)]
pub enum StreamEvent {
	/// The encoder for a new stream is ready.
	Started(StreamInfo),
	/// A frame was encoded and sent.
	Frame(FrameStats),
	/// The stream ended, after its last frame.
	Stopped,
}

/// Where the video pipeline sends its [`StreamEvent`]s. It also keeps the stream in progress, so
/// a reader that joins partway through can be told what it missed.
#[derive(Clone)]
pub struct StatsSender {
	events: broadcast::Sender<StreamEvent>,
	/// The stream in progress. Locked while a stream's start or end is broadcast, and while a
	/// reader reads it and subscribes, so a reader sees each start and end exactly once: in this,
	/// or as an event.
	current: Arc<std::sync::Mutex<Option<StreamInfo>>>,
}

impl StatsSender {
	pub(crate) fn new(capacity: usize) -> Self {
		Self {
			events: broadcast::channel(capacity).0,
			current: Default::default(),
		}
	}

	/// Every event from now on.
	pub(crate) fn subscribe(&self) -> broadcast::Receiver<StreamEvent> {
		self.events.subscribe()
	}

	/// The stream in progress, if any, and every event after it.
	pub(crate) fn join(&self) -> (Option<StreamInfo>, broadcast::Receiver<StreamEvent>) {
		let current = self.lock_current();
		(*current, self.events.subscribe())
	}

	pub(crate) fn started(&self, info: StreamInfo) {
		let mut current = self.lock_current();
		*current = Some(info);
		let _ = self.events.send(StreamEvent::Started(info));
	}

	pub(crate) fn frame(&self, stats: FrameStats) {
		let _ = self.events.send(StreamEvent::Frame(stats));
	}

	pub(crate) fn stopped(&self) {
		let mut current = self.lock_current();
		*current = None;
		let _ = self.events.send(StreamEvent::Stopped);
	}

	fn lock_current(&self) -> std::sync::MutexGuard<'_, Option<StreamInfo>> {
		self.current.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
	}
}

/// The video stream as negotiated with the client.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct StreamInfo {
	/// Width in pixels.
	pub width: u32,
	/// Height in pixels.
	pub height: u32,
	/// Frames per second.
	pub fps: u32,
	/// Target bitrate in bits per second.
	pub bitrate: usize,
	#[serde(rename = "codec")]
	pub video_format: VideoFormat,
	pub dynamic_range: VideoDynamicRange,
	#[serde(rename = "chroma")]
	pub chroma_sampling_type: VideoChromaSampling,
}

impl From<&VideoStreamContext> for StreamInfo {
	fn from(context: &VideoStreamContext) -> Self {
		Self {
			width: context.width,
			height: context.height,
			fps: context.fps,
			bitrate: context.bitrate,
			video_format: context.video_format,
			dynamic_range: context.dynamic_range,
			chroma_sampling_type: context.chroma_sampling_type,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoFormat {
	#[default]
	H264,
	Hevc,
	Av1,
}

impl TryFrom<u32> for VideoFormat {
	type Error = ();

	fn try_from(value: u32) -> Result<Self, Self::Error> {
		match value {
			0 => Ok(Self::H264),
			1 => Ok(Self::Hevc),
			2 => Ok(Self::Av1),
			_ => Err(()),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoDynamicRange {
	#[default]
	Sdr,
	Hdr,
}

impl TryFrom<u32> for VideoDynamicRange {
	type Error = ();

	fn try_from(value: u32) -> Result<Self, Self::Error> {
		match value {
			0 => Ok(Self::Sdr),
			1 => Ok(Self::Hdr),
			_ => Err(()),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoChromaSampling {
	#[default]
	Yuv420,
	Yuv444,
}

impl TryFrom<u32> for VideoChromaSampling {
	type Error = ();

	fn try_from(value: u32) -> Result<Self, Self::Error> {
		match value {
			0 => Ok(Self::Yuv420),
			1 => Ok(Self::Yuv444),
			_ => Err(()),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct VideoStreamContext {
	/// Width of the video stream in pixels.
	pub width: u32,

	/// Height of the video stream in pixels.
	pub height: u32,

	/// Frames per second of the video stream.
	pub fps: u32,

	/// Size of each encoded packet in bytes.
	pub packet_size: usize,

	/// Target bitrate for the video stream in bits per second.
	pub bitrate: usize,

	/// Minimum number of FEC packets to include for each frame.
	pub minimum_fec_packets: u32,

	/// Whether to apply QoS markings to video stream packets.
	pub qos: bool,

	/// Video format to use for encoding the stream.
	pub video_format: VideoFormat,

	/// Dynamic range of the video stream.
	pub dynamic_range: VideoDynamicRange,

	/// Chroma sampling type for the video stream.
	pub chroma_sampling_type: VideoChromaSampling,

	/// Maximum number of reference frames for the video encoder.
	pub max_reference_frames: u32,

	/// Whether the client asked for full-range (0-255) rather than
	/// limited-range (16-235) luma.
	pub full_range: bool,

	/// Whether the client has enabled video encryption.
	pub encrypt_video: bool,
}

/// Controls a single video generation. Network encoding waits for both StartB and a
/// matching video ping; the benchmark can explicitly start without a client.
#[derive(Clone)]
pub(crate) struct VideoStreamHandle {
	start: StartSignal,
	bench_start: StartSignal,
	idr_tx: broadcast::Sender<()>,
	/// Reference frame invalidation requests, carrying the inclusive
	/// `[first, last]` client frame-index range the client could not decode.
	invalidate_tx: broadcast::Sender<(u32, u32)>,
}

impl VideoStreamHandle {
	/// Signal StartB so the packet handler starts listening for the client's ping.
	pub fn trigger(&self) {
		self.start.trigger();
	}

	/// Request an IDR (key) frame from the encoder.
	pub fn request_idr_frame(&self) {
		let _ = self.idr_tx.send(());
	}

	/// Request reference frame invalidation for the inclusive client frame-index
	/// range `[first, last]` the client reported it could not decode.
	///
	/// The encoder drops the affected references and recovers by predicting from
	/// a surviving reference where possible, falling back to an IDR only when no
	/// reference survives — much cheaper than always re-sending a keyframe.
	pub fn invalidate_reference_frames(&self, first: u32, last: u32) {
		let _ = self.invalidate_tx.send((first, last));
	}

	/// Start encoding for the benchmark without requiring a client's ping.
	pub fn start_signal(&self) -> StartSignal {
		self.bench_start.clone()
	}
}

pub(crate) struct VideoStream {
	socket: Arc<UdpGsoSocket>,
	frames: CompositorFrames,
	capture_fps: u32,
	/// Nonces must not repeat when a reconnect negotiates the same AES key.
	nonce_counter: Arc<AtomicU64>,
	gpu: GpuContext,
	hdr_metadata_tx: watch::Sender<HdrModeState>,
	stats_tx: StatsSender,
}

impl VideoStream {
	pub async fn new(
		config: VideoStreamConfig,
		address: String,
		frames: CompositorFrames,
		gpu: GpuContext,
		hdr_metadata_tx: watch::Sender<HdrModeState>,
		_stop: ShutdownManager<SessionShutdownReason>,
		stats_tx: StatsSender,
	) -> Result<Self, ()> {
		tracing::debug!("Initializing video stream.");

		let socket = UdpGsoSocket::new(&address, config.port).await?;

		tracing::debug!(
			"Listening for video messages on {}",
			socket
				.local_addr()
				.map_err(|e| tracing::warn!("Failed to get local address associated with video socket: {e}"))?
		);

		Ok(Self {
			socket: Arc::new(socket),
			frames,
			capture_fps: 0,
			nonce_counter: Arc::default(),
			gpu,
			hdr_metadata_tx,
			stats_tx,
		})
	}

	/// Set the nominal refresh rate of the compositor, which survives media replacements.
	pub fn with_capture_fps(mut self, fps: u32) -> Self {
		self.capture_fps = fps;
		self
	}

	#[allow(clippy::too_many_arguments)]
	pub fn start(
		&self,
		config: VideoStreamConfig,
		context: VideoStreamContext,
		av_ping_token: AvPingToken,
		keys_rx: SessionKeysReceiver,
		stop: ShutdownManager<SessionShutdownReason>,
	) -> Result<VideoStreamHandle, ()> {
		let socket = &self.socket;
		let _ = socket.set_tos_v4(if context.qos { 160 } else { 0 });

		// Room for a keyframe burst, which is several times the average frame,
		// unless configured otherwise.
		if config.send_buffer_size == 0 {
			socket.grow_send_buffer(keyframe_burst_bytes(context.bitrate, context.fps));
		} else {
			socket.set_send_buffer(config.send_buffer_size);
		}

		// StartB enables transport; a matching ping enables encoding. The benchmark
		// opens both without a client so it can continue emitting frame stats.
		let start = StartSignal::new();
		let bench_start = StartSignal::new();
		let encode_start = StartSignal::new();
		let packet_start_waiter = start.waiter();
		let packet_bench_waiter = bench_start.waiter();
		let encode_waiter = encode_start.waiter();

		// IDR broadcast channel.
		let (idr_tx, _idr_rx) = broadcast::channel(1);

		// Reference frame invalidation broadcast channel. Sized for a small burst
		// of loss reports; the encode loop drains all pending each iteration.
		let (invalidate_tx, _invalidate_rx) = broadcast::channel(16);

		// Packet channel. Each queued batch holds its frame's in-flight slot, so the
		// channel never holds more than the in-flight limit.
		let (packet_tx, packet_rx) = mpsc::channel::<PacketBatch>(MAX_FRAMES_IN_FLIGHT);

		// Register the worker before spawning, so shutdown can never finish before
		// an as-yet-unscheduled task releases its socket and packet channel.
		let packet_delay = stop
			.delay_shutdown_token()
			.map_err(|e| tracing::error!("Can't start video packet handler: {e:?}"))?;

		self.hdr_metadata_tx
			.send_replace(HdrModeState::new(context.dynamic_range == VideoDynamicRange::Hdr));

		// Spawn the pipeline before the packet handler, since creating its thread
		// can fail. Both are gated and all waiters exist before StartB is possible.
		VideoPipeline::new(
			self.frames.clone(),
			self.capture_fps,
			self.gpu.clone(),
			config,
			context,
			keys_rx,
			packet_tx,
			idr_tx.clone(),
			idr_tx.subscribe(),
			invalidate_tx.subscribe(),
			self.nonce_counter.clone(),
			stop.clone(),
			self.hdr_metadata_tx.clone(),
			encode_waiter,
			self.stats_tx.clone(),
		)
		.map_err(|()| tracing::error!("Failed to create video pipeline"))?;

		spawn_handle_video_packets(
			packet_rx,
			socket.clone(),
			av_ping_token,
			packet_start_waiter,
			packet_bench_waiter,
			encode_start,
			stop,
			packet_delay,
		);

		Ok(VideoStreamHandle {
			start,
			bench_start,
			idr_tx,
			invalidate_tx,
		})
	}
}

#[allow(clippy::too_many_arguments)]
fn spawn_handle_video_packets(
	packet_rx: mpsc::Receiver<PacketBatch>,
	socket: Arc<UdpGsoSocket>,
	av_ping_token: AvPingToken,
	mut start: StartWaiter,
	mut bench_start: StartWaiter,
	encode_start: StartSignal,
	stop_session_manager: ShutdownManager<SessionShutdownReason>,
	delay_stop: DelayShutdownToken<SessionShutdownReason>,
) {
	tokio::spawn(async move {
		let _delay_stop = delay_stop;
		let (mut packet_rx, socket) = (packet_rx, socket);
		// Trigger session shutdown if we exit unexpectedly.
		let _stop_token = stop_session_manager.trigger_shutdown_token(SessionShutdownReason::VideoPacketHandlerStopped);

		let bench = tokio::select! {
			started = start.wait(&stop_session_manager) => if started { false } else { return },
			started = bench_start.wait(&stop_session_manager) => if started { true } else { return },
		};
		if bench {
			encode_start.trigger();
		}
		let mut encode_start = (!bench).then_some(encode_start);

		let mut buf = [0; 1024];
		let mut client_address = None;
		// Rate-limits the GSO-fallback warning.
		let mut last_send_warn: Option<std::time::Instant> = None;

		while !stop_session_manager.is_shutdown_triggered() {
			tokio::select! {
				// Don't drain the network queue before finding a client: frame #1
				// is its initial keyframe. The bench deliberately drains without a peer.
				batch = stop_session_manager.wrap_cancel(packet_rx.recv()), if encode_start.is_none() => {
					match batch {
						Ok(Some(batch)) => {
							// Taking the batch frees its frame's in-flight slot, so a slow
							// send doesn't hold back encoding; a stall still fills the
							// slots with the batches queued behind it.
							let shards = batch.into_shards();
							if let Some(addr) = client_address {
								if shards.shard_count() == 0 {
									continue;
								}

								// Sends are wrapped in wrap_cancel so a socket that
								// stops draining cannot block session shutdown.
								match stop_session_manager
									.wrap_cancel(socket.send_batch(&shards, addr))
									.await
								{
									Ok(failed_chunks) => {
										if failed_chunks > 0
											&& last_send_warn
												.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(1))
										{
											tracing::warn!(
												"GSO send failed for {failed_chunks} chunk(s), sent per-shard instead"
											);
											last_send_warn = Some(std::time::Instant::now());
										}
									},
									Err(_) => break,
								}
							}
						},
						Ok(None) => {
							tracing::debug!("Video packet channel closed.");
							break;
						},
						Err(_) => break,
					}
				},

				message = stop_session_manager.wrap_cancel(socket.recv_from(&mut buf)) => {
					let (len, address) = match message {
						Ok(Ok((len, address))) => (len, address),
						Ok(Err(e)) => {
							tracing::warn!("Failed to receive message: {e}");
							break;
						},
						Err(_) => break,
					};

					if av_ping_token.matches(&buf[..len]) {
						tracing::trace!("Received video stream PING message from {address}.");
						client_address = Some(address);

						if let Some(encode_start) = encode_start.take() {
							encode_start.trigger();
						}
					} else {
						tracing::warn!("Received unknown message on video stream of length {len}.");
					}
				},
			}
		}

		tracing::debug!("Video packet stream stopped.");
	});
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicUsize, Ordering};
	use std::time::{Duration, Instant};

	use super::packetizer::{FrameType, Packetizer};
	use super::pipeline::FramePacketizer;
	use super::shard_batch::ShardBatch;
	use super::*;

	const TIMEOUT: Duration = Duration::from_secs(1);

	fn spawn_test_packet_handler(
		packet_rx: mpsc::Receiver<PacketBatch>,
		socket: Arc<UdpGsoSocket>,
		token: AvPingToken,
		start: &StartSignal,
		bench_start: &StartSignal,
		stop: &ShutdownManager<SessionShutdownReason>,
	) -> StartWaiter {
		let encode_start = StartSignal::new();
		let encode_waiter = encode_start.waiter();
		spawn_handle_video_packets(
			packet_rx,
			socket,
			token,
			start.waiter(),
			bench_start.waiter(),
			encode_start,
			stop.clone(),
			stop.delay_shutdown_token().unwrap(),
		);
		encode_waiter
	}

	#[tokio::test]
	async fn shutdown_before_start_releases_packet_handler() {
		let socket = Arc::new(UdpGsoSocket::new("127.0.0.1", 0).await.unwrap());
		let (packet_tx, packet_rx) = mpsc::channel(1);
		let start = StartSignal::new();
		let bench_start = StartSignal::new();
		let stop = ShutdownManager::new();
		spawn_test_packet_handler(
			packet_rx,
			socket.clone(),
			AvPingToken::new().unwrap(),
			&start,
			&bench_start,
			&stop,
		);

		stop.trigger_shutdown(SessionShutdownReason::ManagerShutdown).unwrap();
		tokio::time::timeout(TIMEOUT, stop.wait_shutdown_complete())
			.await
			.unwrap();
		assert!(packet_tx.is_closed());
		assert_eq!(Arc::strong_count(&socket), 1);
	}

	async fn ping(client: &tokio::net::UdpSocket, address: std::net::SocketAddr, token: AvPingToken) {
		let mut packet = [0; 20];
		packet[..16].copy_from_slice(token.as_str().as_bytes());
		packet[16..].copy_from_slice(&1u32.to_be_bytes());
		client.send_to(&packet, address).await.unwrap();
	}

	async fn read_video_header(client: &tokio::net::UdpSocket) -> (usize, u16, u32, u32, u8) {
		let mut packet = [0; 1024];
		let (len, _) = tokio::time::timeout(TIMEOUT, client.recv_from(&mut packet))
			.await
			.expect("video handler did not send the queued frame")
			.unwrap();
		assert!(len >= 40);
		(
			len,
			u16::from_be_bytes(packet[2..4].try_into().unwrap()),
			u32::from_be_bytes(packet[4..8].try_into().unwrap()),
			u32::from_le_bytes(packet[20..24].try_into().unwrap()),
			packet[35],
		)
	}

	#[tokio::test]
	async fn resumed_video_waits_for_new_ping_and_starts_with_fresh_rtp_and_keyframe() {
		let socket = Arc::new(UdpGsoSocket::new("127.0.0.1", 0).await.unwrap());
		let address = socket.local_addr().unwrap();
		let former_client = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
		let resumed_client = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
		let former_token = AvPingToken::new().unwrap();
		let resumed_token = AvPingToken::new().unwrap();
		let (_, keys) = watch::channel(crate::session::SessionKeyData {
			remote_input_key: vec![0x42; 16],
			remote_input_key_id: 1,
		});
		let nonce_counter = Arc::new(AtomicU64::new(0));
		let in_flight = Arc::new(AtomicUsize::new(0));
		let config = VideoStreamConfig {
			fec_percentage: 0,
			..Default::default()
		};
		let mut context = VideoStreamContext {
			packet_size: 256,
			..Default::default()
		};
		let mut former = FramePacketizer::new(Packetizer::new(false, keys.clone(), nonce_counter.clone()));
		let former_stop = ShutdownManager::new();
		let start = StartSignal::new();
		let bench_start = StartSignal::new();
		let (former_tx, former_rx) = mpsc::channel(1);
		spawn_test_packet_handler(
			former_rx,
			socket.clone(),
			former_token,
			&start,
			&bench_start,
			&former_stop,
		);
		start.trigger();
		ping(&former_client, address, former_token).await;

		let now = Instant::now();
		let interval = Duration::from_millis(20);
		for (kind, captured_at) in [(FrameType::Key, now), (FrameType::Predicted, now + interval)] {
			let (_, shards) = former
				.packetize(b"encoded frame", kind, captured_at, 0, &context, &config)
				.unwrap();
			former_tx
				.send(PacketBatch::holding_slot(&in_flight, shards))
				.await
				.unwrap();
		}
		assert_eq!(
			read_video_header(&former_client).await,
			(272, 0, 0, 1, FrameType::Key as u8)
		);
		assert_eq!(
			read_video_header(&former_client).await,
			(272, 1, 1800, 2, FrameType::Predicted as u8)
		);

		former_stop
			.trigger_shutdown(SessionShutdownReason::MediaReplaced)
			.unwrap();
		tokio::time::timeout(TIMEOUT, former_stop.wait_shutdown_complete())
			.await
			.unwrap();
		assert!(former_tx.is_closed());

		context.packet_size = 384;
		let mut resumed = FramePacketizer::new(Packetizer::new(false, keys, nonce_counter));
		let stop = ShutdownManager::new();
		let start = StartSignal::new();
		let (resumed_tx, resumed_rx) = mpsc::channel(1);
		let mut encode_waiter =
			spawn_test_packet_handler(resumed_rx, socket, resumed_token, &start, &bench_start, &stop);
		start.trigger();
		let (_, shards) = resumed
			.packetize(b"resumed IDR", FrameType::Key, now + 2 * interval, 0, &context, &config)
			.unwrap();
		resumed_tx
			.send(PacketBatch::holding_slot(&in_flight, shards))
			.await
			.unwrap();

		ping(&former_client, address, former_token).await;
		assert!(
			tokio::time::timeout(Duration::from_millis(30), encode_waiter.wait(&stop))
				.await
				.is_err()
		);
		assert_eq!(
			resumed_tx.capacity(),
			0,
			"the first frame was consumed without a matching ping"
		);

		ping(&resumed_client, address, resumed_token).await;
		assert!(tokio::time::timeout(TIMEOUT, encode_waiter.wait(&stop)).await.unwrap());
		assert_eq!(
			read_video_header(&resumed_client).await,
			(400, 0, 0, 1, FrameType::Key as u8)
		);
		stop.trigger_shutdown(SessionShutdownReason::ManagerShutdown).unwrap();
		tokio::time::timeout(TIMEOUT, stop.wait_shutdown_complete())
			.await
			.unwrap();
	}

	#[tokio::test]
	async fn stalled_sender_fills_in_flight_slots() {
		let socket = Arc::new(UdpGsoSocket::new("127.0.0.1", 0).await.unwrap());
		let (packet_tx, packet_rx) = mpsc::channel(MAX_FRAMES_IN_FLIGHT);
		let start = StartSignal::new();
		let bench_start = StartSignal::new();
		let stop = ShutdownManager::new();
		let mut encode_waiter = spawn_test_packet_handler(
			packet_rx,
			socket,
			AvPingToken::new().unwrap(),
			&start,
			&bench_start,
			&stop,
		);

		let in_flight = Arc::new(AtomicUsize::new(0));
		for _ in 0..MAX_FRAMES_IN_FLIGHT {
			packet_tx
				.try_send(PacketBatch::holding_slot(&in_flight, ShardBatch::empty()))
				.unwrap();
		}
		// Queued frames keep their slots, so the encoding thread's gate closes and
		// it drops captures instead of the queue growing.
		assert_eq!(in_flight.load(Ordering::Relaxed), MAX_FRAMES_IN_FLIGHT);
		assert_eq!(packet_tx.capacity(), 0);

		// The benchmark enables encoding and frees slots without a ping.
		bench_start.trigger();
		assert!(tokio::time::timeout(TIMEOUT, encode_waiter.wait(&stop)).await.unwrap());
		tokio::time::timeout(TIMEOUT, async {
			while in_flight.load(Ordering::Relaxed) > 0 {
				tokio::task::yield_now().await;
			}
		})
		.await
		.expect("sender kept in-flight slots after taking their batches");
		stop.trigger_shutdown(SessionShutdownReason::ManagerShutdown).unwrap();
		tokio::time::timeout(TIMEOUT, stop.wait_shutdown_complete())
			.await
			.unwrap();
	}

	#[test]
	fn keyframe_burst_scales_with_frame_size() {
		// 80 Mbps at 60 fps: ~166 KB average frames.
		assert_eq!(keyframe_burst_bytes(80_000_000, 60), 166_666 * KEYFRAME_BURST_FRAMES);
		// Doubling the frame rate halves the frames, and the burst.
		assert_eq!(keyframe_burst_bytes(80_000_000, 120), 83_333 * KEYFRAME_BURST_FRAMES);
		// A zero frame rate doesn't divide by zero.
		assert_eq!(keyframe_burst_bytes(8_000, 0), 1_000 * KEYFRAME_BURST_FRAMES);
	}

	#[test]
	fn video_bitrate_leaves_room_for_fec_audio_and_overhead() {
		// Sunshine's numbers for 20 Mb/s, 20% FEC and high-quality stereo.
		assert_eq!(video_bitrate_within(20_000_000, 20, 512_000), 14_988_000);
		// At a low bitrate the caps keep most of it for video.
		assert_eq!(video_bitrate_within(1_000_000, 20, 512_000), 576_000);
		// FEC over 80% isn't taken off.
		assert_eq!(video_bitrate_within(20_000_000, 90, 512_000), 18_988_000);
		// The largest bitrate a client can ask for doesn't overflow.
		let largest = u32::MAX as usize;
		assert!(video_bitrate_within(largest, 0, 0) < largest);
	}

	fn config(max_packet_size: usize) -> VideoStreamConfig {
		VideoStreamConfig {
			max_packet_size,
			..Default::default()
		}
	}

	#[test]
	fn rejects_out_of_range_packet_sizes() {
		assert!(!VideoStreamConfig::valid_packet_size(0));
		assert!(!VideoStreamConfig::valid_packet_size(16));
		assert!(!VideoStreamConfig::valid_packet_size(MIN_PACKET_SIZE - 1));
		assert!(VideoStreamConfig::valid_packet_size(MIN_PACKET_SIZE));
		assert!(VideoStreamConfig::valid_packet_size(1392));
		assert!(VideoStreamConfig::valid_packet_size(MAX_PACKET_SIZE));
		assert!(!VideoStreamConfig::valid_packet_size(MAX_PACKET_SIZE + 1));
		assert!(!VideoStreamConfig::valid_packet_size(usize::MAX));
	}

	#[test]
	fn no_cap_honors_requested() {
		assert_eq!(config(0).clamp_packet_size(1392), 1392);
	}

	#[test]
	fn smaller_client_request_is_honored() {
		assert_eq!(config(1200).clamp_packet_size(1024), 1024);
	}

	#[test]
	fn larger_client_request_is_capped() {
		assert_eq!(config(1200).clamp_packet_size(1392), 1200);
	}

	#[test]
	fn undersized_cap_is_ignored() {
		assert_eq!(config(50).clamp_packet_size(1392), 1392);
	}
}
