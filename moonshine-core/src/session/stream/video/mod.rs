use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_shutdown::ShutdownManager;
use quinn_udp::{Transmit, UdpSockRef, UdpSocketState};
use serde::{Deserialize, Serialize};
use tokio::{
	io::Interest,
	net::UdpSocket,
	sync::{broadcast, mpsc, watch, Notify},
};

use crate::session::compositor::frame::{ExportedFrame, HdrModeState};
use crate::session::manager::SessionShutdownReason;
use crate::session::SessionKeysReceiver;

mod packetizer;
mod pipeline;
mod shard_batch;
use pipeline::VideoPipeline;
use shard_batch::ShardBatch;

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

	/// Send-side packet pacing: instead of bursting a frame's UDP packets
	/// back-to-back (which can overflow the last-hop router/NIC queue and cause
	/// loss → forced IDR re-sends), spread them out at a target rate of
	/// `pacing_rate_factor × bitrate`. A frame then drains in roughly
	/// `frame_interval / pacing_rate_factor`, so higher values add less latency
	/// but smooth the burst less. `0` disables pacing (packets are bursted);
	/// values between 0 and 1 are clamped up to 1 (draining slower than the frame
	/// rate would let frames back up).
	#[serde(default = "default_pacing_rate_factor")]
	pub pacing_rate_factor: f32,
}

fn default_pacing_rate_factor() -> f32 {
	8.0
}

impl Default for VideoStreamConfig {
	fn default() -> Self {
		Self {
			port: 47998,
			fec_percentage: 20,
			encrypt: false,
			log_frame_spikes: false,
			pacing_rate_factor: default_pacing_rate_factor(),
		}
	}
}

/// Per-frame encoding statistics emitted by the video pipeline.
///
/// Sent via `broadcast` channel, receivable through `SessionManager::bench_stats_receiver()`.
#[derive(Clone, Debug)]
pub struct FrameStats {
	/// Time the frame spent waiting in the compositor's output channel.
	pub channel_wait: std::time::Duration,
	/// Time spent importing the DMA-BUF into Vulkan.
	pub import: std::time::Duration,
	/// Time spent on GPU color conversion.
	pub convert: std::time::Duration,
	/// Time spent encoding the frame.
	pub encode: std::time::Duration,
	/// Time spent packetizing the encoded data.
	pub packetize: std::time::Duration,
	/// Time spent sending the packets over the channel.
	pub send: std::time::Duration,
	/// Total end-to-end latency for this frame.
	pub total: std::time::Duration,
	/// Number of bytes encoded for this frame.
	pub encoded_bytes: usize,
	/// Whether this frame is a key (IDR) frame.
	pub is_key_frame: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

	/// Whether the client has enabled video encryption.
	pub encrypt_video: bool,
}

/// Handle returned by `VideoStream::start` that gates the pipeline and packet handler.
///
/// The pipeline and packet handler are spawned immediately but block on a `Notify`
/// until `trigger()` is called on `StartB`.
#[derive(Clone)]
pub(crate) struct VideoStreamHandle {
	notify: Arc<Notify>,
	idr_tx: broadcast::Sender<()>,
	/// Set on resume to arm a stream reset; the packet handler fires it once it has
	/// re-learned the reconnecting client's address (see `request_reset`).
	resume_pending: Arc<AtomicBool>,
}

impl VideoStreamHandle {
	/// Signal the video pipeline and packet handler to begin processing.
	pub fn trigger(&self) {
		self.notify.notify_waiters();
	}

	/// Request an IDR (key) frame from the encoder.
	pub fn request_idr_frame(&self) {
		let _ = self.idr_tx.send(());
	}

	/// Arm a stream reset for a resuming client.
	///
	/// Called when a client reconnects to an already-running session. The pipeline
	/// keeps incrementing `frame_number` for the lifetime of the session, but a fresh
	/// Moonlight session expects frame numbers to start at 1; without a reset it counts
	/// the jump as massive frame loss and reports a poor connection. The reset also forces
	/// an IDR so the resumed client has a decodable starting frame.
	///
	/// The reset is not fired immediately: the packet handler still holds the previous
	/// connection's address, and a reconnecting client almost always arrives on a new UDP
	/// source port. Firing now would spend the forced IDR on the stale address, the client
	/// would receive no decodable frame, and it would abort with a connection error
	/// (typically recovering only on a retry). Instead we arm a flag that the packet handler
	/// consumes once it has re-learned the client's address from its first PING, so the IDR
	/// lands where the client is actually listening.
	pub fn request_reset(&self) {
		self.resume_pending.store(true, Ordering::Relaxed);
	}

	/// Clone the start notify for external triggering (e.g. bench binary).
	pub fn clone_start_notify(&self) -> Arc<Notify> {
		self.notify.clone()
	}
}

pub(crate) struct VideoStream {
	socket: UdpSocket,
	frame_rx: std::sync::mpsc::Receiver<ExportedFrame>,
	hdr_metadata_tx: watch::Sender<HdrModeState>,
	stats_tx: tokio::sync::broadcast::Sender<FrameStats>,
}

impl VideoStream {
	pub async fn new(
		config: VideoStreamConfig,
		address: String,
		frame_rx: std::sync::mpsc::Receiver<ExportedFrame>,
		hdr_metadata_tx: watch::Sender<HdrModeState>,
		_stop: ShutdownManager<SessionShutdownReason>,
		stats_tx: tokio::sync::broadcast::Sender<FrameStats>,
	) -> Result<Self, ()> {
		tracing::debug!("Initializing video stream.");

		let socket = UdpSocket::bind((address.as_str(), config.port))
			.await
			.map_err(|e| tracing::error!("Failed to bind to UDP socket: {e}"))?;

		tracing::debug!(
			"Listening for video messages on {}",
			socket
				.local_addr()
				.map_err(|e| tracing::warn!("Failed to get local address associated with video socket: {e}"))?
		);

		Ok(Self {
			socket,
			frame_rx,
			hdr_metadata_tx,
			stats_tx,
		})
	}

	#[allow(clippy::too_many_arguments)]
	pub fn start(
		self,
		config: VideoStreamConfig,
		context: VideoStreamContext,
		keys_rx: SessionKeysReceiver,
		stop: ShutdownManager<SessionShutdownReason>,
	) -> Result<VideoStreamHandle, ()> {
		let Self {
			socket,
			frame_rx,
			hdr_metadata_tx,
			stats_tx,
		} = self;

		// Apply QoS to UDP socket.
		if context.qos {
			let _ = socket.set_tos_v4(160);
		}

		// Wrap the socket with quinn-udp's send state, which drives UDP GSO
		// (`UDP_SEGMENT`) with automatic capability detection and fallback, plus
		// `sendmmsg` batching where GSO is unavailable.
		let udp_state = UdpSocketState::new(UdpSockRef::from(&socket))
			.map_err(|e| tracing::error!("Failed to initialize UDP socket state: {e}"))?;

		// Build the send-side pacer from the negotiated bitrate and the platform's
		// max GSO segment count.
		let pacer = Pacer::new(config.pacing_rate_factor, context.bitrate, udp_state.max_gso_segments());

		// Gate for pipeline + packet handler.
		let start_notify = Arc::new(Notify::new());

		// IDR broadcast channel.
		let (idr_tx, _idr_rx) = broadcast::channel(1);

		// Stream-reset broadcast channel (client reconnect/resume). The packet handler
		// fires it once it has re-learned the reconnecting client's address.
		let (reset_tx, _reset_rx) = broadcast::channel(1);
		let resume_pending = Arc::new(AtomicBool::new(false));

		// Packet channel.
		let (packet_tx, packet_rx) = mpsc::channel::<ShardBatch>(128);

		// Spawn packet handler — gated behind start_notify.
		spawn_handle_video_packets(
			packet_rx,
			socket,
			udp_state,
			pacer,
			start_notify.clone(),
			reset_tx.clone(),
			resume_pending.clone(),
			stop.clone(),
		);

		// Spawn pipeline thread — gated behind start_notify.
		VideoPipeline::new(
			frame_rx,
			config,
			context,
			keys_rx,
			packet_tx,
			idr_tx.subscribe(),
			reset_tx.subscribe(),
			stop.clone(),
			hdr_metadata_tx,
			start_notify.clone(),
			stats_tx,
		)
		.map_err(|()| tracing::error!("Failed to create video pipeline"))?;

		Ok(VideoStreamHandle {
			notify: start_notify,
			idr_tx,
			resume_pending,
		})
	}
}

/// Largest payload the kernel will segment from a single GSO `sendmsg`. The
/// number of equal-sized segments per send is also bounded by this.
const MAX_GSO_BYTES: usize = 65_535;

/// Send-side packet pacer with UDP GSO batching.
///
/// Sends a frame's shards at a target rate (`rate` bytes/sec) rather than
/// bursting them, so a microburst doesn't overflow the last-hop queue and
/// trigger loss → forced IDR re-sends. Crucially the spread is *rate*-based, not
/// spread-across-the-frame: a frame drains in ≈ `frame_bytes / rate`, so small
/// frames go out almost immediately and only large frames (IDRs) are spread —
/// keeping added latency low and proportional to frame size.
///
/// Each send hands quinn-udp a contiguous run of equal-sized shards with a
/// `segment_size`, which it transmits via UDP GSO (`UDP_SEGMENT`) — or
/// `sendmmsg`/per-packet where GSO is unavailable, detected and handled by
/// quinn-udp itself. Pacing happens between these GSO sends.
#[derive(Clone, Copy)]
struct Pacer {
	/// Target send rate in bytes/sec. Zero disables pacing (sends are flushed
	/// back-to-back via the largest GSO batches).
	rate: u64,
	/// Don't issue a sleep shorter than this — instead let sends coalesce. Sub-
	/// millisecond sleeps fight the timer's ~1ms granularity and only add
	/// overhead.
	min_gap: Duration,
	/// Platform cap on segments per GSO send, as reported by quinn-udp (1 when
	/// GSO is unavailable, collapsing each send to a single datagram).
	max_gso_segments: usize,
}

impl Pacer {
	fn new(rate_factor: f32, bitrate: usize, max_gso_segments: usize) -> Self {
		// Target rate in bytes/sec = (bitrate / 8) * factor. A factor of 0 (or no
		// bitrate) disables pacing; otherwise it's clamped to at least 1.0 so a
		// frame never takes longer than its own interval to drain.
		let rate = if rate_factor > 0.0 && bitrate > 0 {
			((bitrate as f64 / 8.0) * rate_factor.max(1.0) as f64) as u64
		} else {
			0
		};

		Self {
			rate,
			// Aligned with the async timer's ~1ms granularity: shorter sleeps
			// would just round up to this anyway.
			min_gap: Duration::from_millis(1),
			max_gso_segments: max_gso_segments.max(1),
		}
	}

	/// Send all shards of a frame to `address`, pacing the sends at `self.rate`.
	///
	/// Shards are emitted in contiguous chunks, one quinn-udp send per chunk.
	/// Pacing uses an absolute schedule anchored at the first send (so transient
	/// send latency doesn't accumulate drift): each chunk is due once the bytes
	/// before it would have drained at `self.rate`, and we only sleep when that
	/// deadline is at least `min_gap` away.
	async fn send_batch(&self, socket: &UdpSocket, state: &UdpSocketState, batch: &ShardBatch, address: SocketAddr) {
		let count = batch.shard_count();
		if count == 0 {
			return;
		}

		let seg_size = batch.shard_size();
		let bytes = batch.as_bytes();

		// Segments per send, bounded by both the platform segment cap and the
		// 64 KB GSO payload limit.
		let gso_max = self.max_gso_segments.min(MAX_GSO_BYTES / seg_size.max(1)).max(1);

		// Choose how many segments go in each send.
		//
		// Without pacing, fill each send to the GSO limit for the fewest syscalls.
		// With pacing, size each chunk to roughly one `min_gap` worth of bytes at
		// the target rate, so consecutive sends land ≈ `min_gap` apart (the finest
		// the timer resolves) — never exceeding the per-send GSO cap.
		let segs_per_send = if self.rate == 0 {
			gso_max
		} else {
			let bytes_per_gap = (self.rate as u128 * self.min_gap.as_nanos() / 1_000_000_000) as usize;
			(bytes_per_gap / seg_size.max(1)).clamp(1, gso_max)
		};

		// Don't bother sleeping when we're already within this margin of a chunk's
		// deadline: the timer would round a sub-millisecond sleep up to ~1ms
		// anyway, and the absolute schedule self-corrects on the next chunk.
		const SLEEP_FLOOR: Duration = Duration::from_micros(250);

		let start = tokio::time::Instant::now();
		let mut sent = 0usize;

		while sent < count {
			// Pace: this chunk is due once the bytes before it have drained at the
			// target rate. The first chunk (sent == 0) lands immediately.
			if self.rate > 0 && sent > 0 {
				let drained_ns = (sent * seg_size) as u128 * 1_000_000_000 / self.rate as u128;
				let deadline = start + Duration::from_nanos(drained_ns as u64);
				if deadline.saturating_duration_since(tokio::time::Instant::now()) >= SLEEP_FLOOR {
					tokio::time::sleep_until(deadline).await;
				}
			}

			let chunk_shards = segs_per_send.min(count - sent);
			let chunk = &bytes[sent * seg_size..(sent + chunk_shards) * seg_size];

			// A single-shard send needs no segmentation; multi-shard sends carry a
			// `segment_size` so quinn-udp uses GSO (or its own fallback).
			let segment_size = (chunk_shards > 1).then_some(seg_size);
			let transmit = Transmit {
				destination: address,
				ecn: None,
				contents: chunk,
				segment_size,
				src_ip: None,
			};

			if let Err(e) = send_transmit(socket, state, &transmit).await {
				tracing::warn!("Failed to send video packets to client: {e}");
			}

			sent += chunk_shards;
		}
	}
}

/// Drive one quinn-udp send over the tokio socket. Tries the send immediately
/// (the common case — the socket is writable) and only awaits writability when
/// it would block, avoiding a reactor round-trip per send. quinn-udp logs and
/// swallows non-fatal send errors, returning only `WouldBlock`.
async fn send_transmit(socket: &UdpSocket, state: &UdpSocketState, transmit: &Transmit<'_>) -> io::Result<()> {
	loop {
		match socket.try_io(Interest::WRITABLE, || state.send(UdpSockRef::from(socket), transmit)) {
			Ok(()) => return Ok(()),
			Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => socket.writable().await?,
			Err(e) => return Err(e),
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn spawn_handle_video_packets(
	mut packet_rx: mpsc::Receiver<ShardBatch>,
	socket: UdpSocket,
	udp_state: UdpSocketState,
	pacer: Pacer,
	start: Arc<Notify>,
	reset_tx: broadcast::Sender<()>,
	resume_pending: Arc<AtomicBool>,
	stop_session_manager: ShutdownManager<SessionShutdownReason>,
) {
	tokio::spawn(async move {
		start.notified().await;

		let mut buf = [0; 1024];
		let mut client_address = None;

		// Trigger session shutdown if we exit unexpectedly.
		let _stop_token = stop_session_manager.trigger_shutdown_token(SessionShutdownReason::VideoPacketHandlerStopped);
		let _delay_stop = stop_session_manager.delay_shutdown_token();

		while !stop_session_manager.is_shutdown_triggered() {
			tokio::select! {
				batch = packet_rx.recv() => {
					match batch {
						Some(batch) => {
							if let Some(client_address) = client_address {
								pacer.send_batch(&socket, &udp_state, &batch, client_address).await;
							}
						},
						None => {
							tracing::debug!("Video packet channel closed.");
							break;
						},
					}
				},

				message = socket.recv_from(&mut buf) => {
					let (len, address) = match message {
						Ok((len, address)) => (len, address),
						Err(e) => {
							tracing::warn!("Failed to receive message: {e}");
							break;
						},
					};

					if &buf[..len] == b"PING" {
						tracing::trace!("Received video stream PING message from {address}.");
						client_address = Some(address);

						// A resume armed a stream reset (frame-counter reset + forced IDR). Fire it
						// now that we know where the reconnecting client is listening, so the forced
						// IDR is sent to the current address instead of the previous connection's.
						if resume_pending.swap(false, Ordering::Relaxed) {
							tracing::info!("Re-learned client address after resume; firing armed stream reset.");
							let _ = reset_tx.send(());
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
