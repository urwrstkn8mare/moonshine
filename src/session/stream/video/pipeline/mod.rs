//! Video encoding and streaming pipeline.
//!
//! This module handles video encoding with pixelforge
//! and packetization for network transmission.

mod dmabuf;
mod fence;
mod hdr_sei;

use std::os::fd::{AsFd, AsRawFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ash::vk;
use async_shutdown::ShutdownManager;
use smithay::backend::allocator::Buffer;
use tokio::sync::{broadcast, mpsc, watch};

use crate::gpu::GpuContext;
use crate::session::SessionKeysReceiver;
use crate::session::compositor::CompositorFrames;
use crate::session::compositor::frame::{FrameColorSpace, HdrMetadata, HdrModeState, color_space, hdr_metadata};
use crate::session::manager::SessionShutdownReason;
use crate::session::stream::StartWaiter;

use crate::session::stream::video::packetizer::{FrameType, Packetizer};
use crate::session::stream::video::shard_batch::ShardBatch;
use crate::session::stream::video::{
	FrameStats, RateControlMode, StatsSender, StreamInfo, VideoChromaSampling, VideoDynamicRange, VideoFormat,
	VideoStreamConfig, VideoStreamContext,
};

pub(crate) use dmabuf::KcmpProbe;
use dmabuf::{DmaBufImporter, DmaBufPlane};

use pixelforge::{
	ChromaFormat, Codec, ColorRange, ColorSpec, EncodeConfig, EncodeFuture, Encoder, EncoderTuningMode, Frame,
	PixelForgeError, RateControl, RgbFormat, RgbSource, Source, StreamColor, VideoContext,
};

/// Maximum number of frames in flight (submitted to the encoder but not yet
/// taken by the network sender) before we start dropping new captures.
///
/// This is the pipeline's backpressure policy, chosen for low latency: rather
/// than block the encoding thread and let already-captured frames queue up
/// (growing end-to-end latency), when the consumer (packetize/network-send)
/// falls this far behind we *drop the new capture before encoding it* so the
/// stream stays realtime. Dropping pre-encode keeps the encoded P-frame chain
/// valid — a dropped frame never enters the encoder's reference state, unlike
/// discarding an already-encoded packet, which would break decoding until the
/// next IDR. Sized just above pixelforge's encode pipeline depth (2) to keep the
/// GPU fed plus one slot of slack, bounding added latency to ~3 frames.
///
/// This bounds queueing inside the application only. A frame leaves the count
/// when the sender takes it, before it is written to the socket, so data the
/// socket and the kernel below it still hold isn't counted here; the socket's
/// share is bounded by its send buffer (see `VideoStreamConfig::send_buffer_size`).
pub(super) const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Map a DRM fourcc format code to the corresponding pixelforge RgbFormat
/// and Vulkan import format.
fn drm_fourcc_to_input(fourcc: u32) -> (RgbFormat, vk::Format) {
	// DRM fourcc values (from drm_fourcc.h):
	// ARGB8888 = 0x34325241, XRGB8888 = 0x34325258
	// ABGR8888 = 0x34324241, XBGR8888 = 0x34324258
	// ABGR2101010 = 0x30334241, XBGR2101010 = 0x30334258
	// ARGB2101010 = 0x30335241, XRGB2101010 = 0x30335258
	// ABGR16161616F = 0x48344241, XBGR16161616F = 0x48344258
	//
	// X-variants share the same bit layout as their A-counterparts; the
	// alpha bits are simply ignored. vkd3d-proton's swapchain typically
	// allocates XBGR2101010 (XB30) for HDR — treating that as the BGRx
	// fallback (8-bit BGRA) produces rainbow corruption when 10-bit
	// packed data is read as 8-bit channels.
	match fourcc {
		0x34324241 | 0x34324258 => (RgbFormat::RGBA, vk::Format::R8G8B8A8_UNORM), // ABGR/XBGR8888
		0x30334241 | 0x30334258 => {
			(RgbFormat::ABGR2101010, vk::Format::A2B10G10R10_UNORM_PACK32) // ABGR/XBGR 2101010
		},
		0x48344241 | 0x48344258 => (RgbFormat::RGBA16F, vk::Format::R16G16B16A16_SFLOAT), // ABGR/XBGR16161616F
		_ => (RgbFormat::BGRx, vk::Format::B8G8R8A8_UNORM),                               // ARGB/XRGB8888 (fallback)
	}
}

/// The colour range a session's stream uses.
fn range(full_range: bool) -> ColorRange {
	if full_range {
		ColorRange::Full
	} else {
		ColorRange::Limited
	}
}

/// What a session's frames hold and what its stream says, before any frame
/// says otherwise: sRGB as BT.709, or PQ as HDR10.
fn session_colors(dynamic_range: VideoDynamicRange, full_range: bool) -> (ColorSpec, StreamColor) {
	let range = range(full_range);
	match dynamic_range {
		VideoDynamicRange::Sdr => (ColorSpec::Srgb, StreamColor::Sdr { range }),
		VideoDynamicRange::Hdr => (ColorSpec::Bt2020Pq, StreamColor::Hdr10 { range }),
	}
}

/// What an HDR session's frame holds, and what the stream becomes for it: SDR
/// frames stay BT.709, HDR frames become BT.2020+PQ.
fn frame_colors(frame: FrameColorSpace, full_range: bool) -> (ColorSpec, StreamColor) {
	let range = range(full_range);
	match frame {
		FrameColorSpace::Srgb => (ColorSpec::Srgb, StreamColor::Sdr { range }),
		FrameColorSpace::Bt2020Pq => (ColorSpec::Bt2020Pq, StreamColor::Hdr10 { range }),
		FrameColorSpace::ScrgbLinear => (ColorSpec::Bt709Linear, StreamColor::Hdr10 { range }),
	}
}

/// Whether the encoder currently writes an HDR10 stream.
fn is_hdr(encoder: &Encoder) -> bool {
	matches!(encoder.config().color, StreamColor::Hdr10 { .. })
}

/// Whether an invalidation request for the lost display orders `first..=last`
/// repeats the last one answered, `answered = (its first lost display order,
/// the display order of the frame answering it)`. The client reports every frame
/// it drops until that answer reaches it; a request that reaches the answering
/// frame means the answer was lost too.
fn covered_by(answered: Option<(u64, u64)>, first: u64, last: u64) -> bool {
	answered.is_some_and(|(answered_first, answer)| first >= answered_first && last < answer)
}

pub(crate) struct VideoPipeline {}

/// A single frame's latency breakdown for periodic summary reporting.
struct LatencySample {
	channel_wait: std::time::Duration,
	import: std::time::Duration,
	convert: std::time::Duration,
	submit: std::time::Duration,
	consumer_queue: std::time::Duration,
	encode_wait: std::time::Duration,
	gpu_encode: Option<std::time::Duration>,
	bitstream_ready: Option<std::time::Duration>,
	packetize: std::time::Duration,
	send: std::time::Duration,
	total: std::time::Duration,
	encoded_bytes: usize,
	is_key_frame: bool,
}

/// Log a summary of latency statistics over a batch of samples.
///
/// `elapsed` is the wall-clock time the batch was collected over, used to
/// derive throughput (fps, bitrate).
fn log_latency_summary(samples: &[LatencySample], elapsed: std::time::Duration) {
	let n = samples.len();
	if n == 0 {
		return;
	}

	let elapsed_s = elapsed.as_secs_f64().max(f64::EPSILON);
	let bytes_total: usize = samples.iter().map(|s| s.encoded_bytes).sum();
	let keyframes = samples.iter().filter(|s| s.is_key_frame).count();

	let mut totals: Vec<u64> = samples.iter().map(|s| s.total.as_micros() as u64).collect();
	totals.sort_unstable();

	let mut channel: Vec<u64> = samples.iter().map(|s| s.channel_wait.as_micros() as u64).collect();
	channel.sort_unstable();

	let mut imports: Vec<u64> = samples.iter().map(|s| s.import.as_micros() as u64).collect();
	imports.sort_unstable();

	let mut converts: Vec<u64> = samples.iter().map(|s| s.convert.as_micros() as u64).collect();
	converts.sort_unstable();

	let mut submits: Vec<u64> = samples.iter().map(|s| s.submit.as_micros() as u64).collect();
	submits.sort_unstable();

	let mut consumer_queues: Vec<u64> = samples.iter().map(|s| s.consumer_queue.as_micros() as u64).collect();
	consumer_queues.sort_unstable();

	let mut encode_waits: Vec<u64> = samples.iter().map(|s| s.encode_wait.as_micros() as u64).collect();
	encode_waits.sort_unstable();

	// Pixelforge reports these only when video-queue GPU timestamps are supported.
	// A missing sample must not be interpreted as zero GPU time.
	let mut gpu_encodes: Vec<u64> = samples
		.iter()
		.filter_map(|s| s.gpu_encode.map(|d| d.as_micros() as u64))
		.collect();
	gpu_encodes.sort_unstable();
	let mut bitstream_readies: Vec<u64> = samples
		.iter()
		.filter_map(|s| s.bitstream_ready.map(|d| d.as_micros() as u64))
		.collect();
	bitstream_readies.sort_unstable();

	let mut packetizes: Vec<u64> = samples.iter().map(|s| s.packetize.as_micros() as u64).collect();
	packetizes.sort_unstable();

	let mut sends: Vec<u64> = samples.iter().map(|s| s.send.as_micros() as u64).collect();
	sends.sort_unstable();

	let p50 = |v: &[u64]| v[v.len() / 2];
	let p95 = |v: &[u64]| v[(v.len() as f64 * 0.95) as usize];
	let p99 = |v: &[u64]| v[(v.len() as f64 * 0.99) as usize];
	let gpu_encode_p50_us = (!gpu_encodes.is_empty()).then(|| p50(&gpu_encodes));
	let gpu_encode_p95_us = (!gpu_encodes.is_empty()).then(|| p95(&gpu_encodes));
	let bitstream_ready_p50_us = (!bitstream_readies.is_empty()).then(|| p50(&bitstream_readies));
	let bitstream_ready_p95_us = (!bitstream_readies.is_empty()).then(|| p95(&bitstream_readies));

	tracing::debug!(
		frames = n,
		fps = (n as f64 / elapsed_s).round() as u32,
		bitrate_kbps = ((bytes_total * 8) as f64 / elapsed_s / 1000.0).round() as u64,
		keyframes,
		total_p50_us = p50(&totals),
		total_p95_us = p95(&totals),
		total_p99_us = p99(&totals),
		channel_p50_us = p50(&channel),
		channel_p95_us = p95(&channel),
		channel_p99_us = p99(&channel),
		import_p50_us = p50(&imports),
		convert_p50_us = p50(&converts),
		convert_p95_us = p95(&converts),
		convert_p99_us = p99(&converts),
		submit_p50_us = p50(&submits),
		submit_p95_us = p95(&submits),
		submit_p99_us = p99(&submits),
		consumer_queue_p50_us = p50(&consumer_queues),
		consumer_queue_p95_us = p95(&consumer_queues),
		consumer_queue_p99_us = p99(&consumer_queues),
		encode_wait_p50_us = p50(&encode_waits),
		encode_wait_p95_us = p95(&encode_waits),
		encode_wait_p99_us = p99(&encode_waits),
		gpu_timing_samples = gpu_encodes.len(),
		gpu_encode_p50_us = ?gpu_encode_p50_us,
		gpu_encode_p95_us = ?gpu_encode_p95_us,
		bitstream_ready_p50_us = ?bitstream_ready_p50_us,
		bitstream_ready_p95_us = ?bitstream_ready_p95_us,
		packetize_p50_us = p50(&packetizes),
		send_p50_us = p50(&sends),
		send_p95_us = p95(&sends),
		send_p99_us = p99(&sends),
		"Frame latency summary (μs)"
	);
}

/// Per-submitted-frame context handed to the packet consumer thread, in
/// submission order. It travels alongside that frame's [`EncodeFuture`] in a
/// single [`ConsumerMessage::Frame`], so the packet is paired with its context by
/// construction — there is no separate packet channel to keep in lockstep.
struct FrameContext {
	/// When the source frame was captured.
	created_at: std::time::Instant,
	/// The compositor's sequence number for the source frame; `None` for the
	/// IDR re-encode path, which has no captured frame.
	sequence: Option<u64>,
	/// Pre-encode latency breakdown measured on the encoding thread.
	channel_wait: std::time::Duration,
	import: std::time::Duration,
	convert: std::time::Duration,
	/// Time spent in the `submit` call.
	submit: std::time::Duration,
	/// When the submit call returned and the frame was ready for packet consumption.
	submitted_at: std::time::Instant,
	/// Whether HDR SEI metadata should be injected on key frames for this frame
	/// (true when the encoder's VUI is BT.2020+PQ).
	inject_hdr: bool,
	/// HDR mastering metadata to inject, if any.
	hdr_metadata: Option<HdrMetadata>,
	/// The color space of the captured frame; `None` for the IDR re-encode path.
	content: Option<FrameColorSpace>,
}

/// An encoded frame in flight, with the context from its submission.
type SubmittedFrame = (FrameContext, EncodeFuture);

/// Decrements the in-flight frame counter when dropped: on an early `continue`
/// in the consumer loop, or when the network sender takes the frame's
/// [`PacketBatch`]. This is what tells the encoding thread a slot has freed up
/// so it can stop dropping new captures.
struct InFlightGuard(Arc<AtomicUsize>);

/// A packetized frame on its way to the network sender.
///
/// The batch holds its frame's in-flight slot while it waits in the send queue,
/// so a sender that stops draining (a stalled socket) fills the slots and the
/// encoding thread drops new captures instead of frames queueing up behind the
/// socket. The sender releases the slot as it takes the batch with
/// [`PacketBatch::into_shards`], before writing it to the socket, so the send
/// itself (e.g. waiting for a large keyframe to fit in the socket buffer) and
/// the data still queued in the socket afterwards don't count against the
/// limit.
pub(super) struct PacketBatch {
	shards: ShardBatch,
	in_flight: InFlightGuard,
}

impl PacketBatch {
	/// Release the frame's in-flight slot and return its shards.
	///
	/// The slot is released here, when the sender dequeues the batch, not when
	/// its shards are sent or leave the host.
	pub(super) fn into_shards(self) -> ShardBatch {
		drop(self.in_flight);
		self.shards
	}

	/// A batch holding a new in-flight slot, as the consumer would produce it.
	#[cfg(test)]
	pub(super) fn holding_slot(in_flight: &Arc<AtomicUsize>, shards: ShardBatch) -> Self {
		in_flight.fetch_add(1, Ordering::Relaxed);
		Self {
			shards,
			in_flight: InFlightGuard(in_flight.clone()),
		}
	}
}

impl Drop for InFlightGuard {
	fn drop(&mut self) {
		self.0.fetch_sub(1, Ordering::Relaxed);
	}
}

/// Wire state is local to a generation. Moonlight expects a new stream's first
/// frame index to be 1, with an RTP sequence and presentation clock starting at zero.
pub(super) struct FramePacketizer {
	packetizer: Packetizer,
	frame_number: u32,
	sequence_number: u32,
	rtp_epoch: Option<Instant>,
	next_rtp_ticks: u64,
}

impl FramePacketizer {
	pub(super) fn new(packetizer: Packetizer) -> Self {
		Self {
			packetizer,
			frame_number: 0,
			sequence_number: 0,
			rtp_epoch: None,
			next_rtp_ticks: 0,
		}
	}

	fn has_sent_frame(&self) -> bool {
		self.frame_number != 0
	}

	pub(super) fn packetize(
		&mut self,
		data: &[u8],
		frame_type: FrameType,
		created_at: Instant,
		latency_100us: u16,
		ctx: &VideoStreamContext,
		config: &VideoStreamConfig,
	) -> Result<(u32, ShardBatch), ()> {
		// Clients take the 90 kHz RTP timestamp as the presentation time. A
		// re-encode can be stamped after the capture time of the next frame.
		let epoch = *self.rtp_epoch.get_or_insert(created_at);
		let elapsed = created_at.duration_since(epoch);
		let ticks = ((elapsed.as_micros() * 90_000 / 1_000_000) as u64).max(self.next_rtp_ticks);
		let first = !self.has_sent_frame();
		self.next_rtp_ticks = ticks + 1;
		self.frame_number += 1;

		match self.packetizer.packetize(
			data,
			frame_type,
			ctx.packet_size,
			ctx.minimum_fec_packets,
			config.fec_percentage,
			self.frame_number,
			&mut self.sequence_number,
			ticks as u32,
			latency_100us,
		) {
			Ok(shards) => Ok((self.frame_number, shards)),
			Err(()) => {
				// No packet from this generation has reached the client yet.
				if first {
					self.frame_number = 0;
					self.sequence_number = 0;
					self.rtp_epoch = None;
					self.next_rtp_ticks = 0;
				}
				Err(())
			},
		}
	}
}

/// Downsample captures only when the retained compositor can run faster than
/// the negotiated stream. Stay on the target schedule instead of accumulating
/// each capture's timing jitter, but don't catch up a backlog after an idle gap.
struct CaptureCadence {
	interval: Duration,
	slack: Duration,
	next: Option<Instant>,
}

impl CaptureCadence {
	fn new(capture_fps: u32, fps: u32, interval: Duration) -> Option<Self> {
		(capture_fps > fps).then(|| Self {
			interval,
			slack: Duration::from_secs_f64(0.5 / f64::from(capture_fps)),
			next: None,
		})
	}

	fn accepts(&mut self, captured_at: Instant) -> bool {
		let earliest_slot = self.next.unwrap_or(captured_at);
		if captured_at + self.slack < earliest_slot {
			return false;
		}
		self.next = Some(earliest_slot.max(captured_at - self.slack) + self.interval);
		true
	}
}

/// Packet consumer thread: for each submitted frame, awaits its
/// [`EncodeFuture`] to get the encoded packet, injects HDR SEI if needed,
/// packetizes and sends it over the network channel, and emits latency stats.
///
/// Running this off the encoding thread is what realizes the latency win of the
/// asynchronous encoder: a packet is resolved by pixelforge's readback thread at
/// roughly the GPU encode time and sent here immediately, rather than waiting for
/// the encoding thread's next loop iteration. Pairing is implicit — each message
/// carries both the context and the future for the same frame.
///
/// `in_flight` counts frames submitted but not yet taken by the network sender;
/// each frame's [`InFlightGuard`] decrements it when the sender takes the frame
/// (or when the frame is dropped here), which is the signal the encoding
/// thread's drop-to-catch-up gate reads.
#[allow(clippy::too_many_arguments)]
async fn run_packet_consumer(
	mut ctx_rx: mpsc::Receiver<SubmittedFrame>,
	packet_tx: mpsc::Sender<PacketBatch>,
	stats: StreamStoppedOnDrop,
	in_flight: Arc<AtomicUsize>,
	idr_tx: broadcast::Sender<()>,
	mut packetizer: FramePacketizer,
	ctx: VideoStreamContext,
	config: VideoStreamConfig,
) {
	let mut latency_samples: Vec<LatencySample> = Vec::with_capacity(512);
	let mut last_summary_time = std::time::Instant::now();
	let frame_interval_us = 1_000_000 / ctx.fps as u128;
	// Driven by the message channel: one entry per submitted frame. When the
	// encoding thread drops its sender, this loop ends after the last frame.
	while let Some((frame_context, future)) = ctx_rx.recv().await {
		// The slot travels with the batch below; all early exits release it here.
		let in_flight_slot = InFlightGuard(in_flight.clone());

		let t_wait_started = std::time::Instant::now();
		let consumer_queue_dur = t_wait_started.saturating_duration_since(frame_context.submitted_at);
		let mut packet = match future.await {
			Ok(packet) => packet,
			Err(e) => {
				// A frame whose bitstream overflowed its destination buffer was
				// truncated and is not sent. Request an IDR so the next frame is a
				// keyframe, keeping the client's reference chain decodable.
				if let PixelForgeError::BufferOverflow { written, capacity } = &e {
					tracing::warn!(
						written,
						capacity,
						"Encoded frame overflowed bitstream buffer; requesting IDR"
					);
					let _ = idr_tx.send(());
				} else {
					tracing::warn!("Failed to read back encoded frame: {e}");
					if !packetizer.has_sent_frame() {
						let _ = idr_tx.send(());
					}
				}
				continue;
			},
		};
		let t_packet_ready = std::time::Instant::now();
		let encode_wait_dur = t_packet_ready.saturating_duration_since(frame_context.submitted_at);

		// Inject HDR metadata into the bitstream on key frames, but only when
		// encoding as BT.2020+PQ. SDR frames encoded as BT.709 carry no HDR SEI.
		if packet.is_key_frame() && frame_context.inject_hdr {
			// Fall back to default HDR10 metadata when the content provides none
			// (e.g. scRGB swapchains carry no mastering metadata), staying
			// consistent with the control-stream HDR metadata.
			let m = frame_context.hdr_metadata.unwrap_or_else(HdrMetadata::fallback);
			packet.data = hdr_sei::inject_hdr_metadata(&packet.data, &m, ctx.video_format);
		}

		// GPU hardware time and submit-to-bitstream-ready wall time come from
		// pixelforge. Neither is interchangeable with the async future's wait.
		let gpu_encode = packet.stats.gpu_time_ns.map(std::time::Duration::from_nanos);
		let bitstream_ready = Some(std::time::Duration::from_nanos(packet.stats.wall_latency_ns));
		let encoded_bytes = packet.data.len();
		let is_key_frame = packet.is_key_frame();

		// A fresh encoder starts with an IDR. If readback of that IDR failed,
		// don't hand the client predicted frames it cannot decode.
		if !packetizer.has_sent_frame() && !is_key_frame {
			let _ = idr_tx.send(());
			continue;
		}

		let t_start = std::time::Instant::now();
		// Capture-to-packetization latency in 100µs units, clamped to u16::MAX
		// (~6.55s). Clamp in the wider u128 *before* the cast, otherwise the
		// `as u16` would wrap for latencies > ~6.55s and report a tiny value.
		let processing_latency = t_start.duration_since(frame_context.created_at);
		let latency_100us = (processing_latency.as_micros() / 100).min(u16::MAX as u128) as u16;

		let frame_type = if is_key_frame {
			FrameType::Key
		} else if packet.is_recovery_frame {
			FrameType::Recovery
		} else {
			FrameType::Predicted
		};
		let (frame_number, shards) = match packetizer.packetize(
			&packet.data,
			frame_type,
			frame_context.created_at,
			latency_100us,
			&ctx,
			&config,
		) {
			Ok(shards) => shards,
			// Drop just this frame rather than tearing down the session. For
			// an established stream the client sees a gap and requests an IDR.
			Err(()) => {
				tracing::warn!("Failed to packetize encoded frame");
				if !packetizer.has_sent_frame() {
					let _ = idr_tx.send(());
				}
				continue;
			},
		};

		let t_packetized = std::time::Instant::now();

		// A closed packet channel means the network sender is gone, i.e. the
		// session is already tearing down. Stop the consumer; the encoding thread
		// then sees its `frame_ctx_tx` fail and exits too, which drops the
		// video-pipeline shutdown token and tears the session down.
		let batch = PacketBatch {
			shards,
			in_flight: in_flight_slot,
		};
		if packet_tx.send(batch).await.is_err() {
			tracing::debug!("Couldn't send packet batch, video packet channel closed.");
			break;
		}

		let t_sent = std::time::Instant::now();

		let packetize_dur = t_packetized - t_start;
		let send_dur = t_sent - t_packetized;
		let total = t_sent.duration_since(frame_context.created_at);

		// Warn on spike frames (total > frame interval) so they stand out.
		if config.log_frame_spikes && total.as_micros() > frame_interval_us {
			tracing::warn!(
				total_us = total.as_micros() as u64,
				channel_us = frame_context.channel_wait.as_micros() as u64,
				import_us = frame_context.import.as_micros() as u64,
				convert_us = frame_context.convert.as_micros() as u64,
				submit_us = frame_context.submit.as_micros() as u64,
				consumer_queue_us = consumer_queue_dur.as_micros() as u64,
				encode_wait_us = encode_wait_dur.as_micros() as u64,
				packetize_us = packetize_dur.as_micros() as u64,
				send_us = send_dur.as_micros() as u64,
				encoded_bytes,
				is_key_frame,
				sequence = frame_context.sequence,
				"SPIKE: frame latency exceeds {}us",
				frame_interval_us
			);
		}

		latency_samples.push(LatencySample {
			channel_wait: frame_context.channel_wait,
			import: frame_context.import,
			convert: frame_context.convert,
			submit: frame_context.submit,
			consumer_queue: consumer_queue_dur,
			encode_wait: encode_wait_dur,
			gpu_encode,
			bitstream_ready,
			packetize: packetize_dur,
			send: send_dur,
			total,
			encoded_bytes,
			is_key_frame,
		});

		stats.0.frame(FrameStats {
			frame_number,
			captured_at: std::time::SystemTime::now() - frame_context.created_at.elapsed(),
			content: frame_context.content,
			channel_wait: frame_context.channel_wait,
			import: frame_context.import,
			convert: frame_context.convert,
			submit: frame_context.submit,
			consumer_queue: consumer_queue_dur,
			encode_wait: encode_wait_dur,
			gpu_encode,
			bitstream_ready,
			packetize: packetize_dur,
			send: send_dur,
			total,
			encoded_bytes,
			is_key_frame,
		});

		// Periodic summary every 5 seconds.
		if last_summary_time.elapsed() >= std::time::Duration::from_secs(5) && !latency_samples.is_empty() {
			log_latency_summary(&latency_samples, last_summary_time.elapsed());
			latency_samples.clear();
			last_summary_time = std::time::Instant::now();
		}
	}
}

impl VideoPipeline {
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		frames: CompositorFrames,
		capture_fps: u32,
		gpu: GpuContext,
		config: VideoStreamConfig,
		context: VideoStreamContext,
		keys_rx: SessionKeysReceiver,
		packet_tx: mpsc::Sender<PacketBatch>,
		idr_tx: broadcast::Sender<()>,
		idr_frame_request_rx: broadcast::Receiver<()>,
		invalidate_request_rx: broadcast::Receiver<(u32, u32)>,
		nonce_counter: Arc<AtomicU64>,
		stop_session_manager: ShutdownManager<SessionShutdownReason>,
		hdr_metadata_tx: watch::Sender<HdrModeState>,
		start: StartWaiter,
		stats_tx: StatsSender,
	) -> Result<Self, ()> {
		tracing::debug!("Initializing video pipeline.");

		let inner = VideoPipelineInner {
			capture_fps,
			nonce_counter,
			gpu,
			config,
			context,
			keys_rx,
		};

		// Capture the main (multi-threaded) runtime handle so the OS-threaded
		// encoding loop can spawn the async packet consumer as a task on it. `new`
		// is called from within the session runtime, so `current()` is valid here.
		let runtime = tokio::runtime::Handle::current();
		let delay_stop = stop_session_manager
			.delay_shutdown_token()
			.map_err(|e| tracing::error!("Can't start the video pipeline: {e:?}"))?;

		std::thread::Builder::new()
			.name("video-pipeline".to_string())
			.spawn(move || {
				let _delay_stop = delay_stop;
				inner.run(
					runtime,
					frames,
					packet_tx,
					idr_tx,
					idr_frame_request_rx,
					invalidate_request_rx,
					stop_session_manager,
					hdr_metadata_tx,
					start,
					stats_tx,
				);
			})
			.map_err(|e| tracing::error!("Failed to start video pipeline thread: {e}"))?;

		Ok(Self {})
	}
}

/// Tells statistics readers the stream ended when dropped.
struct StreamStoppedOnDrop(StatsSender);

impl Drop for StreamStoppedOnDrop {
	fn drop(&mut self) {
		self.0.stopped();
	}
}

struct VideoPipelineInner {
	capture_fps: u32,
	nonce_counter: Arc<AtomicU64>,
	gpu: GpuContext,
	config: VideoStreamConfig,
	context: VideoStreamContext,
	keys_rx: SessionKeysReceiver,
}

impl VideoPipelineInner {
	#[allow(clippy::too_many_arguments)]
	fn run(
		self,
		runtime: tokio::runtime::Handle,
		frames: CompositorFrames,
		packet_tx: mpsc::Sender<PacketBatch>,
		idr_tx: broadcast::Sender<()>,
		idr_frame_request_rx: broadcast::Receiver<()>,
		invalidate_request_rx: broadcast::Receiver<(u32, u32)>,
		stop_session_manager: ShutdownManager<SessionShutdownReason>,
		hdr_metadata_tx: watch::Sender<HdrModeState>,
		mut start: StartWaiter,
		stats_tx: StatsSender,
	) {
		tracing::debug!("Starting video pipeline.");

		// Trigger session shutdown if we exit unexpectedly.
		let _session_stop_token =
			stop_session_manager.trigger_shutdown_token(SessionShutdownReason::VideoEncoderStopped);

		// Wait for the start signal before entering the encode loop.
		let rt = tokio::runtime::Builder::new_current_thread()
			.enable_all()
			.build()
			.expect("Failed to build tokio runtime for video pipeline");
		if !rt.block_on(start.wait(&stop_session_manager)) {
			tracing::debug!("Video pipeline stopped before start signal.");
			return;
		}

		// Create the encoder.
		let (context, encoder) = match self.create_encoder() {
			Ok(result) => result,
			Err(e) => {
				tracing::error!("Failed to create video encoder: {e}");
				return;
			},
		};
		if stop_session_manager.is_shutdown_triggered() {
			return;
		}

		// Stats readers learn the parameters before the first frame, and the
		// consumer emits Stopped after its last frame. Both workers delay
		// shutdown, so only then can the next generation start.
		stats_tx.started(StreamInfo::from(&self.context));
		let stats = StreamStoppedOnDrop(stats_tx);

		// Start the capture and encoding loop.
		if let Err(e) = self.run_encoding_loop(
			runtime,
			frames,
			context,
			encoder,
			packet_tx,
			idr_tx,
			idr_frame_request_rx,
			invalidate_request_rx,
			stop_session_manager,
			hdr_metadata_tx,
			stats,
		) {
			tracing::error!("Video encoding loop failed: {e}");
		}

		tracing::debug!("Video pipeline stopped.");
	}

	fn create_encoder(&self) -> Result<(VideoContext, Encoder), String> {
		let ctx = &self.context;

		// A new video context (its own device) on the process-wide instance (see
		// `crate::gpu`).
		let context = self
			.gpu
			.video_context()
			.map_err(|e| format!("Failed to create video context: {e}"))?;

		// Convert our video format to pixelforge's codec.
		let codec = match ctx.video_format {
			VideoFormat::H264 => Codec::H264,
			VideoFormat::Hevc => Codec::H265,
			VideoFormat::Av1 => Codec::AV1,
		};

		// Convert pixel format.
		let pixel_format = match ctx.chroma_sampling_type {
			VideoChromaSampling::Yuv420 => ChromaFormat::Yuv420,
			VideoChromaSampling::Yuv444 => ChromaFormat::Yuv444,
		};

		// Convert bit depth based on dynamic range.
		let bit_depth = match ctx.dynamic_range {
			VideoDynamicRange::Sdr => pixelforge::BitDepth::Eight,
			VideoDynamicRange::Hdr => pixelforge::BitDepth::Ten,
		};

		// The stream's colour, and the RGB frames the compositor hands over. The
		// frames' real format and colour space are only known per frame; the
		// encoding loop reconfigures the encoder to them as they arrive.
		let (source, color) = session_colors(ctx.dynamic_range, ctx.full_range);

		// Create encode configuration.
		let config = match codec {
			Codec::H264 => EncodeConfig::h264(ctx.width, ctx.height),
			Codec::H265 => EncodeConfig::h265(ctx.width, ctx.height),
			Codec::AV1 => EncodeConfig::av1(ctx.width, ctx.height),
		}
		.with_encoder_tuning_mode(EncoderTuningMode::LowLatency)
		.with_chroma_format(pixel_format)
		.with_bit_depth(bit_depth)
		.with_source(Source::Rgb(RgbSource::new(RgbFormat::BGRx, source)))
		.with_color(color)
		// The 1-frame VBV below caps every frame either way.
		.with_rate_control(match self.config.rate_control {
			RateControlMode::Cbr => RateControl::Cbr {
				bitrate: ctx.bitrate as u32,
			},
			RateControlMode::Vbr => RateControl::Vbr {
				target: ctx.bitrate as u32,
				max: ctx.bitrate as u32,
			},
		})
		.with_frame_rate(ctx.fps, 1)
		.with_gop_size(0) // Infinite GOP, we'll request IDR frames manually
		.with_max_reference_frames(ctx.max_reference_frames)
		// Each reference predicted from costs encode time (RADV H.264 takes ~75%
		// longer with two than with one); invalidation recovers from the whole
		// window either way.
		.with_active_reference_frames(1)
		.with_virtual_buffer_size_ms(1000 / ctx.fps)
		.with_initial_virtual_buffer_size_ms(0);

		// The healthcheck doesn't advertise formats the GPU can't encode, so only a client
		// that requested one anyway should get the "unsupported" hint.
		let encoder = Encoder::new(context.clone(), config).map_err(|e| {
			let hint = if context.supports_encode_format(codec, pixel_format, bit_depth) {
				""
			} else {
				" (this GPU does not support this format, try disabling YUV 4:4:4 or HDR in the client)"
			};
			format!(
				"Failed to create {codec:?} encoder ({:?}, {:?}): {e}{hint}",
				ctx.chroma_sampling_type, ctx.dynamic_range
			)
		})?;

		Ok((context, encoder))
	}

	#[allow(clippy::too_many_arguments)]
	fn run_encoding_loop(
		&self,
		runtime: tokio::runtime::Handle,
		frames: CompositorFrames,
		context: VideoContext,
		mut encoder: Encoder,
		packet_tx: mpsc::Sender<PacketBatch>,
		idr_tx: broadcast::Sender<()>,
		mut idr_frame_request_rx: broadcast::Receiver<()>,
		mut invalidate_request_rx: broadcast::Receiver<(u32, u32)>,
		stop_session_manager: ShutdownManager<SessionShutdownReason>,
		hdr_metadata_tx: watch::Sender<HdrModeState>,
		stats: StreamStoppedOnDrop,
	) -> Result<(), String> {
		let ctx = &self.context;

		// The compositor is launched before the streams start.
		let Some(frames) = frames.get() else {
			return Err("The compositor was not launched".to_string());
		};

		let mut packetizer = Packetizer::new(ctx.encrypt_video, self.keys_rx.clone(), self.nonce_counter.clone());
		packetizer.warm_up(self.config.fec_percentage, ctx.minimum_fec_packets);
		let packetizer = FramePacketizer::new(packetizer);

		// The encoder is asynchronous: each `submit()` returns a future that
		// resolves with that frame's packet once the GPU finishes. We hand each
		// future, together with its per-frame context, to a dedicated consumer
		// thread (in submission order via `frame_ctx_tx`) which awaits it and
		// packetizes/sends — so packets go out as soon as the GPU finishes,
		// independent of this loop's cadence.
		//
		// `in_flight` counts frames submitted but not yet taken by the network
		// sender. The encoding thread reads it to decide when to drop new captures
		// (see `MAX_FRAMES_IN_FLIGHT`); a frame's slot is released when the sender
		// takes its batch, or when the consumer drops it. The channel
		// is sized just above that gate so it never actually blocks the producer —
		// admission is governed by the drop gate, not by the channel filling.
		let in_flight = Arc::new(AtomicUsize::new(0));
		let (frame_ctx_tx, frame_ctx_rx) = mpsc::channel::<SubmittedFrame>(MAX_FRAMES_IN_FLIGHT + 2);
		// Lets the drop warning tell a backed-up send queue (network) from frames
		// still encoding (GPU). Weak, so it doesn't keep the channel open.
		let send_queue = packet_tx.downgrade();
		let consumer = {
			let ctx = self.context.clone();
			let config = self.config.clone();
			let in_flight = in_flight.clone();
			let idr_tx = idr_tx.clone();
			let delay_stop = stop_session_manager
				.delay_shutdown_token()
				.map_err(|e| format!("Can't start the video packet consumer: {e:?}"))?;
			// The packetize/send half is pure async work (await the encode future,
			// packetize, send) with no blocking GPU calls, so it runs as a task on
			// moonshine's main runtime — a green thread multiplexed with the rest of
			// the session — rather than its own OS thread. Only this capture/encode
			// loop needs a real thread, as it blocks on the frame channel and the
			// Vulkan submit path.
			runtime.spawn(async move {
				let _delay_stop = delay_stop;
				run_packet_consumer(
					frame_ctx_rx,
					packet_tx,
					stats,
					in_flight,
					idr_tx,
					packetizer,
					ctx,
					config,
				)
				.await;
			})
		};

		// Rate-limits the drop-to-catch-up warning.
		let mut last_drop_warn: Option<std::time::Instant> = None;

		// Encoding loop - receives frames from compositor.
		let frame_interval = std::time::Duration::from_secs_f64(1.0 / ctx.fps as f64);
		let mut cadence = CaptureCadence::new(self.capture_fps, ctx.fps, frame_interval);
		let mut last_frame_time = std::time::Instant::now();

		// DMA-BUF importer for zero-copy capture (initialized on first DMA-BUF frame).
		let mut dmabuf_importer: Option<DmaBufImporter> = None;

		// Whether at least one frame has been encoded (for IDR re-encode).
		let mut has_encoded = false;

		// Reference frame invalidation maps the client's frame indices (what the
		// packet consumer stamps into outgoing packets) to pixelforge's display
		// order (`pts`). A fresh encoder and packetizer both count from zero,
		// with 1-based frame indices on the wire: display_order = frame_index - 1.
		// A rare readback drop can skew this, which at worst over-invalidates
		// into a (safe) IDR or suppresses reports until they pass the last answer.
		let mut submitted_count: u64 = 0;
		// The last invalidation answered, for `covered_by`.
		let mut last_invalidation: Option<(u64, u64)> = None;
		// An invalidation recovered and its recovery frame isn't submitted yet: the client
		// drops every frame until it arrives, so a static screen re-encodes the last frame.
		let mut pending_recovery = false;
		// A capture skipped for cadence or backpressure may leave an IDR request
		// pending; keep it eligible for re-encoding if the screen goes idle.
		let mut pending_idr = false;

		// Track the last HDR mode state sent to the control stream.
		let mut last_hdr_state = HdrModeState::new(ctx.dynamic_range == VideoDynamicRange::Hdr);

		let result = 'encode: loop {
			if stop_session_manager.is_shutdown_triggered() {
				break Ok(());
			}

			// Drain any pending IDR requests.
			loop {
				match idr_frame_request_rx.try_recv() {
					Ok(()) => {
						tracing::debug!("IDR frame requested");
						encoder.request_idr();
						pending_idr = true;
					},
					Err(broadcast::error::TryRecvError::Lagged(n)) => {
						tracing::debug!("IDR frame channel lagged by {n} messages.");
						encoder.request_idr();
						pending_idr = true;
					},
					Err(broadcast::error::TryRecvError::Closed | broadcast::error::TryRecvError::Empty) => break,
				}
			}

			// Drain any pending reference-frame-invalidation requests. Each takes
			// effect on the next submitted frame, which then predicts from a
			// surviving reference instead of forcing an IDR (the encoder still
			// falls back to an IDR when no reference survives).
			loop {
				match invalidate_request_rx.try_recv() {
					Ok((first, last)) => {
						let first_display_order = u64::from(first.saturating_sub(1));
						let last_display_order = u64::from(last.saturating_sub(1));
						if covered_by(last_invalidation, first_display_order, last_display_order) {
							tracing::trace!(
								"Skipping reference frame invalidation from client frame {first}, already answered"
							);
							continue;
						}
						tracing::debug!(
							"Reference frame invalidation requested from client frame {first} (display order {first_display_order})"
						);
						// The encoder falls back to an IDR when no reference from before the loss is left.
						if encoder.invalidate_reference_frames(first_display_order) {
							pending_recovery = true;
						} else {
							pending_idr = true;
						}
						last_invalidation = Some((first_display_order, submitted_count));
					},
					Err(broadcast::error::TryRecvError::Lagged(n)) => {
						// Missed some loss reports; force an IDR to be safe.
						tracing::debug!("Invalidation channel lagged by {n} messages; forcing IDR.");
						encoder.request_idr();
						pending_idr = true;
					},
					Err(broadcast::error::TryRecvError::Closed | broadcast::error::TryRecvError::Empty) => break,
				}
			}

			// Try to receive a frame from compositor (with timeout).
			let received_frame = match frames.recv_timeout(frame_interval) {
				Ok(frame) => Some(frame),
				Err(ferriscope::RecvTimeoutError::Timeout) => {
					// No frame received within timeout.
					// If the client waits for an IDR or a recovery frame, re-encode the
					// last frame, which the encoder still holds in its input image.
					if (pending_idr || pending_recovery) && has_encoded {
						tracing::debug!("Re-encoding last frame for IDR or recovery (no re-import)");
						// Use current time as created_at for the re-encoded IDR (no actual frame).
						let now = std::time::Instant::now();
						pending_idr = false;
						match encoder.repeat() {
							Ok(future) => {
								let submitted_at = std::time::Instant::now();
								let inject_hdr = is_hdr(&encoder);
								let frame_context = FrameContext {
									created_at: now,
									sequence: None,
									channel_wait: std::time::Duration::ZERO,
									import: std::time::Duration::ZERO,
									convert: std::time::Duration::ZERO,
									submit: submitted_at.duration_since(now),
									submitted_at,
									inject_hdr,
									hdr_metadata: last_hdr_state.metadata,
									content: None,
								};
								// Count this frame in flight; released when the network sender takes
								// it or the consumer drops it.
								in_flight.fetch_add(1, Ordering::Relaxed);
								submitted_count += 1;
								pending_recovery = false;
								let _ = frame_ctx_tx.blocking_send((frame_context, future));
							},
							Err(PixelForgeError::DeviceLost) => {
								break 'encode Err(
									"Failed to re-encode frame for IDR request: the GPU was lost".to_string()
								);
							},
							Err(e) => {
								tracing::warn!("Failed to re-encode frame for IDR or recovery: {e}");
								// Don't retry every interval: make the next captured frame an IDR.
								if pending_recovery {
									pending_recovery = false;
									encoder.request_idr();
								}
							},
						}
					}
					if !pending_idr && last_frame_time.elapsed() > std::time::Duration::from_secs(5) {
						tracing::warn!("No frames received for 5 seconds");
						last_frame_time = std::time::Instant::now();
					}
					None
				},
				Err(ferriscope::RecvTimeoutError::Disconnected) => {
					tracing::debug!("Frame channel disconnected - compositor stopped");
					let _ = stop_session_manager.trigger_shutdown(SessionShutdownReason::CompositorStopped);
					break Ok(());
				},
			};

			if let Some(frame) = received_frame {
				// Drop-to-catch-up: if the consumer (packetize/network-send) is too
				// far behind, skip this capture *before* encoding it so the stream
				// stays realtime instead of accumulating latency. Dropping pre-encode
				// keeps the encoded P-frame chain valid (a skipped frame never enters
				// the encoder's reference state). The IDR re-encode path is never
				// gated — the client needs that keyframe.
				if in_flight.load(Ordering::Relaxed) >= MAX_FRAMES_IN_FLIGHT {
					if last_drop_warn.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(1)) {
						// Batches waiting for the sender hold slots only until it takes them,
						// so any queued batch means the socket isn't draining.
						let queued = send_queue.upgrade().map_or(0, |tx| tx.max_capacity() - tx.capacity());
						if queued > 0 {
							tracing::warn!(
								queued,
								"Video send backpressure: the network isn't keeping up; dropping \
								 captured frames before encode to stay realtime."
							);
						} else {
							tracing::warn!(
								"Video encode backpressure: the encoder isn't keeping up; dropping \
								 captured frames before encode to stay realtime."
							);
						}
						last_drop_warn = Some(std::time::Instant::now());
					}
					continue;
				}
				if cadence
					.as_mut()
					.is_some_and(|cadence| !cadence.accepts(frame.info.captured_at))
				{
					continue;
				}

				let t1_received = std::time::Instant::now();

				// ferriscope only delivers DMA-BUF frames to embedders.
				let ferriscope::FrameData::DmaBuf(dmabuf_frame) = &frame.data;
				let dmabuf = &dmabuf_frame.dmabuf;
				let fourcc = dmabuf.format().code as u32;
				let modifier = u64::from(dmabuf.format().modifier);
				let (width, height) = (dmabuf.width(), dmabuf.height());
				let captured_at = frame.info.captured_at;
				let sequence = frame.info.sequence;

				tracing::trace!(
					"Received frame: format=0x{fourcc:08X}, modifier={modifier:#x}, {width}x{height}, planes={}",
					dmabuf.num_planes()
				);

				let importer = match &mut dmabuf_importer {
					Some(imp) => imp,
					None => match DmaBufImporter::new(context.clone()) {
						Ok(imp) => {
							dmabuf_importer = Some(imp);
							dmabuf_importer.as_mut().unwrap()
						},
						Err(e) => {
							tracing::warn!("Failed to create DMA-BUF importer: {e}");
							continue;
						},
					},
				};

				// Wait for the compositor's render to finish before reading the
				// buffer. `None` means it already had.
				if let Some(fence) = &dmabuf_frame.acquire_fence
					&& let Err(e) = fence::wait_sync_file(fence.as_fd(), fence::FENCE_TIMEOUT)
				{
					tracing::warn!("Dropping frame: {e}");
					continue;
				}

				// The planes borrow the frame's fds, which stay open until the
				// frame is dropped.
				let mut planes_buf = [DmaBufPlane {
					fd: 0,
					offset: 0,
					stride: 0,
					modifier: 0,
				}; 4];
				let mut plane_count = 0;
				for ((fd, offset), stride) in dmabuf.handles().zip(dmabuf.offsets()).zip(dmabuf.strides()).take(4) {
					planes_buf[plane_count] = DmaBufPlane {
						fd: fd.as_raw_fd(),
						offset,
						stride,
						modifier,
					};
					plane_count += 1;
				}
				let planes = &planes_buf[..plane_count];

				// Determine Vulkan format and input format from the frame's DRM fourcc.
				let (frame_input_format, import_vk_format) = drm_fourcc_to_input(fourcc);

				// Import the DMA-BUF (reuses cached VkImage for known DMA-BUF fds).
				// The `Arc` keeps the image alive while the encoder reads it.
				let import = match importer.import_or_reuse(planes[0].fd, width, height, import_vk_format, planes) {
					Ok(import) => import,
					Err(e) => {
						tracing::warn!("Failed to import DMA-BUF: {e}");
						continue;
					},
				};

				let t2_imported = std::time::Instant::now();

				// Tell the encoder what this frame is, and what the stream says about
				// it. Both can change between frames (the render path, or HDR content
				// starting and stopping), and the encoder only does work when they do.
				// In an HDR session, SDR frames are encoded as BT.709 and HDR frames as
				// BT.2020+PQ, with dynamic VUI switching in the encoder. scRGB's and
				// SDR's reference whites are pixelforge's per-source defaults (80 and
				// 203 nits).
				let content = color_space(&frame.info.colour);
				let (frame_source, stream_color) = match ctx.dynamic_range {
					VideoDynamicRange::Sdr => session_colors(ctx.dynamic_range, ctx.full_range),
					VideoDynamicRange::Hdr => frame_colors(content, ctx.full_range),
				};
				let was_hdr = is_hdr(&encoder);
				if let Err(e) = encoder.reconfigure(
					Source::Rgb(RgbSource::new(frame_input_format, frame_source)),
					stream_color,
				) {
					break Err(format!("Failed to reconfigure the encoder for the frame: {e}"));
				}
				if is_hdr(&encoder) != was_hdr {
					tracing::debug!(
						"Switched encoder to {stream_color:?} for a {frame_input_format:?} {frame_source:?} frame (conversion \
						 {:?})",
						encoder.conversion()
					);
				}

				let frame_hdr_metadata = hdr_metadata(&frame.info.colour);

				// Forward HDR mode state changes to the control stream.
				// In HDR sessions, `enabled` reflects whether the current
				// frame is encoded as BT.2020+PQ (true) or BT.709 (false).
				if ctx.dynamic_range == VideoDynamicRange::Hdr {
					let hdr_enabled = is_hdr(&encoder);
					let new_state = HdrModeState {
						enabled: hdr_enabled,
						metadata: frame_hdr_metadata,
					};
					if new_state != last_hdr_state {
						if new_state.enabled {
							match new_state.metadata {
								Some(m) => tracing::info!(
									"Content switched to HDR (BT.2020/PQ): maxCLL {} nits, maxFALL {} nits",
									m.max_cll,
									m.max_fall
								),
								None => tracing::info!("Content switched to HDR (BT.2020/PQ)"),
							}
						} else {
							tracing::info!("Content switched to SDR (BT.709)");
						}
						last_hdr_state = new_state.clone();
						let _ = hdr_metadata_tx.send(new_state);
					}
				}

				// The colour conversion runs on the GPU as part of the submit below;
				// `convert` covers telling the encoder what the frame is.
				let t3_converted = std::time::Instant::now();

				// Encode the frame. This is asynchronous: it submits the conversion
				// and the encode without blocking; the encoded packet is produced by
				// the readback thread and handled by the consumer thread.
				let encode_result = encoder.submit(Frame::image(import.source()));

				let t4_encoded = std::time::Instant::now();

				match encode_result {
					Ok(future) => {
						// Hand this frame's context plus its packet future to the
						// consumer thread, which awaits the future, injects HDR SEI if
						// needed, packetizes and sends it, and records stats.
						let inject_hdr = is_hdr(&encoder);
						let frame_context = FrameContext {
							created_at: captured_at,
							sequence: Some(sequence),
							channel_wait: t1_received.duration_since(captured_at),
							import: t2_imported.duration_since(t1_received),
							convert: t3_converted.duration_since(t2_imported),
							submit: t4_encoded.duration_since(t3_converted),
							submitted_at: t4_encoded,
							inject_hdr,
							hdr_metadata: last_hdr_state.metadata,
							content: Some(content),
						};
						// Count this frame in flight; released when the network sender takes
						// it or the consumer drops it.
						in_flight.fetch_add(1, Ordering::Relaxed);
						submitted_count += 1;
						pending_recovery = false;
						pending_idr = false;
						let consumer_gone = frame_ctx_tx.blocking_send((frame_context, future)).is_err();

						// The encoder reads the frame after `submit` returns. Once it
						// has, dropping the frame returns its buffer to the compositor.
						// The encode itself is already queued, so this wait does not
						// delay the frame; it is the colour conversion, well under a
						// frame interval.
						if let Err(e) = import.source().wait_released() {
							tracing::warn!("Failed to wait for the encoder to read a frame: {e}");
						}
						drop(frame);

						if consumer_gone {
							tracing::debug!("Packet consumer gone; stopping encoding loop.");
							break 'encode Ok(());
						}
					},
					Err(PixelForgeError::DeviceLost) => {
						break 'encode Err("Failed to encode frame: the GPU was lost".to_string());
					},
					Err(e) => tracing::warn!("Failed to encode frame: {e}"),
				}

				if !pending_idr {
					last_frame_time = std::time::Instant::now();
				}

				has_encoded = true;
			}
		};

		// Drain in-flight encodes (barrier so every packet reaches the channel),
		// then stop the consumer and wait for it to send the remaining packets.
		if let Err(e) = encoder.flush() {
			tracing::warn!("Failed to flush encoder: {e}");
		}
		drop(frame_ctx_tx);
		// Wait for the consumer task to drain and send the remaining packets. This
		// runs on a plain OS thread outside the runtime, so block on the task handle.
		if let Err(e) = runtime.block_on(consumer) {
			tracing::warn!("Packet consumer task panicked: {e:?}");
		}

		result
	}
}

#[cfg(test)]
mod tests {
	use super::{CaptureCadence, Duration, Instant, covered_by, drm_fourcc_to_input};
	use ash::vk;
	use pixelforge::RgbFormat;

	// DRM fourccs — kept in the test module to document the byte ordering
	// explicitly and guard against accidental typos in the production match.
	// Layout: fourcc_code(a, b, c, d) = a | b<<8 | c<<16 | d<<24.
	const ABGR8888: u32 = 0x34324241; // "AB24"
	const XBGR8888: u32 = 0x34324258; // "XB24"
	const ABGR2101010: u32 = 0x30334241; // "AB30"
	const XBGR2101010: u32 = 0x30334258; // "XB30"
	const ABGR16161616F: u32 = 0x48344241; // "AB4H"
	const XBGR16161616F: u32 = 0x48344258; // "XB4H"

	#[test]
	fn cadence_only_caps_a_faster_compositor() {
		let interval = Duration::from_millis(20);
		assert!(CaptureCadence::new(50, 50, interval).is_none());
		assert!(CaptureCadence::new(40, 50, interval).is_none());

		let mut cadence = CaptureCadence::new(100, 50, interval).unwrap();
		let first = Instant::now();
		for (ms, accepted) in [(0, true), (8, false), (18, true), (27, false), (39, true)] {
			assert_eq!(cadence.accepts(first + Duration::from_millis(ms)), accepted);
		}
		// Accept the first frame after idle, but skip extra captures rather
		// than trying to refill the target cadence's missed slots.
		let after_idle = first + Duration::from_secs(1);
		assert!(cadence.accepts(after_idle));
		assert!(!cadence.accepts(after_idle + Duration::from_millis(8)));
		assert!(cadence.accepts(after_idle + Duration::from_millis(19)));
	}

	#[test]
	fn covered_by_skips_only_repeats_of_the_answered_loss() {
		// The loss from display order 10 was answered by the frame at 20.
		let answered = Some((10, 20));
		assert!(covered_by(answered, 10, 10));
		assert!(covered_by(answered, 12, 19));
		// The answer itself was lost.
		assert!(!covered_by(answered, 12, 20));
		// Loss from before the answered one.
		assert!(!covered_by(answered, 8, 9));
		assert!(!covered_by(None, 1, 2));
	}

	#[test]
	fn drm_fourcc_abgr_xbgr_8bit_maps_to_rgba() {
		// 8-bit ABGR/XBGR share the same bit layout (alpha vs. padding) and
		// map to R8G8B8A8_UNORM.
		for fourcc in [ABGR8888, XBGR8888] {
			let (fmt, vk) = drm_fourcc_to_input(fourcc);
			assert_eq!(fmt, RgbFormat::RGBA);
			assert_eq!(vk, vk::Format::R8G8B8A8_UNORM);
		}
	}

	#[test]
	fn drm_fourcc_abgr_xbgr_10bit_maps_to_a2b10g10r10() {
		for fourcc in [ABGR2101010, XBGR2101010] {
			let (fmt, vk) = drm_fourcc_to_input(fourcc);
			assert_eq!(fmt, RgbFormat::ABGR2101010);
			assert_eq!(vk, vk::Format::A2B10G10R10_UNORM_PACK32);
		}
	}

	#[test]
	fn drm_fourcc_abgr_xbgr_fp16_maps_to_rgba16f() {
		// Regression guard for the XBGR16161616F (XB4H) case: NVIDIA's Wayland
		// WSI emits the X-variant for FP16 scRGB swapchains, and handling only
		// the A-variant caused 16F buffers to fall through to the 8-bit BGRx
		// fallback and produce garbled pixels.
		for fourcc in [ABGR16161616F, XBGR16161616F] {
			let (fmt, vk) = drm_fourcc_to_input(fourcc);
			assert_eq!(fmt, RgbFormat::RGBA16F);
			assert_eq!(vk, vk::Format::R16G16B16A16_SFLOAT);
		}
	}

	#[test]
	fn drm_fourcc_unknown_falls_back_to_bgrx() {
		// Anything outside the known set must fall back to the 8-bit BGRx
		// assumption — matches the `_ =>` arm in the production match.
		let (fmt, vk) = drm_fourcc_to_input(0xDEADBEEF);
		assert_eq!(fmt, RgbFormat::BGRx);
		assert_eq!(vk, vk::Format::B8G8R8A8_UNORM);
	}
}
