use std::net::IpAddr;
use std::sync::Arc;
use std::time::SystemTime;

use async_shutdown::ShutdownManager;
use tokio::sync::{Mutex, broadcast, watch};

use crate::ShutdownReason;
use crate::gpu::GpuContext;
use crate::session::InitializedSession;
use crate::session::SessionContext;
use crate::session::SessionKeyData;
use crate::session::SessionKeys;
use crate::session::SessionKeysSender;
use crate::session::SessionState;
use crate::session::compositor::CompositorConfig;
use crate::session::stream::AvPingToken;
use crate::session::stream::audio::AudioChannels;
use crate::session::stream::audio::AudioStreamConfig;
use crate::session::stream::audio::AudioStreamContext;
use crate::session::stream::control::ControlStreamConfig;
use crate::session::stream::video::VideoDynamicRange;
use crate::session::stream::video::VideoStreamConfig;
use crate::session::stream::video::VideoStreamContext;
use crate::session::stream::video::{StatsSender, StreamEvent, StreamInfo};

const SESSION_SHUTDOWN_TIMEOUT_SECS: u64 = 10;

/// Where a session is in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionPhase {
	/// Created; the compositor and application are not started yet.
	Initialized,
	/// The compositor and application run; the client has not started the streams yet.
	Launched,
	/// The streams run.
	Active,
}

/// A snapshot of the current session.
#[derive(Clone, Debug)]
pub struct SessionStatus {
	pub phase: SessionPhase,

	/// The session's application and the parameters the client asked for.
	pub context: SessionContext,

	/// Certificate fingerprint of the client that launched the session, if known.
	pub client: Option<String>,

	/// When the session was initialized.
	pub started: SystemTime,

	/// The video stream as negotiated with the client, once it runs.
	pub stream: Option<StreamInfo>,
}

/// The settings a session is created with. Changes apply to the next session.
#[derive(Clone, Debug)]
pub struct SessionSettings {
	pub compositor: CompositorConfig,
	pub video: VideoStreamConfig,
	pub audio: AudioStreamConfig,
	pub control: ControlStreamConfig,
	/// Time in seconds since last ping after which the stream closes.
	pub stream_timeout: u64,
	/// Whether to inhibit system sleep while a session is active.
	pub inhibit_sleep: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionShutdownReason {
	/// Session manager is shutting down.
	ManagerShutdown,
	/// The session was stopped by the user.
	UserStopped,
	/// The launched application exited.
	ApplicationStopped,
	/// Video packet handler stopped unexpectedly.
	VideoPacketHandlerStopped,
	/// Video encoder stopped unexpectedly.
	VideoEncoderStopped,
	/// Audio packet handler stopped unexpectedly.
	AudioPacketHandlerStopped,
	/// PulseAudio server stopped unexpectedly.
	PulseServerStopped,
	/// Audio encoder stopped unexpectedly.
	AudioEncoderStopped,
	/// Control stream stopped unexpectedly.
	ControlStreamStopped,
	/// Input handler stopped unexpectedly.
	InputHandlerStopped,
	/// Compositor stopped unexpectedly.
	CompositorStopped,
	/// The media workers were replaced by a resume; keep the game running.
	MediaReplaced,
	/// The negotiated media workers could not start.
	MediaStartFailed,
}

struct PendingResume {
	peer: IpAddr,
	keys: SessionKeyData,
	ping_token: AvPingToken,
	audio_layout: Option<(AudioChannels, u32)>,
}

struct SessionManagerInner {
	/// Configuration for the compositor.
	compositor_config: CompositorConfig,

	/// The Vulkan objects every session shares (see `crate::gpu`).
	gpu: GpuContext,

	/// Names the sessions' systemd unit and PulseAudio socket directory.
	session_name: String,

	/// Configuration for the video stream.
	video_config: VideoStreamConfig,

	/// Configuration for the audio stream.
	audio_config: AudioStreamConfig,

	/// Configuration for the control stream.
	control_config: ControlStreamConfig,

	/// Address to bind streams to.
	address: String,

	/// Time in seconds since last ping after which the stream closes.
	stream_timeout: u64,

	/// Whether to inhibit system sleep while a session is active.
	inhibit_sleep: bool,

	/// The currently active session, if any.
	session: Option<SessionState>,

	/// Shutdown manager for the active session, used to trigger session shutdown upon request.
	stop: ShutdownManager<SessionShutdownReason>,

	/// Sender for session keys, used to update keys from the webserver in different subsystems.
	///
	/// Subsystems that get updated are: video encoder, audio encoder and input handler.
	keys_tx: Option<SessionKeysSender>,

	/// Stream contexts received via RTSP ANNOUNCE, consumed by `start_session`.
	stream_contexts: Option<(VideoStreamContext, AudioStreamContext)>,
	pending_resume: Option<PendingResume>,

	/// Broadcast sender for per-frame encoding statistics.
	stats_tx: StatsSender,

	/// Watchdog task for monitoring unexpected session shutdowns.
	stop_watcher: Option<tokio::task::JoinHandle<()>>,

	/// Certificate fingerprint of the client that launched the current session, if known.
	session_client: Option<String>,

	/// When the current session was initialized.
	session_started: Option<SystemTime>,

	/// Shutdown manager for the entire application.
	shutdown: ShutdownManager<ShutdownReason>,

	/// Trigger token for the session manager's own shutdown trigger.
	///
	/// Used to trigger an application shutdown if the session manager stops unexpectedly.
	_trigger_token: async_shutdown::TriggerShutdownToken<ShutdownReason>,

	/// Delay token for the session manager's own shutdown trigger.
	///
	/// Used to delay shutdown until the session manager has cleaned up.
	_delay_token: async_shutdown::DelayShutdownToken<ShutdownReason>,
}

impl SessionManagerInner {
	fn reset_session(&mut self) {
		if let Some(handle) = self.stop_watcher.take() {
			handle.abort();
		}
		self.session = None;
		self.keys_tx = None;
		self.stream_contexts = None;
		self.pending_resume = None;
		self.session_client = None;
		self.session_started = None;
		self.stop = ShutdownManager::new();
	}
}

impl Drop for SessionManagerInner {
	fn drop(&mut self) {
		if let Some(handle) = self.stop_watcher.take() {
			handle.abort();
		}
		if self.session.is_some() {
			tracing::debug!("Stopping active session before shutdown.");
			let _ = self.stop.trigger_shutdown(SessionShutdownReason::ManagerShutdown);
			// Wait until shutdown completed.
			if let Ok(handle) = tokio::runtime::Handle::try_current() {
				handle.block_on(self.stop.wait_shutdown_complete());
			}
		}
	}
}

#[derive(Clone)]
pub struct SessionManager {
	inner: Arc<Mutex<SessionManagerInner>>,
	stats_tx: StatsSender,
}

impl SessionManager {
	/// `session_name` names the application's transient systemd unit
	/// (`<session_name>-session.service`) and the PulseAudio socket directory
	/// (`$XDG_RUNTIME_DIR/<session_name>/pulse`). The server uses `"moonshine"`.
	/// Another session manager in the same user session, such as
	/// `moonshine-bench`, needs a name of its own: starting a session stops a
	/// leftover unit of the same name and replaces the socket, which would end
	/// the server's session.
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		compositor_config: CompositorConfig,
		video_config: VideoStreamConfig,
		audio_config: AudioStreamConfig,
		control_config: ControlStreamConfig,
		address: String,
		stream_timeout: u64,
		inhibit_sleep: bool,
		session_name: &str,
		gpu: GpuContext,
		shutdown: ShutdownManager<ShutdownReason>,
	) -> Result<Self, ()> {
		let trigger_token = shutdown.trigger_shutdown_token(ShutdownReason::SessionManagerShutdown);
		let delay_token = shutdown.delay_shutdown_token().map_err(|e| {
			tracing::error!("Failed to create delay shutdown token: {e:?}");
		})?;

		let inner = SessionManagerInner {
			compositor_config,
			gpu,
			session_name: session_name.to_string(),
			video_config,
			audio_config,
			control_config,
			address,
			stream_timeout,
			inhibit_sleep,
			session: None,
			stop: ShutdownManager::new(),
			keys_tx: None,
			stream_contexts: None,
			pending_resume: None,
			stats_tx: StatsSender::new(256),
			stop_watcher: None,
			session_client: None,
			session_started: None,
			shutdown: shutdown.clone(),
			_trigger_token: trigger_token,
			_delay_token: delay_token,
		};

		let stats_tx = inner.stats_tx.clone();
		let inner = Arc::new(Mutex::new(inner));

		Ok(Self { inner, stats_tx })
	}

	/// Returns a receiver for video statistics: each stream's start with its parameters, every
	/// frame's timings, and the stream's end.
	///
	/// Call **before** `initialize_session()` to receive stats from the start.
	/// Multiple receivers can be created — each receives a copy of every message.
	pub fn bench_stats_receiver(&self) -> tokio::sync::broadcast::Receiver<StreamEvent> {
		self.stats_tx.subscribe()
	}

	/// Returns the video stream in progress, if any, and a receiver for every event after it.
	///
	/// Joining partway through a stream, the stream's start comes from the first and everything
	/// after it from the second, with nothing missed or repeated.
	pub fn join_stats(&self) -> (Option<StreamInfo>, broadcast::Receiver<StreamEvent>) {
		self.stats_tx.join()
	}

	/// Trigger the video and audio pipelines to start encoding.
	///
	/// In the normal flow, this is triggered by the control stream when the
	/// client sends `StartB`. Call this from external callers (e.g. bench binary)
	/// that have no Moonlight client. Must be called after `start_session()`.
	pub async fn trigger_streams_start(&self) {
		let inner = self.inner.lock().await;
		if let Some(SessionState::Active(active)) = &inner.session {
			active.trigger_streams_start();
		}
	}

	/// Set the video and audio stream contexts after receiving RTSP ANNOUNCE.
	pub async fn set_stream_context(
		&self,
		video_stream_context: VideoStreamContext,
		audio_stream_context: AudioStreamContext,
		peer: Option<IpAddr>,
	) -> Result<(), ()> {
		let mut guard = self.inner.lock().await;
		match guard.session.as_ref() {
			Some(SessionState::Launched(_)) => {},
			Some(SessionState::Active(active)) => {
				if !guard
					.pending_resume
					.as_ref()
					.is_some_and(|resume| Some(resume.peer) == peer)
				{
					tracing::warn!("Rejecting RTSP ANNOUNCE without a matching resume request");
					return Err(());
				}
				// The game and the Ferriscope output stay alive. Ferriscope cannot
				// resize the output or raise its refresh rate in place.
				let launch = active.context();
				if (video_stream_context.width, video_stream_context.height) != launch.resolution
					|| video_stream_context.fps > launch.refresh_rate
					|| (video_stream_context.dynamic_range == VideoDynamicRange::Hdr) != launch.hdr
				{
					guard.stream_contexts = None;
					tracing::warn!("Cannot change resolution, HDR mode, or exceed the launch refresh rate on resume");
					return Err(());
				}
			},
			_ => {
				tracing::warn!("Rejecting RTSP ANNOUNCE without a launched session");
				return Err(());
			},
		}
		guard.stream_contexts = Some((video_stream_context, audio_stream_context));
		Ok(())
	}

	/// A snapshot of the current session, if there is one.
	pub async fn status(&self) -> Option<SessionStatus> {
		let guard = self.inner.lock().await;
		let session = guard.session.as_ref()?;
		let phase = match session {
			SessionState::Initialized(_) => SessionPhase::Initialized,
			SessionState::Launched(_) => SessionPhase::Launched,
			SessionState::Active(_) => SessionPhase::Active,
		};
		let stream = self.stats_tx.join().0;
		let mut context = session.context().clone();
		if let Some(stream) = &stream {
			context.refresh_rate = stream.fps;
		}
		Some(SessionStatus {
			phase,
			context,
			client: guard.session_client.clone(),
			started: guard.session_started.unwrap_or_else(SystemTime::now),
			stream,
		})
	}

	/// Record which client launched the current session.
	pub(crate) async fn set_session_client(&self, fingerprint: Option<String>) {
		let mut guard = self.inner.lock().await;
		if guard.session.is_some() {
			guard.session_client = fingerprint;
		}
	}

	/// Replace the settings that sessions are created with. A running session keeps its settings.
	pub async fn update_settings(&self, settings: SessionSettings) {
		let mut guard = self.inner.lock().await;
		guard.compositor_config = settings.compositor;
		guard.video_config = settings.video;
		guard.audio_config = settings.audio;
		guard.control_config = settings.control;
		guard.stream_timeout = settings.stream_timeout;
		guard.inhibit_sleep = settings.inhibit_sleep;
	}

	/// Get the current session context if there is an active session; otherwise return `None`.
	pub async fn get_session_context(&self) -> Result<Option<SessionContext>, ()> {
		let guard = self.inner.lock().await;
		Ok(guard.session.as_ref().map(|s| s.context().clone()))
	}

	pub(crate) async fn av_ping_token(&self, peer: IpAddr) -> Option<AvPingToken> {
		let guard = self.inner.lock().await;
		match guard.pending_resume.as_ref() {
			Some(resume) if resume.peer == peer => Some(resume.ping_token),
			_ => guard.session.as_ref().map(SessionState::av_ping_token),
		}
	}

	pub(crate) async fn requested_audio_layout(&self, peer: IpAddr) -> Option<(AudioChannels, u32)> {
		let guard = self.inner.lock().await;
		guard
			.pending_resume
			.as_ref()
			.filter(|resume| resume.peer == peer)
			.and_then(|resume| resume.audio_layout)
			.or_else(|| {
				let ctx = guard.session.as_ref()?.context();
				Some((ctx.audio_channels, ctx.audio_channel_mask))
			})
	}

	/// Initialize a new session with the provided context.
	///
	/// The session is not launched until `launch_session` is called.
	pub async fn initialize_session(&self, mut context: SessionContext) -> Result<(), ()> {
		let mut guard = self.inner.lock().await;

		if guard.session.is_some() || guard.keys_tx.is_some() {
			tracing::warn!("Session already initialized, rejecting InitializeSession command.");
			return Err(());
		}

		// Extract the raw keys from context and create the watch channel.
		let session_keys = match context.keys {
			SessionKeys::Keys(data) => data,
			SessionKeys::Rx(_) => {
				tracing::error!("Session keys already initialized as a watch receiver");
				return Err(());
			},
		};
		let (tx, rx) = watch::channel(session_keys);
		context.keys = SessionKeys::Rx(rx);

		let compositor_config = guard.compositor_config.clone();
		let gpu = guard.gpu.clone();
		let session_name = guard.session_name.clone();
		let video_config = guard.video_config.clone();
		let audio_config = guard.audio_config.clone();
		let control_config = guard.control_config.clone();
		let address = guard.address.clone();
		let stop = guard.stop.clone();
		let stats_tx = guard.stats_tx.clone();
		let session = InitializedSession::new(
			compositor_config,
			video_config,
			audio_config,
			control_config,
			address,
			context,
			session_name,
			gpu,
			stop,
			stats_tx,
		)
		.await?;
		guard.session = Some(SessionState::Initialized(session));
		guard.session_started = Some(SystemTime::now());

		spawn_session_watchdog(&self.inner, &mut guard);
		tracing::info!("Session initialized successfully, waiting to be launched.");

		// Set the keys sender here so that it is not set if session initialization failed.
		guard.keys_tx = Some(tx);
		Ok(())
	}

	/// Launch the session by starting the compositor and application, but don't start streams until RTSP ANNOUNCE is received.
	pub async fn launch_session(&self) -> Result<(), ()> {
		let session = {
			let mut guard = self.inner.lock().await;
			match guard.session.take() {
				Some(SessionState::Initialized(session)) => session,
				Some(SessionState::Launched(launched)) => {
					guard.session = Some(SessionState::Launched(launched));
					tracing::warn!("LaunchSession rejected: session already launched");
					return Err(());
				},
				Some(SessionState::Active(active)) => {
					guard.session = Some(SessionState::Active(active));
					tracing::warn!("LaunchSession rejected: session already active");
					return Err(());
				},
				None => {
					tracing::warn!("LaunchSession rejected: no active session");
					return Err(());
				},
			}
		};

		tracing::info!("Launching session (starting compositor and app).");
		match session.launch().await {
			Ok(launched) => {
				let mut guard = self.inner.lock().await;
				guard.session = Some(SessionState::Launched(launched));
				tracing::info!("Session launched successfully, waiting for RTSP ANNOUNCE.");
				Ok(())
			},
			Err(()) => {
				let mut guard = self.inner.lock().await;
				guard.reset_session();
				tracing::error!("Failed to launch session, waiting for new session.");
				Err(())
			},
		}
	}

	/// Start negotiated media, or replace it after an authenticated resume.
	pub async fn start_session(&self, peer: Option<IpAddr>) -> Result<(), ()> {
		let mut guard = self.inner.lock().await;
		let stop = guard.stop.clone();
		if stop.is_shutdown_triggered() {
			return Err(());
		}
		let video_config = guard.video_config.clone();

		if matches!(guard.session, Some(SessionState::Active(_))) {
			let Some(resume) = &guard.pending_resume else {
				// Older clients send one PLAY for each track.
				return Ok(());
			};
			if Some(resume.peer) != peer {
				return Err(());
			}
			let Some((video, audio)) = guard.stream_contexts.take() else {
				return Err(());
			};
			let resume = guard.pending_resume.take().expect("checked pending resume");
			let keys_tx = guard.keys_tx.as_ref().expect("session keys not initialized").clone();
			let Some(SessionState::Active(active)) = &mut guard.session else {
				unreachable!()
			};
			tracing::info!(
				bitrate_kbps = video.bitrate / 1_000,
				fps = video.fps,
				"Restarting media for the resumed session."
			);
			return active
				.resume(
					video_config,
					video,
					audio,
					resume.keys,
					&keys_tx,
					resume.ping_token,
					&stop,
					std::time::Duration::from_secs(SESSION_SHUTDOWN_TIMEOUT_SECS),
				)
				.await;
		}

		if !matches!(guard.session, Some(SessionState::Launched(_))) {
			tracing::warn!("Rejecting RTSP PLAY without a launched session");
			return Err(());
		}
		let (video, audio) = guard.stream_contexts.take().ok_or_else(|| {
			tracing::warn!("Rejecting RTSP PLAY without negotiated stream settings");
		})?;
		let Some(SessionState::Launched(launched)) = guard.session.take() else {
			unreachable!()
		};

		tracing::info!("Starting session streams.");
		match launched
			.start(
				video_config,
				guard.stream_timeout,
				video,
				audio,
				stop.clone(),
				guard.inhibit_sleep,
			)
			.await
		{
			Ok(active) => {
				guard.session = Some(SessionState::Active(active));
				Ok(())
			},
			Err(()) => {
				let _ = stop.trigger_shutdown(SessionShutdownReason::MediaStartFailed);
				tracing::error!("Failed to start session streams.");
				Err(())
			},
		}
	}

	/// Stop the session and return to Uninitialized state.
	pub async fn stop_session(&self) -> Result<(), ()> {
		let (stop, shutdown) = {
			let mut guard = self.inner.lock().await;
			match guard.session {
				Some(_) => {},
				None => return Ok(()),
			}
			let stop = guard.stop.clone();
			let shutdown = guard.shutdown.clone();

			// Drop session first, which drops the Application.
			guard.reset_session();
			(stop, shutdown)
		};

		// Then trigger shutdown of the compositor & streams.
		let _ = stop.trigger_shutdown(SessionShutdownReason::UserStopped);

		wait_for_session_shutdown(&stop, &shutdown, SESSION_SHUTDOWN_TIMEOUT_SECS).await?;
		tracing::info!("Session stopped by user, waiting for new session.");
		Ok(())
	}

	/// Stage an authenticated resume. Apply it only once RTSP negotiates and plays.
	pub(crate) async fn prepare_resume(
		&self,
		peer: IpAddr,
		keys: SessionKeyData,
		audio_layout: Option<(AudioChannels, u32)>,
	) -> Result<(), ()> {
		let mut guard = self.inner.lock().await;
		if guard.stop.is_shutdown_triggered() || !matches!(guard.session, Some(SessionState::Active(_))) {
			tracing::warn!("Rejecting resume without an active session");
			return Err(());
		}
		let ping_token = AvPingToken::new()?;
		guard.stream_contexts = None;
		guard.pending_resume = Some(PendingResume {
			peer,
			keys,
			ping_token,
			audio_layout,
		});

		Ok(())
	}
}

/// Spawn a watchdog task to monitor the session for unexpected shutdowns.
fn spawn_session_watchdog(inner: &Arc<Mutex<SessionManagerInner>>, guard: &mut SessionManagerInner) {
	if guard.stop_watcher.is_some() {
		tracing::error!("Session watchdog already running, not spawning another.");
		return;
	}

	let inner = inner.clone();
	let stop = guard.stop.clone();
	let shutdown = guard.shutdown.clone();
	// Hold the application's shutdown open until the session has fully stopped:
	// the process exits once it completes, and exiting while the session's
	// threads are still tearing down (the encoder's Vulkan device, the
	// compositor's GL renderer) crashes in NVIDIA's exit handlers.
	let delay = shutdown
		.delay_shutdown_token()
		.inspect_err(|e| tracing::warn!("Failed to delay shutdown for the session: {e:?}"))
		.ok();
	let handle = tokio::spawn(async move {
		tokio::select! {
			reason = stop.wait_shutdown_triggered() => {
				if reason == SessionShutdownReason::UserStopped {
					tracing::info!("Session shutdown requested by user.");
				} else {
					tracing::warn!("Session stopped unexpectedly (reason: {reason:?}), waiting for new session.");
				}
			},
			_ = shutdown.wait_shutdown_triggered() => {
				tracing::debug!("Global shutdown triggered, stopping active session.");
				let _ = stop.trigger_shutdown(SessionShutdownReason::ManagerShutdown);
			},
		}

		// First drop the session so that the application exits as soon as possible.
		// `reset_session` aborts the watchdog's handle, and this task is the
		// watchdog: detach the handle first, or the abort would cancel the wait
		// below.
		{
			let mut guard = inner.lock().await;
			drop(guard.stop_watcher.take());
			guard.reset_session();
		}

		// Then wait for the session to shut down.
		let timeout = std::time::Duration::from_secs(SESSION_SHUTDOWN_TIMEOUT_SECS);
		if tokio::time::timeout(timeout, stop.wait_shutdown_complete())
			.await
			.is_err()
		{
			tracing::error!("Session shutdown timed out after {SESSION_SHUTDOWN_TIMEOUT_SECS}s.");
		}
		drop(delay);
	});
	guard.stop_watcher = Some(handle);
}

/// Wait for the session to shut down within the given timeout.
///
/// If the session does not shut down in time, triggers a global application
/// shutdown to prevent orphaned tasks and resource leaks.
async fn wait_for_session_shutdown(
	stop: &ShutdownManager<SessionShutdownReason>,
	shutdown: &ShutdownManager<ShutdownReason>,
	timeout_secs: u64,
) -> Result<(), ()> {
	match tokio::time::timeout(
		std::time::Duration::from_secs(timeout_secs),
		stop.wait_shutdown_complete(),
	)
	.await
	{
		Ok(_) => Ok(()),
		Err(_) => {
			tracing::error!("Session shutdown timed out after {timeout_secs}s — triggering application shutdown.");
			let _ = shutdown.trigger_shutdown(ShutdownReason::SessionManagerShutdown);
			Err(())
		},
	}
}
