use std::collections::HashMap;

use async_shutdown::ShutdownManager;
use manager::SessionShutdownReason;
use tokio::sync::watch;

use crate::gpu::GpuContext;
use crate::session::compositor::CompositorConfig;
use crate::session::stream::AvPingToken;
use crate::session::stream::StartSignal;
use crate::session::stream::audio::AudioChannels;
use crate::session::stream::audio::AudioStream;
use crate::session::stream::audio::AudioStreamContext;
use crate::session::stream::control::ControlStream;
use crate::session::stream::control::ControlStreamContext;
use crate::session::stream::video::StatsSender;
use crate::session::stream::video::VideoDynamicRange;
use crate::session::stream::video::VideoStream;
use crate::session::stream::video::VideoStreamContext;
use crate::session::stream::video::VideoStreamHandle;

use self::application::Application;
use self::application::ApplicationConfig;
use self::application::ApplicationContext;
use self::compositor::Compositor;
use self::compositor::LaunchedCompositor;
use self::compositor::frame::HdrModeState;
use self::inhibit::SleepInhibitor;
use self::stream::audio::AudioStreamConfig;
use self::stream::control::ControlStreamConfig;
use self::stream::video::VideoStreamConfig;

pub mod application;
pub mod compositor;
pub mod inhibit;
pub mod manager;
pub mod stream;

/// Timeout in seconds for the HTTP launch endpoint to wait for the session to launch.
pub(crate) const APP_LAUNCH_HTTP_TIMEOUT_SECS: u64 = 60;

/// Raw session encryption key data.
#[derive(Clone, Debug)]
pub struct SessionKeyData {
	/// AES GCM key used for encoding video / audio / control messages.
	pub remote_input_key: Vec<u8>,

	/// AES GCM initialization vector for video / audio / control messages.
	pub remote_input_key_id: i64,
}

pub(crate) type SessionKeysReceiver = watch::Receiver<SessionKeyData>;
pub(crate) type SessionKeysSender = watch::Sender<SessionKeyData>;

/// Session keys — either raw keys or a watch receiver.
#[derive(Clone, Debug)]
pub enum SessionKeys {
	Keys(SessionKeyData),
	Rx(SessionKeysReceiver),
}

impl SessionKeys {
	pub(crate) fn new(remote_input_key: Vec<u8>, remote_input_key_id: i64) -> Self {
		Self::Keys(SessionKeyData {
			remote_input_key,
			remote_input_key_id,
		})
	}

	pub(crate) fn clone_rx(&self) -> Option<SessionKeysReceiver> {
		match self {
			Self::Rx(rx) => Some(rx.clone()),
			_ => None,
		}
	}
}

/// Context for a session.
///
/// This is created at launch time and contains all the information about the session
/// that is needed to start the compositor, application, and streams.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct SessionContext {
	/// Application to launch.
	pub application: ApplicationConfig,

	/// ID of the application as reported to the client.
	pub application_id: i32,

	/// Resolution of the video stream (width, height).
	pub resolution: (u32, u32),

	/// Refresh rate of the video stream (in Hz).
	pub refresh_rate: u32,

	/// Encryption keys for encoding traffic.
	pub keys: SessionKeys,

	/// Audio channel count (2, 6, or 8).
	pub audio_channels: AudioChannels,

	/// Audio channel mask.
	pub audio_channel_mask: u32,

	/// If true, the compositor will be launched with HDR support.
	pub hdr: bool,
}

/// The state of the session. This enum enforces the session lifecycle:
///
/// 1. `Initialized` — Session created; compositor and app not yet started.
/// 2. `Launched` — Compositor and app are running; waiting for RTSP negotiation.
/// 3. `Active` — Streams are active.
enum SessionState {
	/// Session initialized; compositor and app not yet started.
	Initialized(InitializedSession),
	/// Compositor and app launched; waiting for RTSP PLAY.
	Launched(LaunchedSession),
	/// Streams active.
	Active(ActiveSession),
}

impl SessionState {
	fn context(&self) -> &SessionContext {
		match self {
			Self::Initialized(session) => session.context(),
			Self::Launched(launched) => launched.context(),
			Self::Active(active) => active.context(),
		}
	}

	fn av_ping_token(&self) -> AvPingToken {
		match self {
			Self::Initialized(session) => session.av_ping_token,
			Self::Launched(session) => session.resources.av_ping_token,
			Self::Active(session) => session.resources.av_ping_token,
		}
	}
}

/// Initialized session state — components created, compositor and app not yet started.
pub(crate) struct InitializedSession {
	context: SessionContext,
	av_ping_token: AvPingToken,
	/// Names the application's systemd unit and the PulseAudio socket directory
	/// (see `SessionManager::new`).
	session_name: String,
	compositor: Compositor,
	audio_stream: AudioStream,
	video_stream: VideoStream,
	control_stream: ControlStream,
	hdr_metadata_rx: watch::Receiver<HdrModeState>,
	stop: ShutdownManager<SessionShutdownReason>,
}

impl InitializedSession {
	#[allow(clippy::too_many_arguments)]
	pub(crate) async fn new(
		compositor_config: CompositorConfig,
		video_config: VideoStreamConfig,
		audio_config: AudioStreamConfig,
		control_config: ControlStreamConfig,
		address: String,
		context: SessionContext,
		session_name: String,
		gpu: GpuContext,
		stop: ShutdownManager<SessionShutdownReason>,
		stats_tx: StatsSender,
	) -> Result<Self, ()> {
		// Create HDR metadata watch channel.
		let (hdr_metadata_tx, hdr_metadata_rx) = watch::channel(HdrModeState::new(context.hdr));

		// Create compositor, audio stream, video stream, and control stream.
		// Read the host's keyboard layout for each stream, so a change on the
		// host applies without a restart.
		let compositor_config = CompositorConfig {
			keyboard: compositor_config.keyboard.resolve().await,
			..compositor_config
		};
		let (compositor, handles) = Compositor::new(compositor_config, (&context).into(), gpu.clone(), stop.clone());
		let audio = AudioStream::new(audio_config, address.clone(), &session_name, stop.clone()).await?;
		let video_stream = VideoStream::new(
			video_config.clone(),
			address.clone(),
			handles.frames,
			gpu,
			hdr_metadata_tx,
			stop.clone(),
			stats_tx,
		)
		.await?
		.with_capture_fps(context.refresh_rate);
		let control_stream = ControlStream::new(control_config, address, handles.input, stop.clone())?;

		Ok(Self {
			context,
			av_ping_token: AvPingToken::new()?,
			session_name,
			compositor,
			audio_stream: audio,
			video_stream,
			control_stream,
			hdr_metadata_rx,
			stop,
		})
	}

	pub(crate) fn context(&self) -> &SessionContext {
		&self.context
	}

	/// Launch the session — starts the compositor and application, but does not start streams.
	pub(crate) async fn launch(self) -> Result<LaunchedSession, ()> {
		let Self {
			context,
			av_ping_token,
			session_name,
			compositor,
			audio_stream: audio,
			video_stream,
			control_stream,
			hdr_metadata_rx,
			stop,
		} = self;

		let launched_compositor = compositor.launch()?;
		let ready = launched_compositor.ready();
		let pulse_socket_path = audio.pulse_socket_path.clone();

		let application = Application::spawn(
			context.application.clone(),
			ApplicationContext {
				unit_name: format!("{session_name}-session.service"),
				pulse_socket_path,
				xdisplay: ready.xdisplay,
				wayland_display: ready.wayland_display.clone(),
				compositor_env: ready.env.clone(),
				// Populate extra_env with width, height and refreshrate values of the client for e.g. scripting
				extra_env: HashMap::from([
					("MOONSHINE_CLIENT_WIDTH".to_string(), context.resolution.0.to_string()),
					("MOONSHINE_CLIENT_HEIGHT".to_string(), context.resolution.1.to_string()),
					(
						"MOONSHINE_CLIENT_FRAMERATE".to_string(),
						context.refresh_rate.to_string(),
					),
				]),
			},
			stop,
		)
		.await?;

		Ok(LaunchedSession {
			resources: SessionResources {
				context,
				av_ping_token,
				_application: application,
				video_stream,
				launched_compositor,
				audio,
			},
			control_stream,
			hdr_metadata_rx,
		})
	}
}

/// Launched session state — compositor and app running, waiting for RTSP negotiation.
pub(crate) struct LaunchedSession {
	resources: SessionResources,
	control_stream: ControlStream,
	hdr_metadata_rx: watch::Receiver<HdrModeState>,
}

/// The game, display and sockets outlive any one negotiated set of streams.
struct SessionResources {
	context: SessionContext,
	av_ping_token: AvPingToken,
	_application: Application,
	video_stream: VideoStream,
	launched_compositor: LaunchedCompositor,
	audio: AudioStream,
}

impl LaunchedSession {
	pub(crate) fn context(&self) -> &SessionContext {
		&self.resources.context
	}

	pub(crate) async fn start(
		self,
		video_config: VideoStreamConfig,
		stream_timeout: u64,
		video_ctx: VideoStreamContext,
		audio_ctx: AudioStreamContext,
		stop: ShutdownManager<SessionShutdownReason>,
		inhibit_sleep: bool,
	) -> Result<ActiveSession, ()> {
		let Self {
			mut resources,
			control_stream,
			hdr_metadata_rx,
		} = self;

		let (media_stop, controls) = resources.start_media(video_config, video_ctx, audio_ctx, &stop)?;
		let control_ctx = ControlStreamContext::new(&resources.context);
		let (controls, control_rx) = watch::channel(controls);
		control_stream.start(stream_timeout, control_ctx, control_rx, hdr_metadata_rx);

		let sleep_inhibitor = if inhibit_sleep {
			SleepInhibitor::acquire().await
		} else {
			None
		};

		Ok(ActiveSession {
			resources,
			media_stop,
			controls,
			_sleep_inhibitor: sleep_inhibitor,
		})
	}
}

impl SessionResources {
	fn start_media(
		&mut self,
		video_config: VideoStreamConfig,
		video_ctx: VideoStreamContext,
		audio_ctx: AudioStreamContext,
		session_stop: &ShutdownManager<SessionShutdownReason>,
	) -> Result<(ShutdownManager<SessionShutdownReason>, MediaControls), ()> {
		let hdr_requested = video_ctx.dynamic_range == VideoDynamicRange::Hdr;
		let audio_layout = (audio_ctx.audio_config.channels, audio_ctx.audio_config.channel_mask);
		let delay = session_stop.delay_shutdown_token().map_err(|e| {
			tracing::warn!("Cannot start media while the session is stopping: {e:?}");
		})?;
		let media_stop = ShutdownManager::new();
		tokio::spawn({
			let session_stop = session_stop.clone();
			let media_stop = media_stop.clone();
			async move {
				tokio::select! {
					reason = session_stop.wait_shutdown_triggered() => {
						let _ = media_stop.trigger_shutdown(reason);
					},
					reason = media_stop.wait_shutdown_triggered() => {
						if reason != SessionShutdownReason::MediaReplaced {
							let _ = session_stop.trigger_shutdown(reason);
						}
					},
				}
				media_stop.wait_shutdown_complete().await;
				drop(delay);
			}
		});

		let result = (|| {
			let keys = self.context.keys.clone_rx().expect("session keys not initialized");
			let hdr = self.launched_compositor.hdr() && hdr_requested;
			let video = self.video_stream.start(
				video_config,
				video_ctx,
				self.av_ping_token,
				keys.clone(),
				media_stop.clone(),
			)?;
			let audio = self
				.audio
				.start(audio_ctx, self.av_ping_token, keys, media_stop.clone())?;
			Ok(MediaControls { video, audio, hdr })
		})();
		match result {
			Ok(controls) => {
				self.context.hdr = hdr_requested;
				(self.context.audio_channels, self.context.audio_channel_mask) = audio_layout;
				Ok((media_stop, controls))
			},
			Err(()) => {
				let _ = media_stop.trigger_shutdown(SessionShutdownReason::MediaStartFailed);
				Err(())
			},
		}
	}
}

pub(crate) struct MediaControls {
	pub video: VideoStreamHandle,
	audio: StartSignal,
	pub hdr: bool,
}

impl MediaControls {
	pub fn trigger(&self) {
		self.video.trigger();
		self.audio.trigger();
	}
}

/// Active session state — streams are active.
pub(crate) struct ActiveSession {
	resources: SessionResources,
	media_stop: ShutdownManager<SessionShutdownReason>,
	controls: watch::Sender<MediaControls>,
	/// Held while the session is active to keep the host awake; dropped on teardown.
	_sleep_inhibitor: Option<SleepInhibitor>,
}

impl ActiveSession {
	pub(crate) fn context(&self) -> &SessionContext {
		&self.resources.context
	}

	pub(crate) fn trigger_streams_start(&self) {
		let controls = self.controls.borrow();
		controls.video.start_signal().trigger();
		controls.audio.trigger();
	}

	#[allow(clippy::too_many_arguments)]
	pub(crate) async fn resume(
		&mut self,
		video_config: VideoStreamConfig,
		video_ctx: VideoStreamContext,
		audio_ctx: AudioStreamContext,
		keys: SessionKeyData,
		keys_tx: &SessionKeysSender,
		av_ping_token: AvPingToken,
		session_stop: &ShutdownManager<SessionShutdownReason>,
		timeout: std::time::Duration,
	) -> Result<(), ()> {
		// Drain the old workers before changing keys, and before two generations
		// could read from the same socket or compositor frame receiver.
		let _ = self.media_stop.trigger_shutdown(SessionShutdownReason::MediaReplaced);
		tokio::time::timeout(timeout, self.media_stop.wait_shutdown_complete())
			.await
			.map_err(|_| tracing::error!("Timed out stopping media for resume"))?;
		if session_stop.is_shutdown_triggered() {
			return Err(());
		}

		keys_tx.send_replace(keys);
		self.resources.av_ping_token = av_ping_token;
		let (media_stop, controls) = self
			.resources
			.start_media(video_config, video_ctx, audio_ctx, session_stop)?;
		self.media_stop = media_stop;
		self.controls.send_replace(controls);
		Ok(())
	}
}
