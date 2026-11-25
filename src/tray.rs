//! A tray icon (a StatusNotifierItem) that shows whether a client streams. Clicking it opens the
//! control panel.

use std::sync::LazyLock;

use async_shutdown::ShutdownManager;
use image::imageops::{self, FilterType};
use image::{ImageFormat, Rgba, RgbaImage};
use ksni::TrayMethods;
use tokio::sync::broadcast::error::RecvError;

use crate::ShutdownReason;
use crate::session::manager::SessionManager;
use crate::session::stream::video::StreamEvent;

/// The moon from the banner, at the banner's own pixel scale.
const MOON: &[u8] = include_bytes!("../assets/icon.png");

/// Icon sizes for the tray to choose from.
const SIZES: [u32; 6] = [16, 22, 24, 32, 48, 64];

/// The dot that marks a stream.
const LIVE: Rgba<u8> = Rgba([232, 64, 72, 255]);

/// The banner's night sky, around the dot and the moon.
const OUTLINE: Rgba<u8> = Rgba([14, 10, 40, 255]);

/// The icons without and with the stream's dot.
static ICONS: LazyLock<[Vec<ksni::Icon>; 2]> = LazyLock::new(|| {
	let moon = image::load_from_memory_with_format(MOON, ImageFormat::Png)
		.expect("the tray icon is a PNG")
		.into_rgba8();
	[false, true].map(|streaming| {
		SIZES
			.iter()
			.map(|&size| {
				let mut icon = imageops::resize(&moon, size, size, FilterType::Lanczos3);
				if streaming {
					draw_dot(&mut icon);
				}
				argb(icon)
			})
			.collect()
	})
});

/// Draw the stream's dot in the bottom right corner.
fn draw_dot(icon: &mut RgbaImage) {
	let size = icon.width() as f32;
	let radius = size * 0.22;
	let ring = (size / 22.0).max(1.0);
	let center = size - radius - size * 0.02;
	for (x, y, pixel) in icon.enumerate_pixels_mut() {
		let distance = (x as f32 + 0.5 - center).hypot(y as f32 + 0.5 - center);
		if distance <= radius {
			*pixel = LIVE;
		} else if distance <= radius + ring {
			*pixel = OUTLINE;
		}
	}
}

fn argb(icon: RgbaImage) -> ksni::Icon {
	let (width, height) = icon.dimensions();
	let mut data = icon.into_raw();
	for pixel in data.as_chunks_mut::<4>().0 {
		pixel.rotate_right(1);
	}
	ksni::Icon {
		width: width as i32,
		height: height as i32,
		data,
	}
}

struct Tray {
	/// Where a browser on this computer opens the control panel.
	url: String,
	/// The application that streams, if a client streams.
	streaming: Option<String>,
}

impl ksni::Tray for Tray {
	fn id(&self) -> String {
		"moonshine".to_string()
	}

	fn title(&self) -> String {
		"Moonshine".to_string()
	}

	fn icon_pixmap(&self) -> Vec<ksni::Icon> {
		ICONS[usize::from(self.streaming.is_some())].clone()
	}

	fn tool_tip(&self) -> ksni::ToolTip {
		ksni::ToolTip {
			title: "Moonshine".to_string(),
			description: match &self.streaming {
				Some(application) => format!("Streaming {application}"),
				None => "Not streaming".to_string(),
			},
			..Default::default()
		}
	}

	fn activate(&mut self, _x: i32, _y: i32) {
		let url = self.url.clone();
		let _ = std::thread::Builder::new()
			.name("open-control-panel".to_string())
			.spawn(move || {
				if let Err(e) = open::that(&url) {
					tracing::warn!("Failed to open the control panel at {url}: {e}");
				}
			});
	}

	fn watcher_offline(&self, reason: ksni::OfflineReason) -> bool {
		// Keep the icon, and show it again when a tray comes back.
		tracing::debug!("No tray shows the tray icon: {reason:?}");
		true
	}
}

/// The application that streams, if a client streams.
async fn streaming_application(session_manager: &SessionManager) -> Option<String> {
	let status = session_manager.status().await?;
	status.stream.is_some().then_some(status.context.application.title)
}

/// Show the tray icon until Moonshine shuts down. `url` opens the control panel.
pub fn spawn(url: String, session_manager: SessionManager, shutdown: ShutdownManager<ShutdownReason>) {
	tokio::spawn(async move {
		// Subscribe before reading the state, so that no change in between is missed.
		let (_, mut events) = session_manager.join_stats();
		let tray = Tray {
			url,
			streaming: streaming_application(&session_manager).await,
		};

		// Moonshine can start before the desktop's tray, so wait for one to appear.
		let handle = match tray.assume_sni_available(true).spawn().await {
			Ok(handle) => handle,
			Err(e) => {
				tracing::warn!("Failed to show the tray icon: {e}");
				return;
			},
		};

		loop {
			tokio::select! {
				_ = shutdown.wait_shutdown_triggered() => break,
				event = events.recv() => match event {
					Ok(StreamEvent::Frame(_)) => {},
					Ok(StreamEvent::Started(_) | StreamEvent::Stopped) | Err(RecvError::Lagged(_)) => {
						let streaming = streaming_application(&session_manager).await;
						handle.update(|tray| tray.streaming = streaming).await;
					},
					Err(RecvError::Closed) => break,
				},
			}
		}

		handle.shutdown().await;
	});
}
