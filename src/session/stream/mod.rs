use aws_lc_rs::rand::{SecureRandom, SystemRandom};
use serde::Deserialize;
use serde::Serialize;

use crate::session::stream::audio::AudioStreamConfig;
use crate::session::stream::control::ControlStreamConfig;
use crate::session::stream::video::VideoStreamConfig;

pub mod audio;
pub mod control;
mod start;
pub mod video;

pub(crate) use start::{StartSignal, StartWaiter};

/// The 16 printable bytes advertised in RTSP SETUP and echoed by audio and video pings.
/// Each resumed media connection advertises a new token.
#[derive(Clone, Copy)]
pub(crate) struct AvPingToken([u8; 16]);

impl AvPingToken {
	pub(crate) fn new() -> Result<Self, ()> {
		const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
		let mut bytes = [0; 16];
		SystemRandom::new()
			.fill(&mut bytes)
			.map_err(|_| tracing::error!("Failed to generate the stream ping token"))?;
		for byte in &mut bytes {
			*byte = ALPHABET[usize::from(*byte & 63)];
		}
		Ok(Self(bytes))
	}

	pub(crate) fn as_str(&self) -> &str {
		std::str::from_utf8(&self.0).expect("ping token is ASCII")
	}

	fn matches(&self, datagram: &[u8]) -> bool {
		// Legacy clients send PING; Sunshine-compatible clients append a big-endian u32 sequence.
		datagram == b"PING" || (datagram.len() == 20 && datagram.starts_with(&self.0))
	}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamConfig {
	/// Port to bind the RTSP server to.
	pub port: u16,

	/// Configuration for the video stream.
	pub video: VideoStreamConfig,

	/// Configuration for the audio stream.
	pub audio: AudioStreamConfig,

	/// Configuration for the control stream.
	pub control: ControlStreamConfig,

	/// Time in seconds since last ping after which the stream closes.
	pub timeout: u64,
}

impl Default for StreamConfig {
	fn default() -> Self {
		Self {
			port: 48010,
			video: Default::default(),
			audio: Default::default(),
			control: Default::default(),
			timeout: 60,
		}
	}
}

#[derive(Debug)]
#[repr(C)]
struct RtpHeader {
	header: u8,
	packet_type: u8,
	sequence_number: u16,
	timestamp: u32,
	ssrc: u32,
}

impl RtpHeader {}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn av_ping_requires_the_full_token_and_sequence_or_legacy_ping() {
		let token = AvPingToken(*b"AbCdEf0123456789");
		let mut ping = *b"AbCdEf0123456789\0\0\0\x01";
		assert!(token.matches(&ping));
		assert!(token.matches(b"PING"));
		assert!(!token.matches(&ping[16..]), "sequence without a token");
		assert!(!token.matches(&ping[..19]), "truncated sequence");
		ping[15] = b'0';
		assert!(!token.matches(&ping), "wrong session's token");
	}
}
