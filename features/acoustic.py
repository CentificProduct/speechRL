"""
Emotion-GSRM: Acoustic Feature Extraction Pipeline
====================================================
Extracts emotion-relevant acoustic features from speech audio files.

Features are organized into four groups:
  1. Prosodic (retained from GSRM): vowel-level pitch, intensity, duration
  2. Voice Quality (new): jitter, shimmer, HNR, spectral tilt
  3. Temporal (new): speech rate, pause duration/frequency
  4. Formant (new): F1/F2 means, vowel space area

All features undergo speaker-level z-normalization and quantile-based
discretization into ordinal categories following the GSRM methodology.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ProsoddicFeatures:
    """Vowel-level prosodic features retained from GSRM."""
    pitch_level: np.ndarray          # Mean F0 per vowel segment (Hz)
    pitch_variation: np.ndarray      # F0 std per vowel segment
    pitch_slope: np.ndarray          # F0 slope per vowel segment (Hz/s)
    intensity_level: np.ndarray      # Mean intensity per vowel (dB)
    intensity_variation: np.ndarray  # Intensity std per vowel
    duration: np.ndarray             # Duration of each vowel segment (s)


@dataclass
class VoiceQualityFeatures:
    """Voice quality features — new for Emotion-GSRM."""
    jitter: float          # Cycle-to-cycle F0 perturbation (%)
    shimmer: float         # Cycle-to-cycle amplitude perturbation (%)
    hnr: float             # Harmonics-to-noise ratio (dB)
    spectral_tilt: float   # Spectral slope (dB/octave)


@dataclass
class TemporalFeatures:
    """Temporal patterning features — new for Emotion-GSRM."""
    speech_rate: float               # Syllables per second
    speech_rate_variation: float     # Std of local speech rate
    pause_duration_mean: float       # Mean pause duration (s)
    pause_frequency: float           # Pauses per second


@dataclass
class FormantFeatures:
    """Formant dynamics — new for Emotion-GSRM."""
    f1_mean: float        # Mean first formant (Hz)
    f2_mean: float        # Mean second formant (Hz)
    vowel_space_area: float  # Triangular VSA from corner vowels (Hz²)


@dataclass
class AcousticFeatureSet:
    """Complete acoustic feature set for a single utterance."""
    utterance_id: str
    prosodic: ProsoddicFeatures
    voice_quality: VoiceQualityFeatures
    temporal: TemporalFeatures
    formant: FormantFeatures
    raw_audio_path: str
    sample_rate: int = 16000
    speaker_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class AcousticFeatureExtractor:
    """
    Extracts emotion-relevant acoustic features from speech audio.

    Uses:
      - Parselmouth (Praat bindings) for pitch, formants, voice quality
      - librosa for temporal features and audio I/O
      - Forced alignment output for vowel segmentation

    Parameters
    ----------
    sample_rate : int
        Target sample rate for audio loading (default: 16000).
    frame_length_ms : float
        Analysis frame length in ms (default: 25.0).
    hop_length_ms : float
        Frame hop in ms (default: 10.0).
    min_pitch : float
        Minimum F0 for pitch tracking (default: 75 Hz).
    max_pitch : float
        Maximum F0 for pitch tracking (default: 500 Hz).
    silence_threshold_db : float
        Threshold below which frames are considered silence (default: -40 dB).
    min_pause_duration : float
        Minimum duration to count as a pause (default: 0.15 s).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        min_pitch: float = 75.0,
        max_pitch: float = 500.0,
        silence_threshold_db: float = -40.0,
        min_pause_duration: float = 0.15,
    ):
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.silence_threshold_db = silence_threshold_db
        self.min_pause_duration = min_pause_duration

        # Derived
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.hop_length = int(sample_rate * hop_length_ms / 1000)

    # ---- Audio loading ----

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Load and resample audio to target sample rate."""
        import librosa

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return y, sr

    # ---- Prosodic features (GSRM-retained) ----

    def extract_prosodic(
        self,
        y: np.ndarray,
        vowel_segments: Optional[list[tuple[float, float]]] = None,
    ) -> ProsoddicFeatures:
        """
        Extract vowel-level prosodic features.

        If vowel_segments is None, falls back to full-utterance extraction
        using a simple energy-based vowel approximation.

        Parameters
        ----------
        y : np.ndarray
            Audio signal.
        vowel_segments : list of (start, end) in seconds, optional
            Vowel boundaries from forced alignment.
        """
        import parselmouth

        snd = parselmouth.Sound(y, sampling_frequency=self.sample_rate)
        pitch_obj = snd.to_pitch(
            time_step=self.hop_length_ms / 1000,
            pitch_floor=self.min_pitch,
            pitch_ceiling=self.max_pitch,
        )
        intensity_obj = snd.to_intensity(
            minimum_pitch=self.min_pitch,
            time_step=self.hop_length_ms / 1000,
        )

        # Fall back to energy-based segments if no alignment provided
        if vowel_segments is None:
            vowel_segments = self._approximate_vowel_segments(y)

        pitch_levels, pitch_vars, pitch_slopes = [], [], []
        int_levels, int_vars, durations = [], [], []

        for start, end in vowel_segments:
            dur = end - start
            durations.append(dur)

            # Pitch within segment
            f0_values = []
            t = start
            while t <= end:
                f0 = pitch_obj.get_value_at_time(t)
                if f0 and not np.isnan(f0):
                    f0_values.append(f0)
                t += self.hop_length_ms / 1000

            if len(f0_values) > 0:
                f0_arr = np.array(f0_values)
                pitch_levels.append(np.mean(f0_arr))
                pitch_vars.append(np.std(f0_arr))
                # Slope via linear regression
                if len(f0_arr) > 1:
                    times = np.linspace(0, dur, len(f0_arr))
                    slope = np.polyfit(times, f0_arr, 1)[0]
                    pitch_slopes.append(slope)
                else:
                    pitch_slopes.append(0.0)
            else:
                pitch_levels.append(0.0)
                pitch_vars.append(0.0)
                pitch_slopes.append(0.0)

            # Intensity within segment
            int_values = []
            t = start
            while t <= end:
                val = intensity_obj.get_value(t)
                if val and not np.isnan(val):
                    int_values.append(val)
                t += self.hop_length_ms / 1000

            if len(int_values) > 0:
                int_arr = np.array(int_values)
                int_levels.append(np.mean(int_arr))
                int_vars.append(np.std(int_arr))
            else:
                int_levels.append(0.0)
                int_vars.append(0.0)

        return ProsoddicFeatures(
            pitch_level=np.array(pitch_levels),
            pitch_variation=np.array(pitch_vars),
            pitch_slope=np.array(pitch_slopes),
            intensity_level=np.array(int_levels),
            intensity_variation=np.array(int_vars),
            duration=np.array(durations),
        )

    # ---- Voice quality features (new) ----

    def extract_voice_quality(self, y: np.ndarray) -> VoiceQualityFeatures:
        """
        Extract voice quality features via Parselmouth.

        Returns jitter, shimmer, HNR, and spectral tilt.
        """
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(y, sampling_frequency=self.sample_rate)

        # Point process for jitter/shimmer
        pitch_obj = snd.to_pitch(
            pitch_floor=self.min_pitch,
            pitch_ceiling=self.max_pitch,
        )
        point_process = call(
            snd, "To PointProcess (periodic, cc)",
            self.min_pitch, self.max_pitch,
        )

        # Jitter (local, relative)
        jitter = call(
            point_process, "Get jitter (local)",
            0.0, 0.0, 0.0001, 0.02, 1.3,
        )

        # Shimmer (local, relative)
        shimmer = call(
            [snd, point_process], "Get shimmer (local)",
            0.0, 0.0, 0.0001, 0.02, 1.3, 1.6,
        )

        # Harmonics-to-noise ratio
        harmonicity = snd.to_harmonicity(
            time_step=self.hop_length_ms / 1000,
            minimum_pitch=self.min_pitch,
        )
        hnr_values = [
            harmonicity.get_value(t)
            for t in np.arange(
                harmonicity.xmin, harmonicity.xmax,
                self.hop_length_ms / 1000,
            )
        ]
        hnr_values = [v for v in hnr_values if v is not None and not np.isnan(v)]
        hnr = float(np.mean(hnr_values)) if hnr_values else 0.0

        # Spectral tilt: slope of the log power spectrum
        spectral_tilt = self._compute_spectral_tilt(y)

        return VoiceQualityFeatures(
            jitter=float(jitter) * 100,  # Convert to percentage
            shimmer=float(shimmer) * 100,
            hnr=hnr,
            spectral_tilt=spectral_tilt,
        )

    # ---- Temporal features (new) ----

    def extract_temporal(self, y: np.ndarray) -> TemporalFeatures:
        """
        Extract temporal patterning features using librosa.

        Computes speech rate, speech rate variation, and pause statistics.
        """
        import librosa

        duration = len(y) / self.sample_rate

        # Energy envelope for VAD
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Binary speech/silence mask
        is_speech = rms_db > self.silence_threshold_db
        frame_duration = self.hop_length / self.sample_rate

        # Detect pauses
        pauses = []
        in_pause = False
        pause_start = 0.0
        for i, active in enumerate(is_speech):
            t = i * frame_duration
            if not active and not in_pause:
                in_pause = True
                pause_start = t
            elif active and in_pause:
                in_pause = False
                pause_dur = t - pause_start
                if pause_dur >= self.min_pause_duration:
                    pauses.append(pause_dur)
        # Handle trailing pause
        if in_pause:
            pause_dur = len(is_speech) * frame_duration - pause_start
            if pause_dur >= self.min_pause_duration:
                pauses.append(pause_dur)

        # Speech rate estimation via onset envelope
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            units="time",
        )

        speech_time = np.sum(is_speech) * frame_duration
        speech_rate = len(onsets) / max(speech_time, 0.1)

        # Local speech rate variation (sliding window)
        window_sec = 1.0
        local_rates = []
        for start_t in np.arange(0, duration - window_sec, window_sec / 2):
            end_t = start_t + window_sec
            n_onsets = np.sum((onsets >= start_t) & (onsets < end_t))
            local_rates.append(n_onsets / window_sec)
        speech_rate_var = float(np.std(local_rates)) if local_rates else 0.0

        pause_duration_mean = float(np.mean(pauses)) if pauses else 0.0
        pause_frequency = len(pauses) / max(duration, 0.1)

        return TemporalFeatures(
            speech_rate=speech_rate,
            speech_rate_variation=speech_rate_var,
            pause_duration_mean=pause_duration_mean,
            pause_frequency=pause_frequency,
        )

    # ---- Formant features (new) ----

    def extract_formants(self, y: np.ndarray) -> FormantFeatures:
        """
        Extract formant dynamics via Parselmouth/Praat.

        Computes F1/F2 means and triangular vowel space area.
        """
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(y, sampling_frequency=self.sample_rate)
        formant_obj = snd.to_formant_burg(
            time_step=self.hop_length_ms / 1000,
            max_number_of_formants=5.0,
            maximum_formant=5500.0,
            window_length=self.frame_length_ms / 1000 * 2,
        )

        f1_values, f2_values = [], []
        n_frames = call(formant_obj, "Get number of frames")
        for i in range(1, n_frames + 1):
            t = call(formant_obj, "Get time from frame number", i)
            f1 = call(formant_obj, "Get value at time", 1, t, "hertz", "Linear")
            f2 = call(formant_obj, "Get value at time", 2, t, "hertz", "Linear")
            if f1 and not np.isnan(f1) and f1 > 0:
                f1_values.append(f1)
            if f2 and not np.isnan(f2) and f2 > 0:
                f2_values.append(f2)

        f1_mean = float(np.mean(f1_values)) if f1_values else 0.0
        f2_mean = float(np.mean(f2_values)) if f2_values else 0.0

        # Triangular vowel space area (approximate from F1/F2 distribution)
        # Uses the triangle formed by extreme vowel positions
        vsa = self._compute_vowel_space_area(f1_values, f2_values)

        return FormantFeatures(
            f1_mean=f1_mean,
            f2_mean=f2_mean,
            vowel_space_area=vsa,
        )

    # ---- Full extraction pipeline ----

    def extract(
        self,
        audio_path: str,
        utterance_id: str,
        speaker_id: Optional[str] = None,
        vowel_segments: Optional[list[tuple[float, float]]] = None,
    ) -> AcousticFeatureSet:
        """
        Run the full acoustic feature extraction pipeline on one utterance.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.
        utterance_id : str
            Unique ID for this utterance.
        speaker_id : str, optional
            Speaker ID for z-normalization grouping.
        vowel_segments : list of (start, end), optional
            Vowel segment boundaries from forced alignment.
        """
        y, sr = self.load_audio(audio_path)

        prosodic = self.extract_prosodic(y, vowel_segments)
        voice_quality = self.extract_voice_quality(y)
        temporal = self.extract_temporal(y)
        formant = self.extract_formants(y)

        return AcousticFeatureSet(
            utterance_id=utterance_id,
            prosodic=prosodic,
            voice_quality=voice_quality,
            temporal=temporal,
            formant=formant,
            raw_audio_path=audio_path,
            sample_rate=sr,
            speaker_id=speaker_id,
        )

    # ---- Helpers ----

    def _approximate_vowel_segments(
        self, y: np.ndarray
    ) -> list[tuple[float, float]]:
        """
        Approximate vowel segments using energy-based heuristic.

        When forced alignment is unavailable, uses high-energy regions
        as a proxy for voiced (vowel-like) segments.
        """
        import librosa

        rms = librosa.feature.rms(
            y=y, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        threshold = np.percentile(rms_db[rms_db > -80], 40)

        frame_dur = self.hop_length / self.sample_rate
        segments = []
        in_segment = False
        seg_start = 0.0

        for i, db in enumerate(rms_db):
            t = i * frame_dur
            if db > threshold and not in_segment:
                in_segment = True
                seg_start = t
            elif db <= threshold and in_segment:
                in_segment = False
                if t - seg_start > 0.03:  # Min 30ms
                    segments.append((seg_start, t))

        if in_segment:
            end_t = len(rms_db) * frame_dur
            if end_t - seg_start > 0.03:
                segments.append((seg_start, end_t))

        return segments if segments else [(0.0, len(y) / self.sample_rate)]

    def _compute_spectral_tilt(self, y: np.ndarray) -> float:
        """Compute spectral tilt as slope of log-power spectrum."""
        import librosa

        S = np.abs(librosa.stft(
            y, n_fft=self.frame_length, hop_length=self.hop_length
        ))
        power = np.mean(S ** 2, axis=1)
        power_db = librosa.amplitude_to_db(np.sqrt(power + 1e-10))

        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
        # Only fit in speech-relevant range (50-5000 Hz)
        mask = (freqs >= 50) & (freqs <= 5000)
        if np.sum(mask) < 2:
            return 0.0

        log_freqs = np.log2(freqs[mask] + 1e-10)
        slope = np.polyfit(log_freqs, power_db[mask], 1)[0]
        return float(slope)

    def _compute_vowel_space_area(
        self, f1_values: list[float], f2_values: list[float]
    ) -> float:
        """
        Estimate vowel space area from F1/F2 distributions.

        Uses the convex hull of extreme formant positions as proxy
        for the classic triangular VSA (/a/, /i/, /u/).
        """
        if len(f1_values) < 10 or len(f2_values) < 10:
            return 0.0

        f1 = np.array(f1_values)
        f2 = np.array(f2_values)

        # Approximate corner vowels from distribution extremes
        # /a/ ~ high F1, mid F2
        # /i/ ~ low F1, high F2
        # /u/ ~ low F1, low F2
        p10_f1, p90_f1 = np.percentile(f1, [10, 90])
        p10_f2, p90_f2 = np.percentile(f2, [10, 90])

        # Triangle vertices (F1, F2)
        a = (p90_f1, np.median(f2))   # /a/
        i_vowel = (p10_f1, p90_f2)    # /i/
        u = (p10_f1, p10_f2)          # /u/

        # Shoelace formula for triangle area
        area = 0.5 * abs(
            (i_vowel[0] - a[0]) * (u[1] - a[1])
            - (u[0] - a[0]) * (i_vowel[1] - a[1])
        )
        return float(area)


# ---------------------------------------------------------------------------
# Normalization & Discretization (GSRM methodology)
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """
    Speaker-level z-normalization and quantile-based discretization.

    Follows the GSRM approach: features are first z-normalized per speaker,
    then discretized into ordinal categories (e.g., 'very_low', 'low',
    'medium', 'high', 'very_high') using quantile boundaries.
    """

    ORDINAL_LABELS = ["very_low", "low", "medium", "high", "very_high"]

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self._speaker_stats: dict[str, dict[str, tuple[float, float]]] = {}
        self._quantile_boundaries: dict[str, np.ndarray] = {}

    def fit_speaker(
        self, speaker_id: str, feature_sets: list[AcousticFeatureSet]
    ) -> None:
        """
        Compute per-speaker mean and std for z-normalization.

        Parameters
        ----------
        speaker_id : str
            The speaker identifier.
        feature_sets : list of AcousticFeatureSet
            All utterances from this speaker.
        """
        stats = {}
        scalar_features = self._collect_scalar_features(feature_sets)

        for feat_name, values in scalar_features.items():
            arr = np.array(values)
            stats[feat_name] = (float(np.mean(arr)), float(np.std(arr) + 1e-8))

        self._speaker_stats[speaker_id] = stats

    def fit_quantiles(self, all_feature_sets: list[AcousticFeatureSet]) -> None:
        """
        Compute global quantile boundaries for discretization.

        Should be called after z-normalizing all speakers' data.
        """
        scalar_features = self._collect_scalar_features(all_feature_sets)

        for feat_name, values in scalar_features.items():
            arr = np.array(values)
            boundaries = np.percentile(
                arr, np.linspace(0, 100, self.n_bins + 1)[1:-1]
            )
            self._quantile_boundaries[feat_name] = boundaries

    def normalize(self, feature_set: AcousticFeatureSet) -> dict[str, float]:
        """
        Z-normalize scalar features using speaker stats.

        Returns a dict of feature_name -> z-normalized value.
        """
        speaker_id = feature_set.speaker_id or "unknown"
        stats = self._speaker_stats.get(speaker_id, {})

        scalars = self._extract_scalars(feature_set)
        normalized = {}
        for name, value in scalars.items():
            if name in stats:
                mean, std = stats[name]
                normalized[name] = (value - mean) / std
            else:
                normalized[name] = value

        return normalized

    def discretize(self, normalized_features: dict[str, float]) -> dict[str, str]:
        """
        Discretize z-normalized features into ordinal categories.

        Returns a dict of feature_name -> ordinal label string.
        """
        discretized = {}
        for name, value in normalized_features.items():
            if name in self._quantile_boundaries:
                boundaries = self._quantile_boundaries[name]
                bin_idx = int(np.searchsorted(boundaries, value))
                bin_idx = min(bin_idx, self.n_bins - 1)
                discretized[name] = self.ORDINAL_LABELS[bin_idx]
            else:
                discretized[name] = "medium"  # Default
        return discretized

    # ---- Internal helpers ----

    def _extract_scalars(self, fs: AcousticFeatureSet) -> dict[str, float]:
        """Extract all scalar features from an AcousticFeatureSet."""
        scalars = {}

        # Prosodic aggregates
        p = fs.prosodic
        scalars["pitch_level_mean"] = float(np.mean(p.pitch_level)) if len(p.pitch_level) else 0.0
        scalars["pitch_variation_mean"] = float(np.mean(p.pitch_variation)) if len(p.pitch_variation) else 0.0
        scalars["pitch_slope_mean"] = float(np.mean(p.pitch_slope)) if len(p.pitch_slope) else 0.0
        scalars["intensity_level_mean"] = float(np.mean(p.intensity_level)) if len(p.intensity_level) else 0.0
        scalars["intensity_variation_mean"] = float(np.mean(p.intensity_variation)) if len(p.intensity_variation) else 0.0
        scalars["duration_mean"] = float(np.mean(p.duration)) if len(p.duration) else 0.0

        # Voice quality
        vq = fs.voice_quality
        scalars["jitter"] = vq.jitter
        scalars["shimmer"] = vq.shimmer
        scalars["hnr"] = vq.hnr
        scalars["spectral_tilt"] = vq.spectral_tilt

        # Temporal
        t = fs.temporal
        scalars["speech_rate"] = t.speech_rate
        scalars["speech_rate_variation"] = t.speech_rate_variation
        scalars["pause_duration_mean"] = t.pause_duration_mean
        scalars["pause_frequency"] = t.pause_frequency

        # Formant
        f = fs.formant
        scalars["f1_mean"] = f.f1_mean
        scalars["f2_mean"] = f.f2_mean
        scalars["vowel_space_area"] = f.vowel_space_area

        return scalars

    def _collect_scalar_features(
        self, feature_sets: list[AcousticFeatureSet]
    ) -> dict[str, list[float]]:
        """Collect all scalar features across multiple utterances."""
        collected: dict[str, list[float]] = {}
        for fs in feature_sets:
            scalars = self._extract_scalars(fs)
            for name, value in scalars.items():
                collected.setdefault(name, []).append(value)
        return collected


# ---------------------------------------------------------------------------
# Feature Formatting (for CoT prompt construction)
# ---------------------------------------------------------------------------

def format_features_for_prompt(
    feature_set: AcousticFeatureSet,
    discretized: dict[str, str],
) -> str:
    """
    Format acoustic features into a structured text block for the
    CoT synthesis prompt (input to GPT-4o teacher).

    Returns a human-readable feature summary suitable for prompt injection.
    """
    lines = [
        f"=== Acoustic Features for Utterance: {feature_set.utterance_id} ===",
        "",
        "## Prosodic Features (vowel-level)",
        f"  Pitch level: {discretized.get('pitch_level_mean', 'N/A')} "
        f"(mean={np.mean(feature_set.prosodic.pitch_level):.1f} Hz)",
        f"  Pitch variation: {discretized.get('pitch_variation_mean', 'N/A')} "
        f"(std={np.mean(feature_set.prosodic.pitch_variation):.1f} Hz)",
        f"  Pitch slope: {discretized.get('pitch_slope_mean', 'N/A')} "
        f"(mean={np.mean(feature_set.prosodic.pitch_slope):.1f} Hz/s)",
        f"  Intensity level: {discretized.get('intensity_level_mean', 'N/A')} "
        f"(mean={np.mean(feature_set.prosodic.intensity_level):.1f} dB)",
        f"  Intensity variation: {discretized.get('intensity_variation_mean', 'N/A')} "
        f"(std={np.mean(feature_set.prosodic.intensity_variation):.1f} dB)",
        f"  Vowel durations: {discretized.get('duration_mean', 'N/A')} "
        f"(mean={np.mean(feature_set.prosodic.duration)*1000:.1f} ms)",
        "",
        "## Voice Quality Features",
        f"  Jitter: {discretized.get('jitter', 'N/A')} ({feature_set.voice_quality.jitter:.2f}%)",
        f"  Shimmer: {discretized.get('shimmer', 'N/A')} ({feature_set.voice_quality.shimmer:.2f}%)",
        f"  HNR: {discretized.get('hnr', 'N/A')} ({feature_set.voice_quality.hnr:.1f} dB)",
        f"  Spectral tilt: {discretized.get('spectral_tilt', 'N/A')} "
        f"({feature_set.voice_quality.spectral_tilt:.2f} dB/oct)",
        "",
        "## Temporal Features",
        f"  Speech rate: {discretized.get('speech_rate', 'N/A')} "
        f"({feature_set.temporal.speech_rate:.1f} syl/s)",
        f"  Speech rate variation: {discretized.get('speech_rate_variation', 'N/A')} "
        f"({feature_set.temporal.speech_rate_variation:.2f})",
        f"  Mean pause duration: {discretized.get('pause_duration_mean', 'N/A')} "
        f"({feature_set.temporal.pause_duration_mean*1000:.0f} ms)",
        f"  Pause frequency: {discretized.get('pause_frequency', 'N/A')} "
        f"({feature_set.temporal.pause_frequency:.2f}/s)",
        "",
        "## Formant Features",
        f"  F1 mean: {discretized.get('f1_mean', 'N/A')} ({feature_set.formant.f1_mean:.0f} Hz)",
        f"  F2 mean: {discretized.get('f2_mean', 'N/A')} ({feature_set.formant.f2_mean:.0f} Hz)",
        f"  Vowel space area: {discretized.get('vowel_space_area', 'N/A')} "
        f"({feature_set.formant.vowel_space_area:.0f} Hz²)",
    ]
    return "\n".join(lines)