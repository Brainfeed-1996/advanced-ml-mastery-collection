import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import time
import logging
from typing import Tuple, Generator, Optional, List
from collections import deque
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DigitalFilter:
    """
    Design and application of digital filters (IIR/FIR).
    """
    def __init__(self, filter_type: str = 'butter', order: int = 4, cutoff: float = 1000.0, fs: float = 44100.0, btype: str = 'lowpass'):
        self.fs = fs
        self.nyq = 0.5 * fs
        self.cutoff = cutoff
        self.btype = btype
        self.order = order
        
        if filter_type == 'butter':
            self.b, self.a = signal.butter(order, cutoff / self.nyq, btype=btype)
        elif filter_type == 'cheby1':
            self.b, self.a = signal.cheby1(order, 1, cutoff / self.nyq, btype=btype)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
        # Initial state for live filtering
        self.zi = signal.lfilter_zi(self.b, self.a)
        
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to a chunk of data (offline mode)."""
        return signal.filtfilt(self.b, self.a, data)

    def process_chunk(self, chunk: np.ndarray, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to a live chunk, maintaining state.
        Returns filtered chunk and new state.
        """
        if state is None:
            # Re-initialize zi with proper scaling if needed, but standard zi is usually fine
            # For strict correctness we scale zi by the first sample of chunk but lfilter handles state update
            state = self.zi * chunk[0] 
            
        y, z_out = signal.lfilter(self.b, self.a, chunk, zi=state)
        return y, z_out

class SpectralAnalyzer:
    """
    Real-time spectral analysis using FFT.
    """
    def __init__(self, n_fft: int = 1024, window: str = 'hann'):
        self.n_fft = n_fft
        self.window = signal.get_window(window, n_fft)
        
    def compute_spectrum(self, buffer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one-sided magnitude spectrum.
        Args:
            buffer: Input signal chunk (must match n_fft size)
        Returns:
            frequencies, magnitude_db
        """
        if len(buffer) != self.n_fft:
            # Pad or truncate
            if len(buffer) < self.n_fft:
                buffer = np.pad(buffer, (0, self.n_fft - len(buffer)))
            else:
                buffer = buffer[-self.n_fft:]
                
        # Apply window
        windowed = buffer * self.window
        
        # FFT
        fft_res = fftpack.fft(windowed)
        
        # Magnitude
        mag = np.abs(fft_res)[:self.n_fft // 2]
        
        # Normalize
        mag = mag / self.n_fft
        
        # Avoid log(0)
        mag_db = 20 * np.log10(mag + 1e-12)
        
        return mag_db

class RealTimeSignalProcessor:
    """
    Simulates a real-time DSP pipeline.
    """
    def __init__(self, fs: float = 44100.0, block_size: int = 1024):
        self.fs = fs
        self.block_size = block_size
        self.filter = DigitalFilter(cutoff=2000.0, fs=fs, btype='lowpass')
        self.analyzer = SpectralAnalyzer(n_fft=block_size)
        self.buffer = deque(maxlen=block_size)
        self.filter_state = self.filter.zi * 0 # Initialize zero state
        
        self.processing_times = []

    def stream_generator(self, duration_sec: float) -> Generator[np.ndarray, None, None]:
        """
        Generates simulated noisy signal blocks.
        Signal: 440Hz Sine + 10kHz Noise
        """
        total_samples = int(duration_sec * self.fs)
        num_blocks = total_samples // self.block_size
        
        t_global = 0
        
        for _ in range(num_blocks):
            t = np.arange(self.block_size) / self.fs + t_global
            
            # 440 Hz Sine
            sig = np.sin(2 * np.pi * 440 * t)
            
            # 10 kHz High frequency noise (to be filtered)
            noise = 0.5 * np.sin(2 * np.pi * 10000 * t)
            
            # Random white noise
            white = 0.1 * np.random.randn(self.block_size)
            
            yield sig + noise + white
            
            t_global += self.block_size / self.fs
            # Simulate real-time arrival
            # time.sleep(self.block_size / self.fs) # Commented out for speed in this script

    def run_pipeline(self, duration_sec: float = 5.0):
        """
        Execute the processing loop.
        """
        logger.info(f"Starting real-time processing simulation. Fs={self.fs}Hz, Block={self.block_size}")
        
        stream = self.stream_generator(duration_sec)
        
        block_count = 0
        total_time = 0
        
        try:
            for raw_block in stream:
                start_time = time.perf_counter()
                
                # 1. Filtering
                filtered_block, self.filter_state = self.filter.process_chunk(raw_block, self.filter_state)
                
                # 2. Spectral Analysis (on the filtered block)
                # We need a full buffer for FFT. In this simple case, block_size == n_fft.
                # If block < n_fft, we'd append to a rolling buffer.
                spectrum_db = self.analyzer.compute_spectrum(filtered_block)
                
                end_time = time.perf_counter()
                proc_time = (end_time - start_time) * 1000 # ms
                self.processing_times.append(proc_time)
                
                # Metrics
                peak_freq_idx = np.argmax(spectrum_db)
                peak_freq_hz = peak_freq_idx * (self.fs / self.block_size / 2) # Approximation
                
                if block_count % 10 == 0:
                    logger.info(f"Block {block_count}: ProcTime={proc_time:.3f}ms | PeakFreq Bin={peak_freq_idx} | MaxDB={np.max(spectrum_db):.2f}")
                
                block_count += 1
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted.")
            
        avg_time = np.mean(self.processing_times)
        logger.info(f"Processing complete. Avg processing time per block: {avg_time:.3f}ms")
        logger.info(f"Max processing time: {np.max(self.processing_times):.3f}ms")

class AdvancedDSPFeatures:
    """
    Additional signal processing utilities.
    """
    @staticmethod
    def envelope_follower(signal_in: np.ndarray, attack: float = 0.01, release: float = 0.1, fs: float = 44100.0) -> np.ndarray:
        """
        Simple envelope follower with attack/release.
        """
        g_att = math.exp(-1.0 / (fs * attack))
        g_rel = math.exp(-1.0 / (fs * release))
        
        envelope = np.zeros_like(signal_in)
        env = 0.0
        
        for i, sample in enumerate(np.abs(signal_in)):
            if sample > env:
                env = g_att * env + (1 - g_att) * sample
            else:
                env = g_rel * env + (1 - g_rel) * sample
            envelope[i] = env
            
        return envelope

    @staticmethod
    def spectral_centroid(magnitude_spectrum: np.ndarray, fs: float) -> float:
        """
        Calculate the "center of mass" of the spectrum.
        """
        freqs = np.linspace(0, fs/2, len(magnitude_spectrum))
        numerator = np.sum(freqs * magnitude_spectrum)
        denominator = np.sum(magnitude_spectrum)
        
        if denominator == 0:
            return 0.0
        return numerator / denominator

if __name__ == "__main__":
    # Test Run
    processor = RealTimeSignalProcessor(fs=48000, block_size=2048)
    processor.run_pipeline(duration_sec=2.0)
    
    # Test Envelope
    t = np.linspace(0, 1, 48000)
    sig = np.sin(2 * np.pi * 5 * t) * np.sin(2 * np.pi * 440 * t) # AM modulation
    env = AdvancedDSPFeatures.envelope_follower(sig)
    logger.info(f"Envelope follower computed. Mean level: {np.mean(env):.4f}")
