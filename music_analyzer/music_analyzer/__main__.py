import logging
import os
from pathlib import Path

import numpy as np
import pyaudio
from scipy import signal

from . import util

SAMPLE_RATE = 44100
FRAMES_PER_BUFFER = 1024

FIFO_PATH = Path("/tmp/music_analyzer_fifo")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_audio_stream() -> tuple[pyaudio.PyAudio, pyaudio.Stream]:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )
    return p, stream


def main() -> None:
    logger.info("Starting music analyzer")

    try:
        os.mkfifo(FIFO_PATH)
    except FileExistsError:
        # fifo already exists
        pass

    # this blocks until the fifo is opened for writing by something else
    logger.info("Waiting for fifo to be opened by another process")
    fifo = FIFO_PATH.open("w")

    logger.info("Starting audio stream")
    p, stream = _get_audio_stream()

    try:
        for _ in range(10):
            # Read data from the audio input stream
            data = np.frombuffer(stream.read(FRAMES_PER_BUFFER), dtype=np.int16)

            # Split data into left and right channels
            left_data = data[::2]
            right_data = data[1::2]

            # Perform initial STFT on the data to generate the frequency bins
            frequencies_l, times_l, Zxx_l = signal.stft(
                left_data,
                fs=SAMPLE_RATE,
                nperseg=FRAMES_PER_BUFFER,
                noverlap=512,
                window="hann",
            )

        bin_index_map = util.get_bin_indexes(frequencies_l, 40, 16000, 24)

        i = 0
        while True:
            # Read data from the audio input stream
            data = np.frombuffer(stream.read(FRAMES_PER_BUFFER), dtype=np.int16)

            # Split data into left and right channels
            left_data = data[::2]
            right_data = data[1::2]

            # Perform STFT on the left data
            frequencies_l, times_l, Zxx_l = signal.stft(
                left_data,
                fs=SAMPLE_RATE,
                nperseg=FRAMES_PER_BUFFER,
                noverlap=512,
                window="hann",
            )
            # TODO: isintance is bad for performance
            assert isinstance(Zxx_l, np.ndarray)

            # Zxx contains the STFT result, which is complex-valued.
            # Take the absolute value to get magnitude
            magnitude = np.abs(Zxx_l[:, -1])

            averaged_bins = [
                magnitude[start:end].mean() for start, end in bin_index_map
            ]

            # Print the magnitude of the left STFT result
            l_bins_data = f"l:{i}:{','.join(map(str, averaged_bins))}"
            logger.debug(repr(l_bins_data))
            fifo.write(l_bins_data + "\n")

            # Perform STFT on the right data
            frequencies_r, times_r, Zxx_r = signal.stft(
                right_data,
                fs=SAMPLE_RATE,
                nperseg=FRAMES_PER_BUFFER,
                noverlap=512,
                window="hann",
            )
            assert isinstance(Zxx_r, np.ndarray)

            # Zxx contains the STFT result, which is complex-valued.
            # Take the absolute value to get magnitude
            magnitude = np.abs(Zxx_r[:, -1])

            averaged_bins = [
                magnitude[start:end].mean() for start, end in bin_index_map
            ]

            # Print the magnitude of the right STFT result
            r_bins_data = f"r:{i}:{','.join(map(str, averaged_bins))}"
            logger.debug(repr(r_bins_data))
            fifo.write(r_bins_data + "\n")
            # without flushing the fifo gets written to in chunks
            fifo.flush()
            i += 1

    except KeyboardInterrupt:
        # TODO: https://github.com/ZhangJianAo/safe-exit
        logger.info("Keyboard interrupt detected, exiting")
        # Clean up the FIFO
        fifo.close()
        FIFO_PATH.unlink()

        # Stop the stream and close PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
