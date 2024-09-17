import numpy as np
import lindi
import pynwb
from pynwb.ecephys import ElectricalSeries
import spikeinterface.preprocessing as spre
from nwbextractors import NwbRecordingExtractor
from qfc.codecs import QFCCodec
from qfc import qfc_estimate_quant_scale_factor

QFCCodec.register_codec()


def preprocess_ephys():
    # https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/download/&dandisetId=000463&dandisetVersion=draft
    url = "https://api.dandiarchive.org/api/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/download/"

    print('Creating LINDI file')
    with lindi.LindiH5pyFile.from_hdf5_file(url) as f:
        f.write_lindi_file("example.nwb.lindi.tar")

    cache = lindi.LocalCache()

    print('Reading LINDI file')
    with lindi.LindiH5pyFile.from_lindi_file("example.nwb.lindi.tar", mode="r", local_cache=cache) as f:
        electrical_series_path = '/acquisition/ElectricalSeries'

        print("Loading recording")
        recording = NwbRecordingExtractor(
            h5py_file=f, electrical_series_path=electrical_series_path
        )
        print(recording.get_channel_ids())

        num_frames = recording.get_num_frames()
        start_time_sec = 0
        # duration_sec = 300
        duration_sec = num_frames / recording.get_sampling_frequency()
        start_frame = int(start_time_sec * recording.get_sampling_frequency())
        end_frame = int(np.minimum(num_frames, (start_time_sec + duration_sec) * recording.get_sampling_frequency()))
        recording = recording.frame_slice(
            start_frame=start_frame,
            end_frame=end_frame
        )

        # bandpass filter
        print("Filtering recording")
        freq_min = 300
        freq_max = 6000
        recording_filtered = spre.bandpass_filter(
            recording, freq_min=freq_min, freq_max=freq_max, dtype=np.float32
        )  # important to specify dtype here
        f.close()

    traces0 = recording_filtered.get_traces(start_frame=0, end_frame=int(1 * recording_filtered.get_sampling_frequency()))
    traces0 = traces0.astype(dtype=traces0.dtype, order='C')

    # noise_level = estimate_noise_level(traces0)
    # print(f'Noise level: {noise_level}')
    # scale_factor = qfc_estimate_quant_scale_factor(traces0, target_residual_stdev=noise_level * 0.2)

    compression_method = 'zlib'
    zlib_level = 3
    zstd_level = 3

    scale_factor = qfc_estimate_quant_scale_factor(
        traces0,
        target_compression_ratio=10,
        compression_method=compression_method,
        zlib_level=zlib_level,
        zstd_level=zstd_level
    )
    print(f'Quant. scale factor: {scale_factor}')
    codec = QFCCodec(
        quant_scale_factor=scale_factor,
        dtype='float32',
        segment_length=int(recording_filtered.get_sampling_frequency() * 1),
        compression_method=compression_method,
        zlib_level=zlib_level,
        zstd_level=zstd_level
    )
    traces0_compressed = codec.encode(traces0)
    compression_ratio = traces0.size * 2 / len(traces0_compressed)
    print(f'Compression ratio: {compression_ratio}')

    print("Writing filtered recording to LINDI file")
    with lindi.LindiH5pyFile.from_lindi_file("example.nwb.lindi.tar", mode="a", local_cache=cache) as f:
        with pynwb.NWBHDF5IO(file=f, mode='a') as io:
            nwbfile = io.read()

            electrical_series = nwbfile.acquisition['ElectricalSeries']
            electrical_series_pre = ElectricalSeries(
                name="ElectricalSeries_pre",
                data=pynwb.H5DataIO(
                    recording_filtered.get_traces(),
                    chunks=(30000, recording.get_num_channels()),
                    compression=codec
                ),
                electrodes=electrical_series.electrodes,
                starting_time=0.0,  # timestamp of the first sample in seconds relative to the session start time
                rate=recording_filtered.get_sampling_frequency(),
            )
            nwbfile.add_acquisition(electrical_series_pre)  # type: ignore
            io.write(nwbfile)


def estimate_noise_level(traces):
    noise_level = np.median(np.abs(traces - np.median(traces))) / 0.6745
    return noise_level


if __name__ == "__main__":
    preprocess_ephys()