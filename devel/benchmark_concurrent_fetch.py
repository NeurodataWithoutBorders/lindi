"""Benchmark concurrent chunk fetching vs serial reads from DANDI Archive.

Compares the old path (h5py + LindiRemfile, serial) against the new path
(zarr + LindiH5ZarrStore, concurrent) for reading external array links.

Produces a bar chart showing timings and speedup for each test case.

Usage:
    python devel/benchmark_concurrent_fetch.py
    python devel/benchmark_concurrent_fetch.py --dandiset 000473
    python devel/benchmark_concurrent_fetch.py --dandiset 000409
    python devel/benchmark_concurrent_fetch.py --output benchmark_results.png
"""
import argparse
import time
import tempfile
import numpy as np
import h5py
import zarr
import matplotlib.pyplot as plt
import lindi
from lindi.LindiRemfile.LindiRemfile import LindiRemfile
from lindi.LindiH5ZarrStore.LindiH5ZarrStore import LindiH5ZarrStore
from lindi.LindiH5ZarrStore.LindiH5ZarrStoreOpts import LindiH5ZarrStoreOpts


DANDISETS = {
    "000473": {
        "url": "https://api.dandiarchive.org/api/assets/11f512ba-5bcf-4230-a8cb-dc8d36db38cb/download/",
        "dataset": "processing/ecephys/LFP/LFP/data",
        "label": "000473 LFP",
        "slices": [
            (np.s_[:1000], "[:1000]"),
            (np.s_[:5000], "[:5000]"),
            (np.s_[:10000], "[:10000]"),
        ],
    },
    "000409": {
        "url": "https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/",
        "dataset": "acquisition/ElectricalSeriesAp/data",
        "label": "000409 Neuropixels",
        "slices": [
            (np.s_[:500], "[:500]"),
            (np.s_[:1000], "[:1000]"),
            (np.s_[:2000], "[:2000]"),
        ],
    },
}


def read_serial(url, dataset_name, selection):
    """Old path: serial reads through h5py + LindiRemfile."""
    remf = LindiRemfile(url, verbose=False, local_cache=None)
    with h5py.File(remf, "r") as f:
        return f[dataset_name][selection]


def read_concurrent(url, dataset_name, selection):
    """New path: concurrent reads through zarr + LindiH5ZarrStore."""
    opts = LindiH5ZarrStoreOpts(num_dataset_chunks_threshold=None)
    with LindiH5ZarrStore.from_file(url, opts=opts) as store:
        arr = zarr.open_array(store=store, path=dataset_name, mode="r")
        return arr[selection]


def read_lindi_file(lindi_json_path, dataset_name, selection):
    """Full pipeline: LindiH5pyFile (concurrent path for remote external links)."""
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_json_path, mode="r")
    return f[dataset_name][selection]


def benchmark_one(func, *args, **kwargs):
    """Time a single call. Returns (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_benchmark(dandiset_key):
    """Run benchmarks for one dandiset. Returns list of result dicts."""
    info = DANDISETS[dandiset_key]
    url = info["url"]
    dataset = info["dataset"]
    label = info["label"]

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {label}")
    print(f"  URL: {url}")
    print(f"  Dataset: {dataset}")
    print(f"{'=' * 60}")

    results = []
    for selection, sel_label in info["slices"]:
        print(f"\n  Slice: {dataset}{sel_label}")
        print(f"  {'-' * 40}")

        # Serial (old path)
        data_serial, t_serial = benchmark_one(read_serial, url, dataset, selection)
        print(f"    Serial (h5py+LindiRemfile): {t_serial:.2f}s")

        # Concurrent (new path)
        data_concurrent, t_concurrent = benchmark_one(read_concurrent, url, dataset, selection)
        print(f"    Concurrent (zarr+LindiH5ZarrStore): {t_concurrent:.2f}s")

        # Equivalence check
        np.testing.assert_array_equal(data_serial, data_concurrent)
        print(f"    Data equivalent: shape={data_serial.shape}, dtype={data_serial.dtype}")

        speedup = t_serial / t_concurrent if t_concurrent > 0 else float("inf")
        print(f"    Speedup: {speedup:.1f}x")

        results.append({
            "dandiset": dandiset_key,
            "label": f"{label}\n{sel_label}",
            "short_label": sel_label,
            "serial": t_serial,
            "concurrent": t_concurrent,
            "speedup": speedup,
            "shape": data_serial.shape,
        })

    return results


def run_lindi_file_benchmark(dandiset_key):
    """Run the full LindiH5pyFile pipeline benchmark."""
    info = DANDISETS[dandiset_key]
    url = info["url"]
    dataset = info["dataset"]
    selection, sel_label = info["slices"][1]  # use middle slice

    print(f"\n{'=' * 60}")
    print(f"LindiH5pyFile Pipeline: {info['label']}")
    print(f"{'=' * 60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Creating .lindi.json...")
        t0 = time.perf_counter()
        fname = f"{tmpdir}/test.nwb.lindi.json"
        with lindi.LindiH5pyFile.from_hdf5_file(url) as f:
            f.write_lindi_file(fname)
        t_create = time.perf_counter() - t0
        print(f"    Created in {t_create:.2f}s")

        # Serial baseline
        data_serial, t_serial = benchmark_one(read_serial, url, dataset, selection)
        print(f"    Serial (h5py+LindiRemfile): {t_serial:.2f}s")

        # LindiH5pyFile (uses concurrent path)
        data_lindi, t_lindi = benchmark_one(read_lindi_file, fname, dataset, selection)
        print(f"    LindiH5pyFile (concurrent): {t_lindi:.2f}s")

        np.testing.assert_array_equal(data_serial, data_lindi)
        speedup = t_serial / t_lindi if t_lindi > 0 else float("inf")
        print(f"    Data equivalent: shape={data_serial.shape}")
        print(f"    Speedup: {speedup:.1f}x")

    return {
        "label": f"LindiH5pyFile\n{info['label']} {sel_label}",
        "serial": t_serial,
        "concurrent": t_lindi,
        "speedup": speedup,
    }


def plot_results(all_results, output_path=None):
    """Create a bar chart comparing serial vs concurrent timings."""
    labels = [r["label"] for r in all_results]
    serial_times = [r["serial"] for r in all_results]
    concurrent_times = [r["concurrent"] for r in all_results]
    speedups = [r["speedup"] for r in all_results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    bars_serial = ax.bar(x - width / 2, serial_times, width, label="Serial (h5py + LindiRemfile)", color="#d35f5f")
    bars_concurrent = ax.bar(x + width / 2, concurrent_times, width, label="Concurrent (zarr + LindiH5ZarrStore)", color="#5f9ed3")

    # Add speedup annotations
    for i, (s_time, c_time, speedup) in enumerate(zip(serial_times, concurrent_times, speedups)):
        y = max(s_time, c_time)
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(i, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Time (seconds)")
    ax.set_title("LINDI Chunk Fetching: Serial vs Concurrent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, max(serial_times) * 1.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark concurrent chunk fetching from DANDI")
    parser.add_argument(
        "--dandiset",
        choices=list(DANDISETS.keys()),
        default=None,
        help="Run benchmarks for a specific dandiset only (default: run all)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save plot to file instead of displaying (e.g. benchmark.png)",
    )
    parser.add_argument(
        "--skip-lindi-file",
        action="store_true",
        help="Skip the full LindiH5pyFile pipeline benchmark (slow due to .lindi.json creation)",
    )
    args = parser.parse_args()

    dandisets = [args.dandiset] if args.dandiset else list(DANDISETS.keys())

    all_results = []
    for key in dandisets:
        all_results.extend(run_benchmark(key))

    if not args.skip_lindi_file:
        for key in dandisets:
            all_results.append(run_lindi_file_benchmark(key))

    print(f"\n{'=' * 60}")
    print("All data equivalence checks passed!")
    print(f"{'=' * 60}")

    plot_results(all_results, output_path=args.output)


if __name__ == "__main__":
    main()
