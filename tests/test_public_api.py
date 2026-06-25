import infly


def test_public_api_does_not_export_benchmark_helpers() -> None:
    assert not hasattr(infly, "BenchmarkReport")
    assert not hasattr(infly, "run_scheduler_benchmark")
