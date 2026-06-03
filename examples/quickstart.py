"""Quick start for realtime-super-res."""
from realtime_super_res import Pipeline, Config

def main():
    config = Config(batch_size=16, mixed_precision=True)
    pipeline = Pipeline(config=config)
    results = pipeline.run("sample_input")
    print(f"Status: {results['status']}, Time: {results['elapsed_seconds']:.3f}s")

if __name__ == "__main__":
    main()
