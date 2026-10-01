import argparse
import gc
import json
import statistics
import time
from pathlib import Path


def check(play: bool = False) -> None:
    import psutil
    import config
    from ops.logging_setup import configure_logging
    from stt.stream_stt import StreamSTT
    from tts.playback import PlaybackEngine

    configure_logging()
    stt = StreamSTT(use_microphone=False)
    sentence = "The weather is cloudy with a temperature of sixteen degrees."
    original = config.TTS_MODEL
    measurements = {}
    try:
        stt._create("jarvis")
        for name in (config.TTS_FALLBACK_MODEL, original):
            if name in measurements or not (Path(__file__).resolve().parent.parent / name).is_file():
                continue
            config.TTS_MODEL = name
            engine = PlaybackEngine()
            model = engine.preload().result(timeout=60)
            durations = []
            for _ in range(2):
                started = time.monotonic()
                samples, rate = model.create(sentence, voice="af_bella", speed=1.15)
                durations.append(time.monotonic() - started)
            measurements[name] = statistics.median(durations)
            print(json.dumps({"model": name, "warm_synthesis_s": round(measurements[name], 3),
                              "audio_s": round(len(samples) / rate, 3),
                              "rss_mb": psutil.Process().memory_info().rss // 1048576,
                              "stt_loaded": True}), flush=True)
            if play and name == original:
                reply = "The time is eleven oh seven in the morning. " + sentence
                started = time.monotonic()
                assert engine.speak([reply]) == reply
                print(json.dumps({"check": "continuous_playback", "passed": True, "elapsed_s": round(time.monotonic() - started, 3)}), flush=True)
            engine.stop()
            del model, engine
            gc.collect()
        if len(measurements) == 2:
            baseline = measurements[config.TTS_FALLBACK_MODEL]
            improvement = 1 - measurements[original] / baseline
            print(json.dumps({"synthesis_improvement_percent": round(improvement * 100, 1)}), flush=True)
    finally:
        config.TTS_MODEL = original
        stt.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark local speech with STT loaded; microphone remains disabled")
    parser.add_argument("--play", action="store_true", help="Also play two synthetic sentences through the output device")
    check(parser.parse_args().play)
