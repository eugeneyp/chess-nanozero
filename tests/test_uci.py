"""Phase 5 tests: UCI protocol + fastchess integration.

All tests use a tiny random checkpoint (1-block/16-filter) — no trained
model required. Tests are self-contained via pytest tmp_path_factory.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

from src.neural_net.model import ChessResNet


@pytest.fixture(scope="module")
def tiny_engine(tmp_path_factory):
    """Create a tiny random checkpoint + config for subprocess tests."""
    tmp = tmp_path_factory.mktemp("uci_test")

    config = {
        "model": {
            "num_res_blocks": 1,
            "num_filters": 16,
            "input_planes": 18,
            "policy_output_size": 4672,
        },
        "mcts": {
            "num_simulations": 1,
            "c_puct": 2.0,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature_threshold_move": 30,
        },
    }
    config_path = tmp / "tiny.yaml"
    config_path.write_text(yaml.dump(config))

    model = ChessResNet.from_config(config)
    ckpt_path = tmp / "tiny.pt"
    torch.save({"epoch": 0, "model_state_dict": model.state_dict()}, ckpt_path)

    return str(config_path), str(ckpt_path)


def test_uci_protocol(tiny_engine):
    """Full UCI handshake: uci → isready → position → go → bestmove → quit."""
    config_path, ckpt_path = tiny_engine

    proc = subprocess.Popen(
        [
            sys.executable,
            "scripts/play_uci.py",
            "--config", config_path,
            "--checkpoint", ckpt_path,
            "--num-simulations", "1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def send(cmd: str) -> None:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def read_until(keyword: str, timeout: float = 60.0) -> list[str]:
        """Collect output lines until one contains keyword."""
        lines: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Non-blocking readline with a short poll
            proc.stdout.flush()
            line = proc.stdout.readline()
            if line:
                lines.append(line.strip())
                if keyword in line:
                    return lines
        raise TimeoutError(
            f"Keyword '{keyword}' not seen within {timeout}s. Got: {lines}"
        )

    try:
        send("uci")
        lines = read_until("uciok", timeout=10)
        assert any("uciok" in l for l in lines), f"uciok missing: {lines}"

        send("isready")
        lines = read_until("readyok", timeout=60)
        assert any("readyok" in l for l in lines), f"readyok missing: {lines}"

        send("position startpos")
        send("go movetime 30000")
        lines = read_until("bestmove", timeout=60)
        assert any(l.startswith("bestmove") for l in lines), (
            f"bestmove missing: {lines}"
        )

        send("quit")
        proc.stdin.close()
        proc.wait(timeout=10)
        assert proc.returncode == 0, f"Non-zero exit: {proc.returncode}"

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_match_runner(tiny_engine):
    """Run a 2-game fastchess match (1 round × 2) — no crash, returncode 0."""
    config_path, ckpt_path = tiny_engine

    # fastchess has a bug with RLIM_INFINITY: it interprets the soft limit as
    # a signed int, so ulimit -n unlimited (= RLIM_INFINITY = INT64_MAX) is
    # read as -1, causing concurrency to compute as -1.  Setting a concrete
    # soft limit (4096) avoids this.
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _soft > 65536:
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, _hard))

    engine_args = (
        f"scripts/play_uci.py "
        f"--config {config_path} --checkpoint {ckpt_path} --num-simulations 1"
    )

    result = subprocess.run(
        [
            "fastchess",
            "-engine",
            f"cmd={sys.executable}",
            f"args={engine_args}",
            "name=nanozero-test",
            "-engine",
            "cmd=stockfish", "name=stockfish-1320",
            "option.UCI_LimitStrength=true", "option.UCI_Elo=1320",
            "-each", "st=30",
            "-rounds", "1", "-repeat",
            "-recover",
            "-concurrency", "1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"fastchess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )
