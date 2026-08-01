# Chess

This repository contains **TrojanHorse** — an "undercover trash" chess bot:
a hybrid UCI engine that plays human-looking moves (from Maia, the
human-like chess AI) while a deep Stockfish/lc0 search guarantees the
position stays winning, ready to run live on Lichess as a registered bot
account via `lichess-bot`.

```
undercover-trash/
├── trojan_engine.py      the UCI hybrid engine (Maia candidates × deep validator)
├── engine.yml            engine tuning
├── lichess/config.yml    lichess-bot configuration (token, challenges)
├── scripts/              setup, weight download, lc0 build, run, self-test
├── samplers/             maia_v1 (PyTorch .pb.gz) | maia3 (Elo-conditioned) | lc0_maia
├── validators/           UCI multipv driver for Stockfish / lc0
├── maia_v1/              Maia nets executed natively in PyTorch
└── tests/                encoder / sampler / filter / UCI session tests
```

**Quick start:**

```bash
./undercover-trash/scripts/setup.sh                  # venv, deps, Stockfish, weights
# create a Lichess bot account + bot:play token, paste it in
# undercover-trash/lichess/config.yml, then:
./undercover-trash/scripts/run_bot.sh -u             # upgrade account (once)
./undercover-trash/scripts/run_bot.sh -v             # go live
```

See **[undercover-trash/README.md](undercover-trash/README.md)** for the full
documentation (architecture, configuration reference, Lichess bot-account
rules, tuning guide, troubleshooting).
