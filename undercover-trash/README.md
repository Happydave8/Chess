# TrojanHorse — the "undercover trash" chess bot

A hybrid UCI chess engine that plays the moves a weak human would play — the
kind that make you do a double-take — while a deep search secretly guarantees
the position stays winning. It is built to run as a Lichess bot through
[`lichess-bot`](https://github.com/lichess-bot/lichess-bot).

```
        ┌──────────────────┐        ┌───────────────────┐        ┌────────────────┐
        │  1. CANDIDATES   │        │  2. DEEP CHECK    │        │  3. SELECTION  │
        │  Maia (1100 Elo) │        │  Stockfish / lc0  │        │  (the filter)  │
        │  proposes the K  │        │  deep multipv     │        │  highest Maia  │
        │  moves a weak    │ ─────► │  search scores    │ ─────► │  move with     │
        │  human would     │        │  every candidate  │        │  winprob ≥ 0.45│
        │  actually play   │        │  (cp loss + WDL)  │        │  & loss ≤ 60cp │
        └──────────────────┘        └───────────────────┘        └────────────────┘
                                                                        │
                                                              bestmove + legal
```

- **Candidate sampler (the human brain).** A low-rated human model proposes
  the moves a weak player would play, ranked by human probability:
  * **maia_v1** (default): the released `maia-1100` … `maia-1900` neural nets
    (Lc0 `.pb.gz` format) executed **natively in PyTorch** — no lc0 needed,
    ~0.2 s per move on CPU.
  * **maia3**: Maia-3 Chessformer with an explicit **Elo condition**
    (`sampler.maia3.elo: 1100`) — the state of the art in human move
    prediction (checkpoints from Hugging Face).
  * **lc0_maia**: the official CSSLab setup — an lc0 binary with
    `--verbose-move-stats` and `go nodes 1`.
- **Deep validator (the safety net).** Stockfish (or lc0, or both) evaluates
  every candidate with a deep multipv search and reports centipawn loss and
  WDL win probability.
- **The Trojan Horse filter.** From the candidates, play the *highest-ranked
  Maia move* that
  1. keeps the position winning (`win_prob >= 0.45`),
  2. costs at most `max_cp_loss` (60 cp) compared with the engine's best,
  3. still *looks* imperfect (`min_cp_loss`, default 15 cp — so it is not
     the obvious engine move), and
  4. is not the engine's own #1 move.
  If nothing qualifies, the engine quietly plays the best move — the bot
  **never actually loses through its own trash act**.

The engine speaks plain UCI (`uci`, `isready`, `ucinewgame`, `position`,
`go`, `stop`, `setoption`, `quit`), so it plugs into lichess-bot, any GUI
(Arena, Cute Chess, BanksiaGUI, …) or `python-chess` directly.

---

## Quick start

```bash
# 1. one-shot setup: venv, deps, lichess-bot, maia3, Stockfish, Maia weights
cd <repo>
./undercover-trash/scripts/setup.sh

# 2. Lichess bot account + token  (see "Lichess setup" below)
echo 'LICHESS_BOT_TOKEN=lip_...' > undercover-trash/lichess/.token.env   # gitignored

# 3. upgrade the account to a bot account (irreversible, once)
./undercover-trash/scripts/run_bot.sh -u

# 4. go live (foreground)
./undercover-trash/scripts/run_bot.sh -v

#    or permanently, 24/7 (systemd):
#    sed -i 's|/home/USER/Chess|/your/checkout/path|g' undercover-trash/scripts/trojanhorse.service
#    sudo cp undercover-trash/scripts/trojanhorse.service /etc/systemd/system/
#    sudo systemctl enable --now trojanhorse
```

Offline smoke test at any time:

```bash
./undercover-trash/scripts/self_test.sh          # UCI session + bestmoves
./undercover-trash/.venv/bin/python undercover-trash/tests/run_tests.py
```

Manual UCI session:

```bash
printf 'uci\nisready\nposition startpos moves e2e4 e7e5\ngo depth 14\nquit\n' \
  | .venv/bin/python undercover-trash/trojan_engine.py --config undercover-trash/engine.yml
```

---

## Lichess setup (rules you must follow)

1. **Create a dedicated account** for the bot: <https://lichess.org/signup>.
   (An account that has already played human games **cannot** be upgraded.)
2. Create an OAuth token with the **`bot:play`** scope ("Play games with the
   bot API") at
   <https://lichess.org/account/oauth/token/create?scopes[]=bot:play> —
   the token is shown **once**. Store it **outside git**:
   ```bash
   echo 'LICHESS_BOT_TOKEN=lip_...' > undercover-trash/lichess/.token.env
   chmod 600 undercover-trash/lichess/.token.env
   ```
   `run_bot.sh` and the systemd unit read it from there; lichess-bot also
   accepts the `LICHESS_BOT_TOKEN` environment variable. The tracked
   `lichess/config.yml` keeps a placeholder and is never used to carry the
   token.
3. Upgrade to a bot account: `./scripts/run_bot.sh -u`
   (or `POST /api/bot/account/upgrade`, see
   <https://lichess.org/api#operation/botAccountUpgrade>). **Irreversible.**
4. Run the bot: `./scripts/run_bot.sh` (add `-v` for verbose logs), or
   install the systemd unit `scripts/trojanhorse.service` for a permanent
   24/7 bot (`systemctl enable --now trojanhorse`).

**Lichess rules / fair play:** Lichess allows bots, but only on accounts
that are registered as bot accounts. Bot profiles carry a public **BOT**
tag, and the Bot API refuses tokens from non-bot accounts. Do **not** use
this engine on a human account — playing with engine assistance on a normal
account violates Lichess' fair-play policy and will get the account closed.
The "undercover" in the name refers to the *style* of chess (human-looking
moves), not to hiding that the account is a bot.

---

## Running from a phone (Termux / Android)

The engine has a **pure-numpy Maia backend** (`sampler.type: maia_numpy`)
so it runs on phones where PyTorch is not available. It produces identical
moves to the PyTorch backend (verified to < 1e-5 logit difference) and needs
only `numpy` + `python-chess` + Stockfish.

**Install (inside Termux):**

```bash
pkg install -y git
git clone https://github.com/Happydave8/Chess.git
cd Chess
./undercover-trash/scripts/install_termux.sh     # everything below, automated
```

or manually:

```bash
pkg update -y && pkg upgrade -y
pkg install -y python git stockfish clang binutils python-numpy
python -m pip install python-chess pyyaml
./undercover-trash/scripts/download_weights.sh maia_v1
echo 'LICHESS_BOT_TOKEN=lip_...' > undercover-trash/lichess/.token.env
chmod 600 undercover-trash/lichess/.token.env
sed -i "s|/home/user/Chess|$PWD|g" undercover-trash/lichess/config.yml
```

**Run:**

```bash
export TROJAN_CONFIG=$PWD/undercover-trash/engine_termux.yml   # phone tuning
./undercover-trash/scripts/run_bot.sh -u                        # upgrade (once)
./undercover-trash/scripts/run_bot.sh                           # play
```

**Keep it alive while the screen is off:**

```bash
pkg install -y tmux termux-api
termux-wake-lock
tmux new-session -s bot -c $PWD './undercover-trash/scripts/run_bot.sh'
# detach:  Ctrl+B then D      reattach:  tmux attach -t bot
```

Notes:
- `engine_termux.yml` uses `maia_numpy`, `candidate_count: 8`, lighter
  search depths (10/18) and finds the Stockfish binary via `$PATH`
  (`pkg install stockfish`). Adjust for your phone's speed.
- Torch/maia3/lc0 are **not** available on Termux — the numpy backend makes
  that irrelevant.
- If Android kills the process (battery optimizations), enable the wake
  lock, keep Termux in the foreground, or use a Termux:Boot script to start
  it on boot.

---

## Configuration

Everything is in `engine.yml` (paths resolve relative to that file); each
key can also be overridden with UCI `setoption` at runtime.

| Key | Default | Meaning |
|---|---|---|
| `sampler.type` | `maia_v1` | `maia_v1` \| `maia3` \| `lc0_maia` |
| `sampler.candidate_count` | 12 | K — how many human moves to check deeply |
| `sampler.maia_v1.weights` | `weights/maia-1100.pb.gz` | which Maia net (1100 = most blunder-prone) |
| `sampler.maia3.model` | `maia3-5m` | HF alias: `maia3-5m` \| `maia3-23m` \| `maia3-79m` |
| `sampler.maia3.elo` | 1100 | the human strength to imitate |
| `sampler.lc0_maia.lc0_path` | `lc0` | lc0 binary for the official sampler |
| `validator.type` | `stockfish` | `stockfish` \| `lc0` \| `both` |
| `validator.stockfish.path` | `bin/stockfish` | SF binary (setup.sh installs it) |
| `validator.lc0.weights` | `""` | lc0 net (see `download_weights.sh lc0net`) |
| `filter.win_prob_threshold` | 0.45 | minimum win probability to allow a candidate |
| `filter.win_prob_mode` | `expected` | `expected` = win+draw/2 (recommended), `win_only` = literal WDL win share |
| `filter.max_cp_loss` | 60 | max centipawn cost vs the engine's best move |
| `filter.min_cp_loss` | 15 | min centipawn cost — how "blunder-ish" the move must look (0 = off) |
| `filter.require_not_best` | true | never play the engine's #1 move |
| `filter.selection` | `rank` | `rank` (always Maia's best qualifying) \| `softmax` (variety) |
| `filter.softmax_temp` / `top_j` | 1.5 / 3 | softmax sampling parameters |
| `time.quick_depth` | 12 | pass-A depth (finding the engine's best move) |
| `time.depth_cap` | 22 | pass-B multipv depth ceiling |
| `time.movetime_scale` | 0.5 | fraction of the clock-derived budget per move |

### Tuning the "trashiness"

* **More trash, more risk:** raise `min_cp_loss` (up to ~40), lower
  `win_prob_threshold` (down to ~0.35) or `max_cp_loss` (down to ~30) —
  the bot will play visibly worse moves that are still objectively fine.
* **More safety:** raise `win_prob_threshold` (0.5+), lower `min_cp_loss`
  (0–10). With `min_cp_loss: 0` and `require_not_best: false` the engine
  degrades to "play the best human move".
* **Variety:** `selection: softmax` samples among the top-`top_j`
  qualifying candidates, weighted by human probability.
* **Time controls:** the budget auto-scales (`remaining/40 + 0.75·inc`,
  clamped 150 ms–15 s). For bullet consider `candidate_count: 8`,
  `quick_depth: 10`, `depth_cap: 16`.

---

## Components in detail

### `maia_v1/` — native PyTorch execution of the released Maia nets

`maia-1100.pb.gz` … `maia-1900.pb.gz` are Leela Chess Zero network files
(SE-ResNet, 112-plane input, convolution policy head, WDL value head).
`maia_v1/lc0net.py` loads them straight into PyTorch and `maia_v1/encoder.py`
implements lc0's `INPUT_CLASSICAL_112_PLANE` encoder, verified against lc0's
`src/neural/encoder.cc` and the Maia training-data writer
(`chunkparser.py`). The policy output is the exact distribution lc0 with
`go nodes 1` would produce. `net_pb2.py`, `policy_index.py` and
`lc0_az_policy_map.py` are vendored from CSSLab/maia-chess (GPL-3.0,
see `maia_v1/ORIGIN.md`).

### `maia3` — Elo-conditioned human model

The [CSSLab/maia3](https://github.com/CSSLab/maia3) package is installed by
`setup.sh`. `sampler.maia3.elo` conditions the model on the strength of the
human we imitate; the checkpoint is downloaded from Hugging Face on first
use (`UofTCSSLab/Maia3-5M` etc.).

### Validators

`validators/uci_driver.py` is a small UCI client that speaks to Stockfish /
lc0 over pipes: `MultiPV` searches (set as a UCI option, which is what both
engines actually expect), `searchmoves` for per-candidate evaluation, WDL /
`winrate` parsing, and on-demand `stop`. With `validator.type: both`, a
candidate must pass **both** engines' thresholds.

---

## Files

```
undercover-trash/
├── trojan_engine.py        UCI engine (the hybrid pipeline)
├── engine.yml              engine configuration
├── run_engine.sh           launcher lichess-bot executes as `engine.name`
├── maia_v1/                Maia .pb.gz → PyTorch (encoder, net, vendored files)
├── samplers/               maia_v1 | maia3 | lc0_maia
├── validators/             UCI driver (Stockfish/lc0) + types
├── lichess/config.yml      complete lichess-bot configuration
├── scripts/
│   ├── setup.sh            venv + deps + lichess-bot + maia3 + Stockfish
│   ├── download_weights.sh Maia nets / Maia-3 / lc0 net
│   ├── build_lc0.sh        optional lc0 build from source
│   ├── run_bot.sh          lichess-bot launcher (-u upgrades the account)
│   └── self_test.sh        offline UCI smoke test
└── tests/run_tests.py      encoder / sampler / filter / UCI session tests
```

## Requirements

Python ≥ 3.10, PyTorch (CPU or CUDA build), python-chess, a Stockfish
binary (auto-installed by `setup.sh`). The Maia v1 sampler runs on CPU only;
Maia-3 and lc0 benefit from a GPU but work on CPU.

## Credits & licenses

- Maia nets and code: [CSSLab/maia-chess](https://github.com/CSSLab/maia-chess)
  (GPL-3.0) and [CSSLab/maia3](https://github.com/CSSLab/maia3).
- Lc0: [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) (GPL-3.0).
- Stockfish: [official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish) (GPL-3.0).
- Lichess bot bridge: [lichess-bot/lichess-bot](https://github.com/lichess-bot/lichess-bot) (AGPL-3.0).

This project is for research and entertainment on registered bot accounts.
