# Stonefish and Gazebo with nothing to build: Docker

Strathe calibrates your own Stonefish or Gazebo model by replaying your logs in that simulator
(`fit-stonefish`, `export-stonefish`, `describe-stonefish`, `fit-sim`, `export-sim`, `convert-sim`). Normally that
needs the simulator installed and Strathe's small replay tool compiled against it.

**With Docker you skip all of that.** Strathe builds a ready-made Linux image with the simulator and its replay tool,
once. Every Stonefish and Gazebo command then uses it, on a Mac, Linux or Windows, and gives the same results as a
local build:
- **Gazebo:** the same windows replayed both ways agree to 10⁻⁹ m/s, the precision the replay writes its numbers at.
  A full `fit-sim` ran through it.
- **Stonefish:** URI's full ALPHA RISE export run through the image matches the local build's in all 242 numbers
  (PROJECT_STATUS.md #92).

```bash
python strathe.py setup-sim gazebo --docker      # once: builds Strathe's Gazebo image and checks it
python strathe.py fit-sim --sim gazebo --world my_world.sdf --data my_logs --reference-point 0 0 0
```

Nothing else changes. The second command finds no Gazebo replay tool on the machine, so it uses the image and
says so on its first line.

## 1. Install Docker (once)

| Your machine | Install | Notes |
|---|---|---|
| **Mac** | [Docker Desktop](https://www.docker.com/products/docker-desktop/), or **Colima** without admin rights: `brew install colima docker` then `colima start --cpu 4 --memory 4 --disk 40` | Colima is free and open source; it shares only your home folder (section 5) |
| **Linux** | [Docker Engine](https://docs.docker.com/engine/install/), then `sudo usermod -aG docker $USER` and log in again | Runs at full speed: no virtual machine |
| **Windows** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) (it uses WSL 2) | Not yet tested on Windows (section 5) |

- **Licence:** Docker Desktop is free for personal use, education and small companies (under 250 employees and under
  $10 million revenue). A larger company needs a paid Docker subscription, or uses Colima or Docker Engine, which are
  free.
- **Resources:** give Docker at least 4 GB of memory and 4 cores. Docker Desktop: Settings > Resources. Colima: the
  `colima start` options above.
- **Disk:** the Stonefish image is 0.5 GB and the Gazebo image 2.7 GB, plus about 3 GB of build cache that
  `docker builder prune` frees.

**Check it:** `python strathe.py check` has a Docker line:

```
✅ Docker 29.5.2, running; Strathe's images built: none yet (python strathe.py setup-sim stonefish / gazebo --docker)
```

## 2. Build Strathe's image (once per simulator)

```bash
python strathe.py setup-sim stonefish --docker   # Stonefish (patched) + the replay tool: about 8 minutes
python strathe.py setup-sim gazebo --docker      # Gazebo Harmonic + the replay tool: about 4 minutes
```

The times are from an M1 MacBook Air with 4 cores given to Docker. Each build:
- downloads Ubuntu 24.04 and the simulator;
- compiles the replay tool (and, for Stonefish, the fork with Strathe's added-mass and damping patch);
- finishes with the same line `check` prints:

```
✅ Stonefish replay tool in Docker (strathe-stonefish-replay), with the patch: per-axis added mass and Strathe's damping
✅ Gazebo replay tool in Docker (strathe-gz-replay) (gz-sim 8.15.0)
```

**On the web page** (`python strathe.py web`), the card "Run Stonefish and Gazebo with nothing to build (Docker)" shows
Docker's state and builds either image with one button. The build's progress shows in the job's log.

**After updating Strathe,** run the same command again to rebuild. Docker reuses what hasn't changed.

## 3. Use it

**You don't have to do anything.** When the replay tool isn't built on this machine and Strathe's image is, every
Stonefish and Gazebo command uses the image. Its first line says so:

```
Using Strathe's Docker image strathe-gz-replay: the Gazebo replay tool isn't built here (calibration/simulators/DOCKER.md)
```

**To choose explicitly,** use `--replay-tool`:

| `--replay-tool` | Uses |
|---|---|
| (not given) | the tool built here, else Strathe's image if it's built |
| `docker` | Strathe's image for that simulator, even when a local tool is built |
| `docker:IMAGE` | any image whose entry point speaks the replay contract: your own simulator's, or a copy of Strathe's under another name |
| a path | that tool |

In the web forms, leave "The replay tool" empty, or type `docker`.

```bash
python strathe.py fit-stonefish --scenario my_world.scn --robot my_auv --part Hull --data my_logs --reference-point 0 0 0
python strathe.py export-stonefish --replay-tool docker --calibration calibration_output_my_auv --scenario my_world.scn --robot my_auv --part Hull --reference-point 0 0 0
python strathe.py fit-sim --sim gazebo --replay-tool docker --world my_world.sdf --data my_logs --reference-point 0 0 0
```

## 4. What happens when it runs

- **One container per replay:** `docker run --rm` starts the image, replays a batch of windows and exits. Nothing
  opens on screen; the replays are headless.
- **Your folders, at the same paths:** your model's folder, the simulator's data folder (meshes), any folder your model
  refers to by full path, and the work folder are mounted at the same paths inside. On Windows, `C:\x` is `/c/x`
  inside.
- **It runs as you:** the files it writes are yours, not root's.
- **Work folders** go under `~/.strathe/tmp`, which every Docker set-up can see. Strathe removes them after the run;
  anything left there for over two days is cleared.
- **Before the first replay,** Strathe checks that Docker can see each folder. Docker otherwise mounts an empty folder
  without saying so, and the replay's results would vanish. A folder it can't see is named, with the fix (section 5).
- **To look at the calibrated model,** you need Gazebo or Stonefish on your machine, for example
  `gz sim calibrated_world/world.sdf` from a Gazebo fit. The images have no display. Calibrating doesn't need one.

**Speed:**
- **On a Mac,** Docker runs in a Linux virtual machine. Gazebo replays were about 1.5× slower than the local build
  (40 windows: 130 s against 86 s).
- **On Linux,** Docker runs natively, at about local speed.
- **`--workers`** runs several replays at once, each in its own container.

## 5. If something goes wrong

| Strathe says | Do this |
|---|---|
| Docker isn't installed | Section 1 |
| Docker is installed but its engine isn't running | Open Docker Desktop, or `colima start`, or `sudo systemctl start docker` on Linux |
| Strathe's gazebo image (strathe-gz-replay) isn't built yet | `python strathe.py setup-sim gazebo --docker` |
| Docker can't see /some/folder | **Colima** shares only your home folder: move your model or data under it, or restart Colima sharing the folder (`colima stop && colima start --mount /some/folder:w`). **Docker Desktop:** Settings > Resources > File sharing. **Linux:** Docker sees every folder; check the path |
| the image didn't build | Docker's output is printed above the message. The usual causes are no internet connection, or Docker's disk is full: `docker system df` shows it, `docker system prune` frees it |
| the build stops while compiling Stonefish, with "Killed" | Docker has too little memory: give it 4 GB or more (section 1) |

- **Apple Silicon or Intel:** the image is built for your machine's processor. There's nothing to choose.
- **Windows:** Strathe translates the paths (`C:\x` → `/c/x`), but this hasn't been run on a Windows machine yet.
  Running Strathe inside WSL 2, where it behaves as on Linux, is the safer route for now.
- **To remove everything:** `docker image rm strathe-stonefish-replay strathe-gz-replay`, then
  `docker builder prune`. With Colima, `colima delete` also removes its virtual machine.

## 6. Your own simulator in Docker

Any image works with `--replay-tool docker:IMAGE` (or `replay: docker:IMAGE` in your simulator's YAML) if its entry
point is your replay command and speaks the contract in [SIMULATOR_PROTOCOL.md](SIMULATOR_PROTOCOL.md). It runs as your
user id, with `HOME=/tmp`. So it must not need root, and must write only to the files it's given and to `/tmp`. The two
Dockerfiles Strathe uses are short examples: `calibration/stonefish/Dockerfile` and
`calibration/simulators/gazebo/Dockerfile`.
