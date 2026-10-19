# Stonefish (patched) and Strathe's replay tool in a container: `--replay-tool docker:strathe-stonefish-replay`, with
# nothing to install but Docker (calibration/stonefish/README.md). Build it once, from the repository's root:
#   docker build -t strathe-stonefish-replay -f calibration/stonefish/Dockerfile calibration/stonefish
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git cmake g++ make \
      libglm-dev libsdl2-dev libfreetype-dev libgl1-mesa-dev libglu1-mesa-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*
COPY stonefish_added_mass.patch CMakeLists.txt strathe_stonefish_replay.cpp /strathe/
RUN git clone -q https://github.com/GSO-soslab/stonefish.git /src && git -C /src checkout -q 7d52673 \
    && git -C /src apply /strathe/stonefish_added_mass.patch \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
    && cmake --build /src/build -j"$(nproc)" && cmake --install /src/build \
    && cmake -S /strathe -B /strathe/build && cmake --build /strathe/build \
    && cp /strathe/build/strathe_stonefish_replay /usr/local/bin/ && rm -rf /src/build
ENTRYPOINT ["strathe_stonefish_replay"]
