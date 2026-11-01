# The Gazebo replay tool in a container: `strathe fit-sim --sim gazebo --replay-tool docker:strathe-gz-replay`, with
# nothing to install but Docker (calibration/simulators/gazebo/README.md). Build it once, from the repository's root:
#   docker build -t strathe-gz-replay calibration/simulators/gazebo
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends gz-harmonic libgz-sim8-dev cppzmq-dev cmake g++ make \
    && rm -rf /var/lib/apt/lists/*
COPY CMakeLists.txt strathe_gz_replay.cpp /src/
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release && cmake --build /build && cp /build/strathe_gz_replay /usr/local/bin/
ENV GZ_IP=127.0.0.1
ENTRYPOINT ["strathe_gz_replay"]
