# Arduino Uno Q Deployment

BirdNET-Go supports the Arduino Uno Q (Qualcomm QRB2210 / QCM2290) as a native
Linux `arm64` deployment.

The safest live configuration on this hardware is still:

- `backend: tflite`
- `version: "2.4"`

ONNX remains useful for comparison and debugging, and a validated split-model
NCNN path exists for experiments. Full-graph hardware acceleration is still
experimental on this board, and QNN is not a practical route for this fork.

## Hardware Specifications

| Component | Detail |
|-----------|--------|
| SoC | Qualcomm QRB2210 / QCM2290 |
| CPU | 4x Kryo-V2 (Cortex-A55 class) @ 2.016 GHz |
| GPU | Adreno 702 |
| RAM | ~4 GB |
| CPU flags | ARMv8 SIMD — no `asimddp`, no `i8mm`, no SVE |

## Current Backend Guidance

Use TFLite as the production baseline on the Uno Q.

- `tflite`: best default for a stable portable deployment
- `onnx`: useful for validation and backend comparison
- `ncnn`: available only when a validated split-model directory is present

For the current Uno Q-specific notes, see:

- [`docs/unoq/README.md`](../../docs/unoq/README.md)
- [`docs/unoq/acceleration-summary.md`](../../docs/unoq/acceleration-summary.md)
- [`docs/unoq/ncnn-conversion.md`](../../docs/unoq/ncnn-conversion.md)

## Quick Install

```bash
sudo bash install-uno-q.sh
```

The installer handles hardware detection, directory setup, ONNX Runtime library
installation, and systemd service creation.

## Manual Install

### 1. Build the binary

From a machine with Go and the ARM64 cross-compiler:

```bash
# Using Taskfile
task linux_arm64

# Or manually
GOOS=linux GOARCH=arm64 CGO_ENABLED=1 \
  CC=aarch64-linux-gnu-gcc go build -o birdnet-go .
```

Copy the binary to the Uno Q:
```bash
scp birdnet-go arduino@uno-q:/opt/birdnet-go/birdnet-go
```

### 2. Install runtime dependencies

The Uno Q build can use multiple inference backends. ONNX Runtime stays
runtime-loaded, while TFLite support is built into the binary.

If you want ONNX available:

```bash
ORT_VERSION=1.17.1
curl -sSL -o /tmp/ort.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
tar -xzf /tmp/ort.tgz -C /tmp
sudo cp /tmp/onnxruntime-linux-aarch64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION} /usr/local/lib/
sudo ln -sf libonnxruntime.so.${ORT_VERSION} /usr/local/lib/libonnxruntime.so
sudo ldconfig
```

### 3. Install model files

For TFLite-only deployment, the embedded model is enough.

If you want ONNX or NCNN available, place the external model assets under a
dedicated model directory such as `/opt/birdnet-go/model/` and reference them
from `config.yaml`.

### 4. Configure

Create `/etc/birdnet-go/config.yaml`:

```yaml
birdnet:
  backend: tflite
  version: "2.4"
  threads: 4
  usexnnpack: false
  sensitivity: 1.0
  threshold: 0.8
  locale: en

  rangefilter:
    model: latest
    threshold: 0.01

webserver:
  enabled: true
  port: 8080

realtime:
  audio:
    source: sysdefault
    export:
      enabled: true
      type: wav

output:
  sqlite:
    enabled: true
    path: /var/lib/birdnet-go/data/database/birdnet.db
```

### 5. Create systemd service

Create `/etc/systemd/system/birdnet-go.service`:

```ini
[Unit]
Description=BirdNET-Go Bird Sound Identification
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=arduino
Group=arduino
WorkingDirectory=/opt/birdnet-go
Environment=ORT_LIBRARY_PATH=/usr/local/lib/libonnxruntime.so
Environment=LD_LIBRARY_PATH=/usr/local/lib
ExecStart=/opt/birdnet-go/birdnet-go serve --config /etc/birdnet-go/config.yaml
Restart=always
RestartSec=5
SupplementaryGroups=audio
MemoryMax=512M

[Install]
WantedBy=multi-user.target
```

**Important:** Write this as a real file, not a symlink. The Uno Q's
`/home/arduino` is on a separate partition that may not be mounted when systemd
loads unit files during early boot.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now birdnet-go
```

### 6. Verify

```bash
sudo systemctl status birdnet-go
journalctl -fu birdnet-go
```

Open `http://<uno-q-ip>:8080` in a browser.

## Known Limitations

- **QNN is not supported for this target**: the Uno Q's Hexagon DSP is not a
  practical BirdNET-Go acceleration route in this fork.
- **GPU acceleration remains experimental**: split-model NCNN can be validated,
  but it is not the default production recommendation.
- **Spectrogram generation is CPU-side**: installing SoX on the Uno Q improves
  dashboard spectrogram generation substantially compared with FFmpeg fallback.

## Troubleshooting

**Service fails to start:**
```bash
journalctl -xeu birdnet-go
```

**Slow spectrogram generation:**
```bash
sudo apt-get install -y sox libsox-fmt-all
```

**ONNX Runtime not found:**
```bash
# Check library is installed
ls -la /usr/local/lib/libonnxruntime*
# Check ldconfig
ldconfig -p | grep onnxruntime
# Set explicit path
export ORT_LIBRARY_PATH=/usr/local/lib/libonnxruntime.so
```

**No audio device:**
```bash
# List audio devices
arecord -l
# Ensure user is in audio group
sudo usermod -aG audio arduino
```
