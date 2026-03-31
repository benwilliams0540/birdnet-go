# Arduino Uno Q Deployment

BirdNET-Go supports the Arduino Uno Q (Qualcomm QRB2210 / QCM2290) as a native
binary deployment. The ONNX backend with Go-native FFT preprocessing achieves
~150ms inference per 3-second audio chunk on this hardware.

## Hardware Specifications

| Component | Detail |
|-----------|--------|
| SoC | Qualcomm QRB2210 / QCM2290 |
| CPU | 4x Kryo-V2 (Cortex-A55 class) @ 2.016 GHz |
| GPU | Adreno 702 |
| RAM | ~4 GB |
| CPU flags | ARMv8 SIMD — no `asimddp`, no `i8mm`, no SVE |

## Why ONNX Backend?

The standard TFLite backend uses XNNPACK for CPU acceleration, which relies
heavily on the ARM dot-product extension (`asimddp`). The Uno Q's Kryo-V2 cores
lack this extension, making XNNPACK fall back to slower scalar paths (~292ms).

The ONNX backend splits inference into two stages:
1. **Go-native FFT preprocessing** — computes mel spectrograms using an O(N log N)
   RFFT instead of the O(N^2) DFT embedded in the full TFLite model
2. **CNN-only ONNX model** — runs just the convolutional neural network portion

This achieves ~150ms per chunk, a ~2x speedup over TFLite on this hardware.

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

### 2. Install ONNX Runtime

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

Copy from a BirdNET-Q installation or download:
```bash
sudo mkdir -p /opt/birdnet-go/model
sudo cp birdnet_cnn.onnx /opt/birdnet-go/model/
sudo cp birdnet_preproc.npz /opt/birdnet-go/model/
```

The range filter still uses TFLite and needs:
```bash
sudo cp BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite /opt/birdnet-go/model/
```

### 4. Configure

Create `/etc/birdnet-go/config.yaml`:

```yaml
birdnet:
  backend: onnx
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

## Performance Comparison

| Backend | Inference Time | Notes |
|---------|---------------|-------|
| TFLite (XNNPACK) | ~292 ms | Slower due to missing `asimddp` |
| TFLite (no XNNPACK) | ~750 ms | Single-threaded fallback |
| **ONNX (Go FFT + CNN)** | **~150 ms** | Recommended for Uno Q |

## Benchmarking

Run the built-in benchmark on the device:

```bash
/opt/birdnet-go/birdnet-go benchmark --config /etc/birdnet-go/config.yaml
```

## Known Limitations

- **No GPU acceleration**: QNN delegate is blocked by dynamic tensor shapes
  from RFFT in the TFLite model. The ONNX CNN sub-model has static shapes,
  so QNN via ONNX Runtime execution provider may work in a future release.
- **No Hexagon DSP**: The device tree lacks the fastrpc platform device node.
- **Range filter stays on TFLite**: The range filter model is lightweight and
  does not have the RFFT performance issue.

## Troubleshooting

**Service fails to start:**
```bash
journalctl -xeu birdnet-go
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
