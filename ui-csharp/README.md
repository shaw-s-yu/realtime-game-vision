# C# WPF Config UI — optional alternative to Python Dear PyGui panel

This folder contains a minimal WPF app skeleton that edits `config.yaml` live while Python app watches file changes via ConfigManager hot-reload.

Why C# WPF instead of Python UI?
- Native Windows look, easy drag-drop designer in Visual Studio
- Separate process so Python CV loop not blocked by UI thread
- Good if your team is C# heavy and wants to extend to full Windows console app later

Why Python Dear PyGui panel is default in repo?
- Single pip install, no Visual Studio needed, runs in same repo, cross-platform.
- Already integrated: `python -m src.main --ui`

Choose one, not both needed.

## WPF project structure
```
ui-csharp/
  RealtimeGameVisionConfig.sln
  RealtimeGameVisionConfig/
    MainWindow.xaml
    MainWindow.xaml.cs
    ConfigEditor.cs
    App.xaml
    App.xaml.cs
    RealtimeGameVisionConfig.csproj
```

## How it works
1. WPF app loads `..\..\config.yaml` relative to exe output (adjust path in ConfigEditor.cs).
2. UI shows sliders / checkboxes bound to yaml values matching Python config structure.
3. On value change -> writes yaml immediately via YamlDotNet library.
4. Python app has ConfigManager with auto_reload=True polling file mtime every 0.5s -> picks up change next frame without restart. No socket needed.

This is file-based IPC — simplest robust cross-language method. Latency <0.5s which is fine for tuning.

## Build steps on Windows
```powershell
cd ui-csharp
# requires .NET 8 SDK installed: winget install Microsoft.DotNet.SDK.8
dotnet restore
dotnet build -c Release
dotnet run --project RealtimeGameVisionConfig
# or open RealtimeGameVisionConfig.sln in Visual Studio 2022 and F5
```

Make sure Python app is running with default config path so both point to same `config.yaml` in repo root. In WPF app adjust `ConfigPath` in ConfigEditor.cs if needed.

## Hot reload behavior in Python
Python main loop already polls config file mtime every 0.5 sec via ConfigManager. Parameters that apply live without restart:
- capture.process_fps, capture.target_fps
- detector.conf, detector.iou, detector.max_det
- overlay.show_trails, overlay.show_ocr, overlay.show_labels, overlay.trail_length
- ocr.enabled, ocr.det_thresh, ocr.rec_thresh
- vlm.enabled, vlm.interval

Parameters requiring restart:
- capture.output_width / output_height (needs recreate dxcam)
- detector.model / device (needs reload YOLO weights)
- ocr.lang (needs reinit RapidOCR models)
For those, WPF UI shows "(restart required)" label, or Python UI disables slider.

## Extending to named pipes or TCP later
If you outgrow file polling latency, replace ConfigEditor.cs Save() with named pipe client sending JSON patch to Python named pipe server. Python side already has ConfigManager.update() thread-safe method ready for socket input. File approach is 90% good enough for tuning UI at human speed.
