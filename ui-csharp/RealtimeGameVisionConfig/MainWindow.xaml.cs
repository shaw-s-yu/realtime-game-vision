using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;

namespace RealtimeGameVisionConfig
{
    public partial class MainWindow : Window
    {
        private ConfigEditor editor;
        private bool loading = true;

        public MainWindow()
        {
            InitializeComponent();
            // config.yaml assumed 2 levels up from bin/Debug/net8.0-windows => repo root
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));
            var configPath = Path.Combine(repoRoot, "config.yaml");
            if (!File.Exists(configPath))
            {
                // fallback to current directory parent search
                configPath = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "config.yaml"));
                if (!File.Exists(configPath))
                    configPath = "config.yaml";
            }
            editor = new ConfigEditor(configPath);
            LoadToUI();
            loading = false;
            StatusText.Text = $"Loaded {editor.ConfigPath}";
        }

        private void LoadToUI()
        {
            loading = true;
            var c = editor.Load();
            try {
                ProcessFpsSlider.Value = Convert.ToDouble(c.Get("capture.process_fps", 10));
                TargetFpsSlider.Value = Convert.ToDouble(c.Get("capture.target_fps", 30));
                WidthSlider.Value = Convert.ToDouble(c.Get("capture.output_width", 1280));
                ConfSlider.Value = Convert.ToDouble(c.Get("detector.conf", 0.25));
                IouSlider.Value = Convert.ToDouble(c.Get("detector.iou", 0.45));
                MaxDetSlider.Value = Convert.ToDouble(c.Get("detector.max_det", 100));
                SetCombo(DeviceCombo, c.Get("detector.device","cuda")?.ToString());
                OcrEnabled.IsChecked = Convert.ToBoolean(c.Get("ocr.enabled", true));
                SetCombo(OcrLangCombo, c.Get("ocr.lang","ch")?.ToString());
                OcrRoiOnly.IsChecked = Convert.ToBoolean(c.Get("ocr.roi_only", true));
                OcrDetSlider.Value = Convert.ToDouble(c.Get("ocr.det_thresh",0.3));
                OcrRecSlider.Value = Convert.ToDouble(c.Get("ocr.rec_thresh",0.5));
                ShowTrails.IsChecked = Convert.ToBoolean(c.Get("overlay.show_trails", true));
                ShowOcr.IsChecked = Convert.ToBoolean(c.Get("overlay.show_ocr", true));
                ShowLabels.IsChecked = Convert.ToBoolean(c.Get("overlay.show_labels", true));
                TrailSlider.Value = Convert.ToDouble(c.Get("overlay.trail_length",15));
                VlmEnabled.IsChecked = Convert.ToBoolean(c.Get("vlm.enabled", false));
                VlmInterval.Value = Convert.ToDouble(c.Get("vlm.interval",3));
            } catch {}
            loading = false;
        }

        private void SetCombo(ComboBox cb, string val) {
            foreach (ComboBoxItem it in cb.Items) { if ((it.Content as string)?.ToLower() == val?.ToLower()) { it.IsSelected = true; break; } }
        }
        private string GetCombo(ComboBox cb) => (cb.SelectedItem as ComboBoxItem)?.Content as string ?? "";

        private void OnValueChanged(object sender, RoutedEventArgs e)
        {
            if (loading) return;
            try {
                editor.Set("capture.process_fps", (int)ProcessFpsSlider.Value);
                editor.Set("capture.target_fps", (int)TargetFpsSlider.Value);
                editor.Set("capture.output_width", (int)WidthSlider.Value);
                editor.Set("detector.conf", Math.Round(ConfSlider.Value,2));
                editor.Set("detector.iou", Math.Round(IouSlider.Value,2));
                editor.Set("detector.max_det", (int)MaxDetSlider.Value);
                editor.Set("detector.device", GetCombo(DeviceCombo));
                editor.Set("ocr.enabled", OcrEnabled.IsChecked == true);
                editor.Set("ocr.lang", GetCombo(OcrLangCombo));
                editor.Set("ocr.roi_only", OcrRoiOnly.IsChecked == true);
                editor.Set("ocr.det_thresh", Math.Round(OcrDetSlider.Value,2));
                editor.Set("ocr.rec_thresh", Math.Round(OcrRecSlider.Value,2));
                editor.Set("overlay.show_trails", ShowTrails.IsChecked == true);
                editor.Set("overlay.show_ocr", ShowOcr.IsChecked == true);
                editor.Set("overlay.show_labels", ShowLabels.IsChecked == true);
                editor.Set("overlay.trail_length", (int)TrailSlider.Value);
                editor.Set("vlm.enabled", VlmEnabled.IsChecked == true);
                editor.Set("vlm.interval", (int)VlmInterval.Value);
                editor.Save();
                StatusText.Text = $"Saved {DateTime.Now:T} — Python hot-reloads in ~0.5s";
            } catch (Exception ex) { StatusText.Text = "Error: " + ex.Message; }
        }

        private void Save_Click(object sender, RoutedEventArgs e) { editor.Save(); StatusText.Text = "Saved manually " + DateTime.Now.ToString("T"); }
        private void Reload_Click(object sender, RoutedEventArgs e) { LoadToUI(); StatusText.Text = "Reloaded from disk"; }
    }
}
