import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime
import time
import sys
import os
import threading
import queue
import json
import re  # for phone validation/normalization

# =====================  MANAGED TWILIO DEFAULTS  ============================
# Preferred: set environment variables before running the app.
TWILIO_SID_DEFAULT   = os.getenv("SECURESIGHT_TWILIO_SID",   "")      # e.g. "ACxxxxxxxx..."
TWILIO_TOKEN_DEFAULT = os.getenv("SECURESIGHT_TWILIO_TOKEN", "")      # e.g. "xxxxxxxx..."
TWILIO_FROM_DEFAULT  = os.getenv("SECURESIGHT_TWILIO_FROM",  "whatsapp:+14155238886")  # Twilio WA sandbox or approved sender
# If you want to hardcode instead of env vars, set them here directly:
# TWILIO_SID_DEFAULT   = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# TWILIO_TOKEN_DEFAULT = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# TWILIO_FROM_DEFAULT  = "whatsapp:+14155238886"
# ===========================================================================

# --- Windows console: prefer UTF-8 and add a safe print ----------------------
def _configure_windows_console_utf8():
    if sys.platform.startswith("win"):
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

_configure_windows_console_utf8()

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        def _fix(x):
            s = str(x)
            try:
                return s.encode(enc, errors="replace").decode(enc, errors="ignore")
            except Exception:
                return s.encode("ascii", errors="replace").decode("ascii")
        print(*map(_fix, args), **kwargs)

# --- IMPORT FOR TWILIO ---
try:
    from twilio.rest import Client
except ImportError:
    Client = None
    safe_print("Twilio module not found. Please run: pip install twilio")

# ------------------------ Model ------------------------ #
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, 128)
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

# ------------------------ Utilities ------------------------ #
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

USER_ICON_DIR = r"C:\Users\Lenovo\Downloads\desktop\SECURESIGHT\SECURE-SIGHT-AI\multicamera"
USER_ICON_ICO = os.path.join(USER_ICON_DIR, "icon.ico")
USER_ICON_PNG = os.path.join(USER_ICON_DIR, "icon.png")

PREFS_PATH = resource_path("securesight_prefs.json")
DEFAULT_PREFS = {
    "snapshot_dir": resource_path("snapshots"),
    "auto_snapshot_on_crime": True,
    "grid_max_cols": 3,
    "record_codec": "mp4v",
    
    # [NEW] Added cameras key for persistence
    "cameras": {},

    # --- Twilio Preferences (managed by app; user doesn't edit) ---
    "alert_to_name": "",          # auto = profile name
    "alert_to_num": "",           # auto = whatsapp:+<profile number>
    "twilio_sid": TWILIO_SID_DEFAULT,
    "twilio_token": TWILIO_TOKEN_DEFAULT,
    "twilio_from_num": TWILIO_FROM_DEFAULT,

    # --- Profile (user provides) ---
    "user_name": "",
    "user_mobile": ""             # E.164, e.g., +91XXXXXXXXXX
}

# ------------------------ Toast ------------------------ #
class Toast:
    def __init__(self, master):
        self.master = master
        self._win = None

    def show(self, text, duration=2000):
        if self._win is not None:
            try:
                self._win.destroy()
            except Exception:
                pass
        self._win = tk.Toplevel(self.master)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        frame = tk.Frame(self._win, bg="#2b2b2b", bd=0, highlightthickness=0)
        frame.pack(fill="both", expand=True)
        lbl = tk.Label(frame, text=text, bg="#2b2b2b", fg="#eaeaea", padx=14, pady=10, font=("Segoe UI", 10))
        lbl.pack()
        self._win.update_idletasks()
        x = self.master.winfo_rootx() + self.master.winfo_width() - self._win.winfo_width() - 20
        y = self.master.winfo_rooty() + self.master.winfo_height() - self._win.winfo_height() - 20
        self._win.geometry(f"+{x}+{y}")
        self.master.after(duration, lambda: self._win and self._win.destroy())

# ------------------------ Camera Thread ------------------------ #
class CameraThread(threading.Thread):
    def __init__(self, camera_name, camera_source, model, device, transform, data_queue, widget, clip_len=16):
        super().__init__(daemon=True)
        self.camera_name = camera_name
        self.camera_source = camera_source
        self.model = model
        self.device = device
        self.transform = transform
        self.data_queue = data_queue
        self.widget = widget
        self.clip_len = clip_len
        self._stop_event = threading.Event()
        self._record_flag = threading.Event()
        self._writer = None
        self._fps_calc_last = time.time()
        self._frames = 0

    def stop(self):
        self._stop_event.set()

    def toggle_recording(self, enable: bool):
        if enable:
            self._record_flag.set()
        else:
            self._record_flag.clear()
            try:
                if self._writer:
                    self._writer.release()
            finally:
                self._writer = None

    def _ensure_writer(self, frame, prefs):
        if self._writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*prefs.get("record_codec", "mp4v"))
        os.makedirs(prefs.get("snapshot_dir", resource_path("snapshots")), exist_ok=True)
        out_dir = prefs.get("snapshot_dir")
        fname = f"{self.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(out_dir, fname)
        h, w = frame.shape[:2]
        self._writer = cv2.VideoWriter(out_path, fourcc, 25.0, (w, h))
        self.data_queue.put((self.camera_name, None, None, f"INFO: Recording to {out_path}", 0.0))

    def run(self):
        frame_buffer = []
        cap = None
        try:
            try:
                source_to_open = int(self.camera_source)
            except ValueError:
                source_to_open = self.camera_source
            cap = cv2.VideoCapture(source_to_open)
            if not cap.isOpened():
                raise ConnectionError(f"Cannot open camera source: {self.camera_source}")

            last_status = "IDLE"
            last_confidence = 0.0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.data_queue.put((self.camera_name, None, None, "ERROR: FEED LOST", 0.0))
                    break

                # FPS calc
                self._frames += 1
                now = time.time()
                if now - self._fps_calc_last >= 1.0:
                    fps = self._frames / (now - self._fps_calc_last)
                    self._fps_calc_last = now
                    self._frames = 0
                    self.data_queue.put((self.camera_name, None, None, f"FPS:{fps:.1f}", 0.0))

                # Recording
                if self._record_flag.is_set():
                    app_prefs = App.read_prefs_silent()
                    self._ensure_writer(frame, app_prefs)
                    if self._writer:
                        try:
                            self._writer.write(frame)
                        except Exception:
                            pass

                # --- Model Inference (unchanged) ---
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor_img = self.transform(pil_img)
                frame_buffer.append(tensor_img)

                if len(frame_buffer) == self.clip_len:
                    clip_tensor = torch.stack(frame_buffer).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = self.model(clip_tensor)
                        probs = F.softmax(output, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        confidence = probs.max().item() * 100
                    last_status = "CRIME" if pred == 1 else "NORMAL"
                    last_confidence = confidence
                    frame_buffer.pop(0)

                # --- UI Frame Processing ---
                imgtk = None
                if self.widget and self.widget._vp_w and self.widget._vp_h:
                    try:
                        frame_resized = cv2.resize(frame, (self.widget._vp_w, self.widget._vp_h))
                        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img_rgb)
                        imgtk = ImageTk.PhotoImage(image=img)
                    except Exception:
                        imgtk = None

                self.data_queue.put((self.camera_name, frame, imgtk, last_status, last_confidence))
                time.sleep(0.006)
        except Exception as e:
            try:
                self.data_queue.put((self.camera_name, None, None, f"ERROR: {e}", 0.0))
            except Exception:
                pass
        finally:
            try:
                if self._writer:
                    self._writer.release()
            except Exception:
                pass
            if cap:
                cap.release()
            safe_print(f"Thread for {self.camera_name} stopped.")

# ------------------------ Video Tile (16:9 Canvas) ------------------------ #
class VideoDisplayWidget(tk.Frame):
    def __init__(self, parent, app, camera_name, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self.camera_name = camera_name
        self.config(bg="#0e0e0e", bd=0, highlightthickness=0)
        self.grid_propagate(False)

        self.COLOR_TEXT = "#dcdcdc"
        self.COLOR_IDLE = "#6c757d"

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.header = tk.Frame(self, bg="#151515")
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_columnconfigure(1, weight=1)

        self.badge = tk.Label(self.header, text="●", fg="#00ff7f", bg="#151515")
        self.badge.grid(row=0, column=0, padx=(8,4), pady=6)

        self.name_label = tk.Label(self.header, text=camera_name, font=("Segoe UI", 10, "bold"), bg="#151515", fg=self.COLOR_TEXT)
        self.name_label.grid(row=0, column=1, sticky="w")

        self.actions = tk.Frame(self.header, bg="#151515")
        self.actions.grid(row=0, column=2, sticky="e", padx=6)
        self.btn_fs = tk.Button(self.actions, text="⤢",
                                command=lambda: self.app.toggle_fullscreen_view(self),
                                bd=0, bg="#222", fg="#ddd", width=3)
        self.btn_fs.pack(side="left", padx=2)
        btn_snap = tk.Button(self.actions, text="📸",
                             command=lambda: self.app.snapshot_camera(self.camera_name),
                             bd=0, bg="#222", fg="#ddd", width=3)
        btn_snap.pack(side="left", padx=2)

        self.canvas = tk.Canvas(self, bg="#000", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.status_bar = tk.Label(self, text="IDLE", font=("Segoe UI", 9, "bold"), bg="#2a2a2a", fg="#ffffff")
        self.status_bar.grid(row=2, column=0, sticky="ew")

        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Start Stream", command=lambda: self.app.start_camera_by_name(self.camera_name))
        self.menu.add_command(label="Stop Stream", command=lambda: self.app.stop_camera_by_name(self.camera_name))
        self.menu.add_separator()
        self.menu.add_command(label="Toggle Fullscreen", command=lambda: self.app.toggle_fullscreen_view(self))
        self.menu.add_command(label="Snapshot", command=lambda: self.app.snapshot_camera(self.camera_name))
        self.menu.add_command(label="Start Recording", command=lambda: self.app.toggle_recording(self.camera_name, True))
        self.menu.add_command(label="Stop Recording", command=lambda: self.app.toggle_recording(self.camera_name, False))

        for w in (self, self.canvas, self.header, self.name_label):
            w.bind("<Button-3>", self._show_menu)
            w.bind("<Double-Button-1>", lambda e: self.app.toggle_fullscreen_view(self))

        self.last_frame = None
        self.alert_logged = False

        self._vp_w = None
        self._vp_h = None
        self.canvas.bind("<Configure>", lambda e: self._on_resize())
        
        # [REMOVED] The <Map> binding from the previous fix is no longer needed.
        
    # [REMOVED] The _schedule_initial_resize method is no longer needed.

    def set_fullscreen_icon(self, is_fullscreen: bool):
        try:
            self.btn_fs.config(text="🗗" if is_fullscreen else "⤢")
        except Exception:
            pass

    def _show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def _compute_16x9_viewport(self, avail_w, avail_h):
        if avail_w < 2 or avail_h < 2:
            return None, None
        target_w = min(avail_w, int(avail_h * 16 / 9))
        target_h = int(target_w * 9 / 16)
        if target_h > avail_h:
            target_h = min(avail_h, int(avail_w * 9 / 16))
            target_w = int(target_h * 16 / 9)
        return target_w, target_h

    # ===================================================================
    # [UPDATED METHOD] This function now contains the self-correcting logic
    # ===================================================================
    def _on_resize(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # [NEW CHECK] If the widget is still tiny (e.g., 1x1 or < 50px), 
        # the grid hasn't settled. Reschedule the resize and wait.
        if w < 50 or h < 50:
            try:
                # Check again in 50ms
                self.after(50, self._on_resize)
            except Exception:
                pass # Widget might be destroyed
            return # Stop here, don't try to calculate viewport
            
        # [OLD LOGIC] If we're here, w/h are non-trivial. Proceed.
        self._vp_w, self._vp_h = self._compute_16x9_viewport(w, h)
        self._redraw_last_frame()

    def _redraw_last_frame(self):
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()

        if self._vp_w is None or self._vp_h is None:
            self.canvas.create_text(cw // 2, ch // 2, text="Waiting for feed…", fill="#6c757d", font=("Segoe UI", 12))
            return

        cx, cy = cw // 2, ch // 2

        if self.last_frame is None:
            self.canvas.create_rectangle(cx - self._vp_w // 2, cy - self._vp_h // 2, cx + self._vp_w // 2, cy + self._vp_h // 2,
                                         outline="#222", width=1)
            self.canvas.create_text(cx, cy, text="Waiting for feed…", fill="#6c757d", font=("Segoe UI", 12))
            return

        # --- THIS IS THE SLOW PATH ---
        frame_resized = cv2.resize(self.last_frame, (self._vp_w, self._vp_h))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(cx, cy, image=imgtk, anchor="center", tags="video")

    def update_display(self, frame, imgtk, status, confidence):
        try:
            if frame is not None:
                self.last_frame = frame.copy()

            if imgtk:
                # --- FAST PATH ---
                self.canvas.delete("all")
                cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
                cx, cy = cw // 2, ch // 2
                self.canvas.imgtk = imgtk
                self.canvas.create_image(cx, cy, image=imgtk, anchor="center", tags="video")
            else:
                # --- SLOW/FALLBACK PATH ---
                if self._vp_w is None or self._vp_h is None:
                    self._on_resize()
                else:
                    self._redraw_last_frame()

            if status == "CRIME":
                self.status_bar.config(text=f"CRIME ({confidence:.1f}%)", bg="#ff3b5b")
                self.badge.config(fg="#ff3b5b")
            elif status == "NORMAL":
                self.status_bar.config(text=f"NORMAL ({confidence:.1f}%)", bg="#00a86b")
                self.badge.config(fg="#00ff7f")
            elif status and status.startswith("FPS:"):
                self.status_bar.config(text=f"{self.status_bar.cget('text')}   \u00A0| \u00A0{status}")
            elif status and status.startswith("INFO:"):
                self.app.toast.show(f"{self.camera_name}: {status[5:]}")
            elif status:
                self.status_bar.config(text=status, bg="#2a2a2a")
        except Exception:
            pass

    def show_error(self, message="FEED LOST"):
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                                text=message, fill="#ff3b5b", font=("Segoe UI", 12, "bold"))
        self.status_bar.config(text="ERROR", bg="#ff3b5b")
        self.badge.config(fg="#ff3b5b")

# ------------------------ Main App ------------------------ #
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1380x820")
        self.window.minsize(1120, 680)
        self.window.configure(bg="#141414")

        # Theme
        self.COLOR_BG = "#141414"
        self.COLOR_PANEL = "#1f1f1f"
        self.COLOR_TEXT = "#e0e0e0"
        self.COLOR_ACCENT = "#00e5ff"
        self.COLOR_ALERT = "#ff3b5b"
        self.COLOR_OK = "#00ff7f"
        self.FONT = ("Segoe UI", 10)
        self.FONT_B = ("Segoe UI", 10, "bold")
        self.FONT_TITLE = ("Segoe UI", 20, "bold")

        # Model
        self.model_path = resource_path("best_model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.num_frames_per_clip = 16
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

        # [MODIFIED] Load preferences first
        self.toast = Toast(self.window)
        self.prefs = self.read_prefs()
        
        # State
        # [MODIFIED] Load cameras from preferences
        self.cameras = self.prefs.get("cameras", {}) 
        self.data_queue = queue.Queue()
        self.active_streams = {}
        self.fullscreen_widget = None
        self.fs_exit_btn = None
        self._icon_photo = None
        
        # Ensure snapshot dir exists
        os.makedirs(self.prefs.get("snapshot_dir", resource_path("snapshots")), exist_ok=True)

        # Debug (booleans only)
        safe_print(f"[INFO] Prefs file: {PREFS_PATH}")
        safe_print(
            "[INFO] Twilio fields set ->",
            "to_num:", bool((self.prefs.get('alert_to_num') or '').strip()),
            "from_num:", bool((self.prefs.get('twilio_from_num') or '').strip()),
            "sid:", bool((self.prefs.get('twilio_sid') or '').strip()),
            "token:", bool((self.prefs.get('twilio_token') or '').strip()),
            "lib_installed:", Client is not None
        )

        # Styles & Shell
        self.setup_styles()
        self.set_app_icon()
        self.build_menubar()
        self.build_toolbar()
        self.build_main_panes()
        self.bind_shortcuts()

        # Require user identity; Twilio recipient mirrors profile number
        self.ensure_user_identity()

        # Loop
        self.process_queue_loop()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ---------- Icon setup ---------- #
    def set_app_icon(self):
        ico_candidates = [resource_path("icon.ico"), USER_ICON_ICO]
        png_candidates = [resource_path("icon.png"), USER_ICON_PNG]
        for p in ico_candidates:
            if p and os.path.exists(p):
                try: self.window.iconbitmap(p); return
                except Exception: pass
        for p in png_candidates:
            if p and os.path.exists(p):
                try:
                    self._icon_photo = tk.PhotoImage(file=p)
                    self.window.iconphoto(True, self._icon_photo)
                    return
                except Exception:
                    pass

    # ---------- Preferences ---------- #
    @staticmethod
    def read_prefs_silent():
        try:
            with open(PREFS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_PREFS.copy()

    def read_prefs(self):
        prefs = self.read_prefs_silent()
        for k, v in DEFAULT_PREFS.items():
            prefs.setdefault(k, v)
        # Backfill empties from defaults for managed fields
        def _blank(x): return x is None or (isinstance(x, str) and x.strip() == "")
        for k in ("twilio_sid", "twilio_token", "twilio_from_num"):
            if _blank(prefs.get(k)) and not _blank(DEFAULT_PREFS.get(k)):
                prefs[k] = DEFAULT_PREFS[k]
        return prefs

    def save_prefs(self):
        try:
            with open(PREFS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.prefs, f, indent=2)
        except Exception as e:
            self.toast.show(f"Failed to save preferences: {e}")

    # ---------- UI Construct ---------- #
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TFrame", background=self.COLOR_BG)
        style.configure("Side.TFrame", background=self.COLOR_PANEL)
        style.configure("TLabel", background=self.COLOR_BG, foreground=self.COLOR_TEXT, font=self.FONT)
        style.configure("Accent.TButton", font=self.FONT_B, padding=8, relief="flat")
        style.map("Accent.TButton", background=[("active", "#2b2b2b")])
        style.configure("Tool.TButton", padding=6, relief="flat")
        style.configure("Treeview", background=self.COLOR_PANEL, foreground=self.COLOR_TEXT,
                        fieldbackground=self.COLOR_PANEL, rowheight=28)
        style.configure("Treeview.Heading", background="#2a2a2a", foreground="#cfcfcf")
        style.configure("TLabelframe", background=self.COLOR_PANEL)
        style.configure("TLabelframe.Label", background=self.COLOR_PANEL,
                         foreground=self.COLOR_ACCENT, font=self.FONT_B)
        self.style = style

    def build_menubar(self):
        m = tk.Menu(self.window)
        m_file = tk.Menu(m, tearoff=0)
        m_file.add_command(label="Add Camera…", accelerator="Ctrl+N", command=self.prompt_add_camera)
        m_file.add_separator()
        m_file.add_command(label="Preferences…", accelerator="Ctrl+,", command=self.open_preferences)
        m_file.add_command(label="Profile…", command=self.open_identity_dialog)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.on_closing)
        m.add_cascade(label="File", menu=m_file)

        m_view = tk.Menu(m, tearoff=0)
        m_view.add_command(label="Toggle Fullscreen", accelerator="F11",
                           command=lambda: self.toggle_fullscreen_view(self.last_clicked_widget or None))
        m.add_cascade(label="View", menu=m_view)

        m_cam = tk.Menu(m, tearoff=0)
        m_cam.add_command(label="Start Selected", accelerator="Ctrl+R", command=self.start_analysis)
        m_cam.add_command(label="Stop Selected", accelerator="Ctrl+Shift+R", command=self.stop_analysis)
        m_cam.add_separator()
        m_cam.add_command(label="Start All", command=self.start_all)
        m_cam.add_command(label="Stop All", command=self.stop_all)
        m.add_cascade(label="Cameras", menu=m_cam)

        m_help = tk.Menu(m, tearoff=0)
        m_help.add_command(label="About", command=lambda: messagebox.showinfo("About", "SecureSight AI\nModern UI • Tkinter"))
        m.add_cascade(label="Help", menu=m_help)

        self.window.config(menu=m)

    def build_toolbar(self):
        tb = tk.Frame(self.window, bg="#1a1a1a")
        tb.pack(fill="x")
        def tbtn(text, cmd):
            btn = tk.Button(tb, text=text, command=cmd, bd=0, bg="#252525", fg="#eaeaea", padx=12, pady=6)
            btn.pack(side="left", padx=4, pady=6)
            return btn
        tbtn("＋ Add", self.prompt_add_camera)
        tbtn("▶ Start", self.start_analysis)
        tbtn("■ Stop", self.stop_analysis)
        tbtn("⤢ Fullscreen", lambda: self.toggle_fullscreen_view(self.last_clicked_widget or None))
        tbtn("📸 Snapshot", lambda: self.snapshot_camera(self.camera_select_dropdown.get() or None))
        tbtn("⚙ Preferences", self.open_preferences)

    def build_main_panes(self):
        paned = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(paned, style="Side.TFrame")
        side.grid_columnconfigure(0, weight=1)
        tk.Label(side, text="Cameras", font=self.FONT_B, bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT)\
            .grid(row=0, column=0, sticky="ew", padx=12, pady=(10,4))
        self.tree = ttk.Treeview(side, columns=("source", "status"), show="headings", selectmode="browse")
        self.tree.heading("source", text="Source")
        self.tree.heading("status", text="Status")
        self.tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        side.grid_rowconfigure(1, weight=1)
        paned.add(side, weight=1)

        # [MODIFIED] This now populates the tree from prefs-loaded self.cameras
        for name, src in self.cameras.items():
            self.tree.insert("", "end", iid=name, values=(src, "Idle"))

        center = ttk.Notebook(paned)
        self.video_tab = ttk.Frame(center)
        center.add(self.video_tab, text="Live Feeds")
        self.analytics_tab = ttk.Frame(center)
        center.add(self.analytics_tab, text="Analytics & Logs")
        paned.add(center, weight=4)

        self.video_grid_frame = tk.Frame(self.video_tab, bg="#1f1f1f")
        self.video_grid_frame.pack(fill="both", expand=True, padx=12, pady=12)
        for i in range(6):
            self.video_grid_frame.grid_columnconfigure(i, weight=1)
        for i in range(6):
            self.video_grid_frame.grid_rowconfigure(i, weight=1)

        self.placeholder_label = tk.Label(
            self.video_grid_frame,
            text="Add cameras and start analysis\nYour feeds will appear here.",
            font=self.FONT_TITLE, bg="#1f1f1f", fg="#6c757d"
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")

        box = ttk.Labelframe(self.analytics_tab, text="EVENT LOG")
        box.pack(fill="both", expand=True, padx=12, pady=12)
        self.log_text = tk.Text(box, height=12, state="disabled", bg="#101010", fg="#e0e0e0",
                                font=("Consolas", 10), relief="flat")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        ctl = ttk.Labelframe(side, text="ANALYZE")
        ctl.grid(row=2, column=0, sticky="ew", padx=10, pady=(0,12))
        tk.Label(ctl, text="Select Camera:", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=0, column=0, sticky="w", padx=10, pady=(8,2))
        
        # [MODIFIED] This now populates the dropdown from prefs-loaded self.cameras
        self.camera_select_dropdown = ttk.Combobox(ctl, values=list(self.cameras.keys()), state="readonly")
        if self.cameras: self.camera_select_dropdown.current(0)
        
        self.camera_select_dropdown.grid(row=1, column=0, sticky="ew", padx=10)
        btns = tk.Frame(ctl, bg=self.COLOR_PANEL)
        btns.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        tk.Button(btns, text="▶ Start", command=self.start_analysis, bd=0, bg="#00bcd4", fg="#0b0b0b")\
            .pack(side="left", expand=True, fill="x", padx=(0,6))
        tk.Button(btns, text="■ Stop", command=self.stop_analysis, bd=0, bg="#ff3b5b", fg="#0b0b0b")\
            .pack(side="left", expand=True, fill="x", padx=(6,0))
        tk.Button(ctl, text="Remove from List", command=self.remove_camera, bd=0, bg="#2b2b2b", fg="#ddd")\
            .grid(row=3, column=0, sticky="ew", padx=10, pady=(0,10))

        self.status = tk.StringVar(value=f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'} | Active: 0")
        sb = tk.Label(self.window, textvariable=self.status, anchor="w", bg="#1a1a1a", fg="#cfcfcf", padx=10)
        sb.pack(fill="x", side="bottom")

        self.last_clicked_widget = None

    # ---------- Shortcuts ---------- #
    def bind_shortcuts(self):
        self.window.bind("<Control-n>", lambda e: self.prompt_add_camera())
        self.window.bind("<Control-N>", lambda e: self.prompt_add_camera())
        self.window.bind("<Control-r>", lambda e: self.start_analysis())
        self.window.bind("<Control-R>", lambda e: self.start_analysis())
        self.window.bind("<Control-Shift-R>", lambda e: self.stop_analysis())
        self.window.bind("<F11>", lambda e: self.toggle_fullscreen_view(self.last_clicked_widget or None))
        self.window.bind("<Control-comma>", lambda e: self.open_preferences())
        self.window.bind("<Escape>", lambda e: self._exit_fullscreen())

    # ---------- Core actions ---------- #
    def build_tile(self, name):
        return VideoDisplayWidget(self.video_grid_frame, self, name, bg="#000000")

    def start_camera_by_name(self, name):
        if name in self.active_streams:
            self.toast.show(f"'{name}' already running")
            return
        source = self.cameras.get(name)
        if source is None:
            messagebox.showwarning("Unknown", f"No camera named '{name}'")
            return

        new_widget = self.build_tile(name)

        camera_thread = CameraThread(
            camera_name=name,
            camera_source=source,
            model=self.model,
            device=self.device,
            transform=self.transform,
            data_queue=self.data_queue,
            widget=new_widget,
            clip_len=self.num_frames_per_clip,
        )
        camera_thread.start()
        self.active_streams[name] = (camera_thread, new_widget)
        self.redraw_video_grid()
        self.update_tree_status(name, "Running")
        self.log_event(f"Analysis started for: '{name}'")
        self.update_statusbar()

    def stop_camera_by_name(self, name):
        if name not in self.active_streams:
            self.toast.show(f"'{name}' not running")
            return
        thread, widget = self.active_streams[name]
        thread.stop()
        widget.destroy()
        del self.active_streams[name]
        if self.fullscreen_widget == widget:
            self.fullscreen_widget = None
            self._hide_fs_exit_button()
            for (_t, w) in self.active_streams.values():
                w.set_fullscreen_icon(False)
        self.redraw_video_grid()
        self.update_tree_status(name, "Stopped")
        self.log_event(f"Analysis stopped for: '{name}'")
        self.update_statusbar()

    def start_analysis(self):
        selected_name = self.camera_select_dropdown.get()
        if not selected_name:
            messagebox.showwarning("Input Required", "Please select a camera to analyze.")
            return
        self.start_camera_by_name(selected_name)

    def stop_analysis(self):
        selected_name = self.camera_select_dropdown.get()
        if not selected_name:
            messagebox.showwarning("Input Required", "Please select a camera to stop.")
            return
        self.stop_camera_by_name(selected_name)

    def start_all(self):
        for name in list(self.cameras.keys()):
            if name not in self.active_streams:
                self.start_camera_by_name(name)

    def stop_all(self):
        for name in list(self.active_streams.keys()):
            self.stop_camera_by_name(name)

    def toggle_recording(self, name, enable):
        if name not in self.active_streams:
            self.toast.show("Start the stream first")
            return
        thread, _ = self.active_streams[name]
        thread.toggle_recording(enable)
        self.toast.show(f"{'Recording' if enable else 'Stopped recording'}: {name}")

    def snapshot_camera(self, name):
        if not name:
            self.toast.show("Select a camera first")
            return
        stream = self.active_streams.get(name)
        if not stream:
            self.toast.show("Stream is not running")
            return
        _, widget = stream
        frame = widget.last_frame
        if frame is None:
            self.toast.show("No frame yet")
            return
        out_dir = self.prefs.get("snapshot_dir", resource_path("snapshots"))
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(out_dir, fname)
        try:
            cv2.imwrite(path, frame)
            self.toast.show(f"Saved snapshot: {fname}")
            self.log_event(f"Snapshot saved: {path}")
        except Exception as e:
            self.toast.show(f"Failed to save snapshot: {e}")

    # ---------- Fullscreen helpers ---------- #
    def _show_fs_exit_button(self):
        if self.fs_exit_btn is not None:
            return
        self.fs_exit_btn = tk.Button(
            self.window, text="🗗  Exit Fullscreen",
            command=self._exit_fullscreen,
            bd=0, bg="#2b2b2b", fg="#eaeaea", padx=12, pady=6,
            activebackground="#3a3a3a", activeforeground="#ffffff"
        )
        self.fs_exit_btn.place(relx=1.0, rely=0.0, x=-16, y=52, anchor="ne")

    def _hide_fs_exit_button(self):
        if self.fs_exit_btn is not None:
            try:
                self.fs_exit_btn.destroy()
            except Exception:
                pass
            self.fs_exit_btn = None

    def _exit_fullscreen(self):
        self.fullscreen_widget = None
        self.redraw_video_grid()
        self._hide_fs_exit_button()
        for (_t, w) in self.active_streams.values():
            w.set_fullscreen_icon(False)

    # ---------- Grid & fullscreen ---------- #
    def toggle_fullscreen_view(self, widget):
        if widget is None:
            return
        self.last_clicked_widget = widget
        if self.fullscreen_widget == widget:
            self.fullscreen_widget = None
            self._hide_fs_exit_button()
            for (_t, w) in self.active_streams.values():
                w.set_fullscreen_icon(False)
        else:
            self.fullscreen_widget = widget
            self._show_fs_exit_button()
            for (_t, w) in self.active_streams.values():
                w.set_fullscreen_icon(w is self.fullscreen_widget)
        self.redraw_video_grid()

    def redraw_video_grid(self):
        if not self.active_streams:
            self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.placeholder_label.place_forget()
        widgets = [w for (_t, w) in self.active_streams.values()]
        if self.fullscreen_widget and self.fullscreen_widget in widgets:
            for w in widgets:
                if w is not self.fullscreen_widget:
                    w.grid_forget()
            self.fullscreen_widget.grid(row=0, column=0, rowspan=20, columnspan=20, sticky="nsew", padx=6, pady=6)
        else:
            max_cols = int(self.prefs.get("grid_max_cols", 3))
            for i, w in enumerate(widgets):
                r, c = divmod(i, max_cols)
                w.grid_forget()
                w.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
        if self.fullscreen_widget is None or self.fullscreen_widget not in widgets:
            self._hide_fs_exit_button()

    # ---------- Queue loop & logs ---------- #
    def process_queue_loop(self):
        try:
            while not self.data_queue.empty():
                camera_name, frame, imgtk, status, confidence = self.data_queue.get_nowait()

                if camera_name in self.active_streams:
                    thread, widget = self.active_streams[camera_name]

                    if frame is None and imgtk is None:
                        if status.startswith("ERROR"):
                            widget.show_error(status)
                            self.log_event(f"Error on '{camera_name}': {status}", level="alert")
                            self.update_tree_status(camera_name, "Error")
                        elif status.startswith("INFO:") or status.startswith("FPS:"):
                            widget.update_display(widget.last_frame if widget.last_frame is not None else frame, imgtk, status, confidence)
                    else:
                        widget.update_display(frame, imgtk, status, confidence)

                        if status == "CRIME" and self.prefs.get("auto_snapshot_on_crime", True):
                            try:
                                if widget.last_frame is not None and not widget.alert_logged:
                                    self.snapshot_camera(camera_name)
                                    self.log_event(f"CRIME detected on {camera_name}. Triggering snapshot and alert.", level="alert")
                                    threading.Thread(target=self.send_crime_alert_threaded,
                                                     args=(camera_name, confidence),
                                                     daemon=True).start()
                                    widget.alert_logged = True
                            except Exception:
                                pass
                        elif status == "NORMAL":
                            widget.alert_logged = False
            self.update_statusbar()
        except Exception as e:
            safe_print(f"Queue loop error: {e}")
        finally:
            self.window.after(30, self.process_queue_loop)

    def log_event(self, message, level="info"):
        self.log_text.config(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def update_statusbar(self):
        active = len(self.active_streams)
        mode = 'CUDA' if torch.cuda.is_available() else 'CPU'
        self.status.set(f"Device: {mode} | Active: {active} | {datetime.now().strftime('%H:%M:%S')}")

    def update_tree_status(self, name, status):
        if self.tree.exists(name):
            vals = list(self.tree.item(name, 'values'))
            if len(vals) < 2:
                vals = [self.cameras.get(name, ''), status]
            else:
                vals[1] = status
            self.tree.item(name, values=tuple(vals))

    # ---------- Camera list mgmt ---------- #
    def prompt_add_camera(self):
        self.open_add_camera_dialog()

    def open_add_camera_dialog(self):
        win = tk.Toplevel(self.window)
        win.title("Add Camera")
        win.transient(self.window)
        win.configure(bg=self.COLOR_PANEL)
        win.resizable(False, False)
        win.grab_set()

        tk.Label(win, text="Name:", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=0, column=0, sticky="w", padx=12, pady=(12,4))
        name_var = tk.StringVar()
        name_entry = ttk.Entry(win, textvariable=name_var, width=36)
        name_entry.grid(row=1, column=0, sticky="ew", padx=12)

        tk.Label(win, text="Source (index or URL):", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=2, column=0, sticky="w", padx=12, pady=(12,4))
        src_var = tk.StringVar()
        src_entry = ttk.Entry(win, textvariable=src_var, width=36)
        src_entry.grid(row=3, column=0, sticky="ew", padx=12)

        btn_row = tk.Frame(win, bg=self.COLOR_PANEL)
        btn_row.grid(row=4, column=0, sticky="e", padx=12, pady=12)
        def do_add():
            name = name_var.get().strip()
            source = src_var.get().strip()
            if not name or not source:
                messagebox.showwarning("Input Required", "Please enter both a name and a source.")
                return
            self.add_camera_from_values(name, source)
            win.destroy()
        tk.Button(btn_row, text="Cancel", command=win.destroy, bd=0, bg="#2b2b2b", fg="#ddd")\
            .pack(side="right", padx=(6,0))
        tk.Button(btn_row, text="Add", command=do_add, bd=0, bg="#00a86b", fg="#0b0b0b")\
            .pack(side="right")

        name_entry.focus_set()
        win.bind("<Return>", lambda e: do_add())
        win.bind("<Escape>", lambda e: win.destroy())

    def add_camera(self):
        self.open_add_camera_dialog()

    def add_camera_from_values(self, name, source):
        if not name or not source:
            messagebox.showwarning("Input Required", "Please enter both a name and a source.")
            return
        if name in self.cameras:
            messagebox.showwarning("Name Exists", f"A camera named '{name}' already exists.")
            return
        
        self.cameras[name] = source
        # [MODIFIED] Save cameras to preferences
        self.prefs["cameras"] = self.cameras
        self.save_prefs()
        
        self.camera_select_dropdown['values'] = list(self.cameras.keys())
        self.camera_select_dropdown.set(name)
        self.tree.insert("", "end", iid=name, values=(source, "Idle"))
        self.log_event(f"Camera added: '{name}'")

    def remove_camera(self):
        selected_name = self.camera_select_dropdown.get()
        if not selected_name:
            messagebox.showwarning("No Selection", "Please select a camera to remove.")
            return
        if selected_name in self.active_streams:
            messagebox.showerror("Error", f"Cannot remove '{selected_name}'. Stop the stream first.")
            return
        if messagebox.askyesno("Confirm", f"Remove camera '{selected_name}' from the list?"):
            del self.cameras[selected_name]
            
            # [MODIFIED] Save updated camera list to preferences
            self.prefs["cameras"] = self.cameras
            self.save_prefs()
            
            if self.tree.exists(selected_name):
                self.tree.delete(selected_name)
            self.camera_select_dropdown['values'] = list(self.cameras.keys())
            if self.cameras:
                self.camera_select_dropdown.current(0)
            else:
                self.camera_select_dropdown.set("")
            self.log_event(f"Camera removed: '{selected_name}'")

    # ---------- Preferences dialog (Twilio removed; it's managed) ---------- #
    def open_preferences(self):
        win = tk.Toplevel(self.window)
        win.title("Preferences")
        win.transient(self.window)
        win.configure(bg=self.COLOR_PANEL)
        win.resizable(False, False)
        win.grab_set()

        f_general = ttk.Labelframe(win, text="General Settings")
        f_general.pack(fill="x", expand=True, padx=12, pady=(12, 6))
        
        tk.Label(f_general, text="Snapshots folder:", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=0, column=0, sticky="w", padx=12, pady=(12,4))
        path_var = tk.StringVar(value=self.prefs.get("snapshot_dir", resource_path("snapshots")))
        ent = ttk.Entry(f_general, textvariable=path_var, width=46)
        ent.grid(row=1, column=0, sticky="w", padx=12)
        def browse():
            d = filedialog.askdirectory(initialdir=path_var.get() or os.getcwd())
            if d: path_var.set(d)
        tk.Button(f_general, text="Browse…", command=browse, bd=0, bg="#2b2b2b", fg="#ddd")\
            .grid(row=1, column=1, padx=(6,12))

        auto_snap = tk.BooleanVar(value=self.prefs.get("auto_snapshot_on_crime", True))
        tk.Checkbutton(f_general, text="Auto-snapshot on CRIME", variable=auto_snap,
                         bg=self.COLOR_PANEL, fg=self.COLOR_TEXT, selectcolor="#2b2b2b")\
            .grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(10,0))

        tk.Label(f_general, text="Grid max columns:", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=3, column=0, sticky="w", padx=12, pady=(12,4))
        cols_var = tk.IntVar(value=int(self.prefs.get("grid_max_cols", 3)))
        spn = tk.Spinbox(f_general, from_=1, to=6, textvariable=cols_var, width=5)
        spn.grid(row=4, column=0, sticky="w", padx=12)

        tk.Label(f_general, text="Record codec (mp4v/avc1)", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=5, column=0, sticky="w", padx=12, pady=(12,4))
        codec_var = tk.StringVar(value=self.prefs.get("record_codec", "mp4v"))
        ttk.Combobox(f_general, values=["mp4v", "avc1"], state="readonly",
                     textvariable=codec_var, width=8).grid(row=6, column=0, sticky="w", padx=12, pady=(0, 12))

        # Managed Twilio summary (read-only)
        f_info = ttk.Labelframe(win, text="WhatsApp Alerts (Managed)")
        f_info.pack(fill="x", expand=True, padx=12, pady=6)
        managed = [
            ("Account SID", "twilio_sid"),
            ("Auth Token", "twilio_token"),
            ("From (sender)", "twilio_from_num"),
            ("Recipient (profile)", "alert_to_num"),
        ]
        r = 0
        for label, key in managed:
            val = ("****" if key in ("twilio_sid","twilio_token") and self.prefs.get(key) else self.prefs.get(key, ""))
            if key == "alert_to_num":
                val = self._to_whatsapp_address(self.prefs.get("user_mobile","")) or "(not set)"
            tk.Label(f_info, text=f"{label}:", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT).grid(row=r, column=0, sticky="w", padx=12, pady=4)
            tk.Label(f_info, text=str(val), bg=self.COLOR_PANEL, fg="#9ad")\
                .grid(row=r, column=1, sticky="w", padx=12, pady=4)
            r += 1

        def save_and_close():
            self.prefs["snapshot_dir"] = path_var.get()
            self.prefs["auto_snapshot_on_crime"] = bool(auto_snap.get())
            self.prefs["grid_max_cols"] = int(cols_var.get())
            self.prefs["record_codec"] = codec_var.get()
            # Keep Twilio managed; recipient mirrors profile number on save
            self._sync_twilio_recipient_from_profile(save=True)
            self.redraw_video_grid()
            win.destroy()
            self.toast.show("Preferences updated")
        
        tk.Button(win, text="Save & Close", command=save_and_close, bd=0, bg="#00a86b", fg="#0b0b0b", padx=10, pady=5)\
            .pack(side="right", padx=12, pady=(6, 12))
        tk.Button(win, text="Cancel", command=win.destroy, bd=0, bg="#2b2b2b", fg="#ddd", padx=10, pady=5)\
            .pack(side="right", padx=6, pady=(6, 12))

    # ---------- First-run Identity (Profile) ---------- #
    def ensure_user_identity(self):
        name = (self.prefs.get("user_name") or "").strip()
        mobile = (self.prefs.get("user_mobile") or "").strip()
        if not name or not mobile:
            self.open_identity_dialog(force=True)
        else:
            # Always mirror recipient to profile number & name
            self._sync_twilio_recipient_from_profile(save=True)

    def _sync_twilio_recipient_from_profile(self, save=False):
        name = (self.prefs.get("user_name") or "").strip()
        mobile = (self.prefs.get("user_mobile") or "").strip()
        wa = self._to_whatsapp_address(mobile) if mobile else ""
        self.prefs["alert_to_name"] = name
        if wa:
            self.prefs["alert_to_num"] = wa
        # backfill managed creds from defaults if empty
        for k, v in (("twilio_sid", TWILIO_SID_DEFAULT),
                     ("twilio_token", TWILIO_TOKEN_DEFAULT),
                     ("twilio_from_num", TWILIO_FROM_DEFAULT)):
            if not (self.prefs.get(k) or "").strip():
                self.prefs[k] = v
        if save:
            self.save_prefs()

    def _to_whatsapp_address(self, raw_number: str) -> str:
        s = (raw_number or "").strip()
        if s.startswith("whatsapp:"):  # allow already formatted
            return s
        if re.fullmatch(r"\+?\d{7,15}", s):
            if not s.startswith("+"):
                s = "+" + s
            return "whatsapp:" + s
        return ""

    def open_identity_dialog(self, force: bool=False):
        win = tk.Toplevel(self.window)
        win.title("Your Profile")
        win.transient(self.window)
        win.configure(bg=self.COLOR_PANEL)
        win.resizable(False, False)
        if force:
            win.grab_set()

        name_var = tk.StringVar(value=self.prefs.get("user_name", ""))
        mobile_var = tk.StringVar(value=self.prefs.get("user_mobile", ""))

        pad = {"padx":12, "pady":4}
        tk.Label(win, text="Enter your name and WhatsApp number (E.164). Alerts will go to this number.",
                 bg=self.COLOR_PANEL, fg=self.COLOR_TEXT).grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        tk.Label(win, text="Your Name:", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(win, textvariable=name_var, width=36).grid(row=1, column=1, sticky="ew", **pad)

        tk.Label(win, text="Mobile (e.g., +91XXXXXXXXXX):", bg=self.COLOR_PANEL, fg=self.COLOR_TEXT)\
            .grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(win, textvariable=mobile_var, width=36).grid(row=2, column=1, sticky="ew", **pad)

        btn_row = tk.Frame(win, bg=self.COLOR_PANEL)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="e", padx=12, pady=12)

        def valid():
            name = (name_var.get() or "").strip()
            mobile = (mobile_var.get() or "").strip()
            if not name:
                messagebox.showwarning("Required", "Please enter your name.", parent=win); return False
            if not re.fullmatch(r"\+?\d{7,11}", mobile):
                messagebox.showwarning("Invalid number", "Use E.164 format, e.g., +91XXXXXXXXXX.", parent=win); return False
            return True

        def save_and_close():
            if not valid():
                if force: return
                else: return
            self.prefs["user_name"] = (name_var.get() or "").strip()
            self.prefs["user_mobile"] = (mobile_var.get() or "").strip()
            self._sync_twilio_recipient_from_profile(save=True)
            try: self.toast.show("Profile saved")
            except Exception: pass
            win.destroy()

        tk.Button(btn_row, text="Save", command=save_and_close, bd=0, bg="#00a86b", fg="#0b0b0b", padx=10, pady=6)\
            .pack(side="right", padx=(6,0))

        def cancel():
            if force:
                if messagebox.askyesno("Skip?", "Skip entering profile? Alerts will be disabled until you set it.", parent=win):
                    win.destroy()
            else:
                win.destroy()

        tk.Button(btn_row, text="Cancel", command=cancel, bd=0, bg="#2b2b2b", fg="#ddd", padx=10, pady=6)\
            .pack(side="right")

        win.bind("<Return>", lambda e: save_and_close())
        win.bind("<Escape>", lambda e: cancel())

    # ---------- Twilio readiness + Alert Function ---------- #
    def _twilio_ready(self):
        if Client is None:
            return False, "Twilio library not installed. Run: pip install twilio"
        sid = (self.prefs.get("twilio_sid") or "").strip()
        token = (self.prefs.get("twilio_token") or "").strip()
        from_num = (self.prefs.get("twilio_from_num") or "").strip()
        to_num = self._to_whatsapp_address(self.prefs.get("user_mobile",""))
        if to_num and self.prefs.get("alert_to_num") != to_num:
            self.prefs["alert_to_num"] = to_num
        problems = []
        if not sid: problems.append("Account SID is empty.")
        if not token: problems.append("Auth Token is empty.")
        if not from_num or not from_num.startswith("whatsapp:+"):
            problems.append("From number must look like whatsapp:+14155238886.")
        if not to_num:
            problems.append("Profile mobile is empty/invalid; set it in File → Profile.")
        ok = len(problems) == 0
        return ok, "\n".join(problems)

    def send_crime_alert_threaded(self, camera_name, confidence):
        # Always mirror recipient from profile
        self._sync_twilio_recipient_from_profile(save=True)

        ok, why = self._twilio_ready()
        if not ok:
            safe_print(f"[WARN] Cannot send WhatsApp alert: {why}")
            try:
                self.data_queue.put((camera_name, None, None, f"INFO: WhatsApp alert not sent: {why}", 0.0))
            except Exception:
                pass
            return

        message_text = (
            "🚨 SecureSight AI Alert!\n\n"
            "Suspicious activity (CRIME) detected.\n\n"
            f"Camera: {camera_name}\n"
            f"Confidence: {confidence:.2f}%\n"
            f"Recipient: {self.prefs.get('user_name') or 'User'}"
        )

        try:
            client = Client(self.prefs["twilio_sid"], self.prefs["twilio_token"])
            msg = client.messages.create(
                from_=self.prefs["twilio_from_num"],
                body=message_text,
                to=self.prefs["alert_to_num"]
            )
            safe_print(f"[OK] Alert sent via WhatsApp (SID: {msg.sid}) for {camera_name}")
            try:
                self.data_queue.put((camera_name, None, None, "INFO: WhatsApp alert sent.", 0.0))
            except Exception:
                pass
        except Exception as e:
            safe_print(f"[ERROR] Failed to send alert for {camera_name}: {e}")
            try:
                self.data_queue.put((camera_name, None, None, f"INFO: WhatsApp alert failed: {e}", 0.0))
            except Exception:
                pass

    # ---------- Model ---------- #
    def load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model file not found at: {self.model_path}")
            self.window.destroy()
            sys.exit(1)
        try:
            model = CNNLSTM(num_classes=2).to(self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            safe_print(f"Model loaded successfully on {self.device}.")
            return model
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading the model: {e}")
            self.window.destroy()
            sys.exit(1)

    # ---------- Shutdown ---------- #
    def on_closing(self):
        if self.active_streams:
            if messagebox.askokcancel("Quit", "Streams are running. Stop all and exit?"):
                for name, (thread, widget) in list(self.active_streams.items()):
                    thread.stop()
                self.window.after(400, self.window.destroy)
            else:
                return
        else:
            self.window.destroy()

# ------------------------ Entrypoint ------------------------ #
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "SecureSight AI – Modern")
    root.mainloop()