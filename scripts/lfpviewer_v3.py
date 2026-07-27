import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import argparse
from pathlib import Path


# ---------------------Input parser----------------------
# use example: python script.py /media/nikolas/EXELU_SSD1/2025-09-20_16-51-22
parser = argparse.ArgumentParser(description="Process a folder")
parser.add_argument("folder", type=Path, help="Path to the folder")

args = parser.parse_args()

folder_path = args.folder

if not folder_path.is_dir():
    raise ValueError(f"{folder_path} is not a valid directory")

print(f"Folder provided: {folder_path}")

datfile = folder_path / f"{folder_path.name}.lfp"
print(f"processing recording: {datfile}")
particulars_path = folder_path / "session_particulars.txt"

# ----------------- your settings -----------------
FS = 1000
NCH = 139
DTYPE = np.int16

chan_idx = np.arange(0, 127)     # channels to display
window_s = 1.0                   # visible window
step_s = 0.25                    # scroll step
decim = 10                       # downsample for display speed
spacing_factor = 6.0             # vertical spacing multiplier


# -------------------------------------------------


def load_bad_channels(txt_path, one_indexed=True):
    txt_path = Path(txt_path)
    bad = set()
    for line in txt_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("bad_channels="):
            nums = line.split("=", 1)[1].strip().split()
            bad = {int(n) for n in nums}
            break
    if one_indexed:
        bad = {c - 1 for c in bad}
    return bad


BAD_CHANNELS = load_bad_channels(particulars_path, one_indexed=True)
is_bad = np.array([int(c) in BAD_CHANNELS for c in chan_idx], dtype=bool)
good_mask = ~is_bad

def update_y_axis_labels(step=2, start=0):
    """
    Label every `step`-th electrode starting at `start`.
    step=2 -> every other electrode
    start=0 -> even channels (0,2,4,...)
    start=1 -> odd channels (1,3,5,...)
    """
    y_axis = plot.getAxis('left')

    ticks = [
        (float(offsets[i]), str(int(chan_idx[i])))
        for i in range(start, len(chan_idx), step)
    ]

    y_axis.setTicks([ticks])

    # ensure enough space for 3-digit labels
    y_axis.setWidth(55)
    y_axis.setStyle(autoExpandTextSpace=False)

def update_home_ranges():
    global home_xrange, home_yrange
    home_xrange = (0.0, float(window_s))
    home_yrange = (float(offsets[-1] - spacing), float(offsets[0] + spacing))


# ----------------- RAM buffer -----------------
class RamWindowBuffer:
    """
    Buffers a multi-second chunk in RAM (decimated), slices windows from it quickly.
    Refill happens only when you scroll outside buffered region.
    """
    def __init__(self, memmap, fs, n_frames, channels, decim, buffer_s=20.0, out_dtype=np.float32):
        self.mm = memmap
        self.fs = float(fs)
        self.n_frames = int(n_frames)
        self.channels = np.array(channels, dtype=int)
        self.decim = int(decim)
        self.buffer_s = float(buffer_s)
        self.out_dtype = out_dtype

        self.buf = None                 # (buf_samples_decim, n_channels)
        self.buf_start = 0              # frame index in original sampling
        self.buf_end = 0                # frame index
        self._last_center = None

    def _refill(self, center_frame):
        half = int(0.5 * self.buffer_s * self.fs)
        start = max(0, int(center_frame) - half)
        end = min(self.n_frames, start + int(self.buffer_s * self.fs))
        start = max(0, end - int(self.buffer_s * self.fs))

        # Read (frames x channels) from memmap, then select channels, then decimate
        block = self.mm[start:end, self.channels]      # still a view-ish into memmap
        block = block[::self.decim, :]                 # decimate
        self.buf = block.astype(self.out_dtype, copy=False)

        self.buf_start = start
        self.buf_end = end
        self._last_center = center_frame

    def get(self, start_s, window_s):
        start_frame = int(start_s * self.fs)
        end_frame = min(self.n_frames, start_frame + int(window_s * self.fs))
        center = (start_frame + end_frame) // 2

        if (self.buf is None or start_frame < self.buf_start or end_frame > self.buf_end):
            self._refill(center)

        rel0 = start_frame - self.buf_start
        rel1 = end_frame - self.buf_start

        i0 = max(0, rel0 // self.decim)
        i1 = min(self.buf.shape[0], (rel1 + self.decim - 1) // self.decim)

        X = self.buf[i0:i1, :]
        # relative time for stable axis (fast!)
        t = (np.arange(X.shape[0], dtype=np.float32) * self.decim) / self.fs
        return t, X


# ----------------- data mapping -----------------
# Memmap shaped as (n_frames, n_channels)
file_size = Path(datfile).stat().st_size
bytes_per_sample = np.dtype(DTYPE).itemsize
n_frames = file_size // (NCH * bytes_per_sample)

mm = np.memmap(datfile, dtype=DTYPE, mode="r", shape=(n_frames, NCH))

buf = RamWindowBuffer(
    memmap=mm,
    fs=FS,
    n_frames=n_frames,
    channels=chan_idx,
    decim=decim,
    buffer_s=20.0,          # try 10..60
    out_dtype=np.float32
)

duration_s = n_frames / FS


# ----------------- GUI -----------------
pg.setConfigOptions(antialias=False, useOpenGL=False)  # huge win if OpenGL works on your machine

app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(title="Fast DAT Viewer (pyqtgraph)") #show=True, 
win.resize(1600, 900)
win.show()
# win.showMaximized()
QtCore.QTimer.singleShot(0, win.showMaximized)


plot = win.addPlot(row=0, col=0)
plot.showGrid(x=False, y=False)
plot.setMenuEnabled(False)
plot.setMouseEnabled(x=False, y=False)
plot.hideButtons()
plot.setClipToView(True)
plot.enableAutoRange(x=False, y=False)
# Fix x axis to 0..window_s so ticks never relayout
plot.setXRange(0, window_s, padding=0)
base_pen = pg.mkPen(color=(120, 180, 255), width=1)
sel_pen  = pg.mkPen(color=(255, 220, 80), width=3)


# Build one curve per good channel
curves = []
offsets = None
spacing = None

start_s = 0.0   

proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=30, slot=lambda evt: on_mouse_moved(evt))
hover_label = pg.LabelItem(justify="left")
win.addItem(hover_label, row=2, col=0)

selected_label = pg.LabelItem(justify="left")
win.addItem(selected_label, row=3, col=0)   # pick a row that exists for you
selected_label.setText("<b>Selected channel: (none)</b>")

def on_mouse_moved(evt):
    global start_s
    try:
        pos = evt[0]
        if plot.sceneBoundingRect().contains(pos):
            mouse_point = plot.vb.mapSceneToView(pos)
            x = float(mouse_point.x())
            if 0.0 <= x <= window_s:
                hover_label.setText(f"t = {start_s + x:.4f} s")
    except Exception as e:
        print("on_mouse_moved error:", e)

def select_channel(i): # for highlighting a trace
    """Highlight channel index i (index into chan_idx/curves), update label."""
    global selected_idx

    if i is None or i < 0 or i >= len(chan_idx) or curves[i] is None:
        return

    # unselect previous
    if selected_idx is not None and curves[selected_idx] is not None:
        curves[selected_idx].setPen(base_pen)
        curves[selected_idx].setZValue(0)

    # select new
    selected_idx = i
    curves[i].setPen(sel_pen)
    curves[i].setZValue(10)

    ch = int(chan_idx[i])
    selected_label.setText(f"<b>Selected channel: {ch}</b>")


def set_home_view():
    vb.setRange(xRange=home_xrange, yRange=home_yrange, padding=0, disableAutoRange=True)


def enter_zoom_mode():
    global zoom_mode
    zoom_mode = True
    plot.setMouseEnabled(x=True, y=True)          # allow mouse interaction
    vb.setMouseMode(pg.ViewBox.RectMode)          # drag-rectangle zoom
    vb.setCursor(QtCore.Qt.CrossCursor)

def exit_zoom_mode():
    global zoom_mode
    zoom_mode = False
    plot.setMouseEnabled(x=False, y=False)        # back to “viewer mode”
    vb.setMouseMode(pg.ViewBox.PanMode)           # default (won’t matter much if mouse disabled)
    vb.setCursor(QtCore.Qt.ArrowCursor)

def reset_view_and_exit_zoom():
    set_home_view()
    exit_zoom_mode()


# Precompute spacing from a quick sample (fast)
t0, X0 = buf.get(0.0, window_s)
mad = np.median(np.abs(X0 - np.median(X0, axis=0, keepdims=True)), axis=0)
typical = float(np.median(mad[mad > 0])) if np.any(mad > 0) else 1.0
spacing = spacing_factor * typical
offsets = np.arange(len(chan_idx))[::-1].astype(np.float32) * spacing

plot.setYRange(offsets[-1] - spacing, offsets[0] + spacing, padding=0)
home_yrange = (float(offsets[-1] - spacing), float(offsets[0] + spacing))
vb = plot.getViewBox()
selected_idx = None
# ----------------for highlighting clicked channel trace---------------
def on_mouse_clicked(ev):
    # If you're in zoom mode, let the ViewBox handle interactions
    if zoom_mode:
        return

    pos = ev.scenePos()
    if not plot.sceneBoundingRect().contains(pos):
        return

    p = vb.mapSceneToView(pos)
    y = float(p.y())

    # Find nearest row by offset (works great for stacked traces)
    i = int(np.argmin(np.abs(offsets - y)))

    # Only allow selecting good channels
    if not good_mask[i]:
        return

    select_channel(i)

plot.scene().sigMouseClicked.connect(on_mouse_clicked)

# ---------------------------------------------------------------

home_xrange = (0.0, float(window_s))
home_yrange = (float(offsets[-1] - spacing), float(offsets[0] + spacing))

zoom_mode = False

update_y_axis_labels()

# # Create curves
# for i in range(len(chan_idx)):
#     if not good_mask[i]:
#         curves.append(None)
#         continue
#     light_blue = pg.mkPen(color=(120, 180, 255), width=1)  # RGB
#     c = plot.plot(pen=light_blue)
#     curves.append(c)

# Create curves
for i in range(len(chan_idx)):
    if not good_mask[i]:
        curves.append(None)
        continue
    c = plot.plot(pen=base_pen)
    curves.append(c)


# top-left label showing absolute start time
label = pg.LabelItem(justify="left")

# ----------------for highlighting clicked channel trace---------------
def select_channel(i):
    """Highlight channel index i (index into chan_idx/curves), update label."""
    global selected_idx

    if i is None or i < 0 or i >= len(chan_idx) or curves[i] is None:
        return

    # unselect previous
    if selected_idx is not None and curves[selected_idx] is not None:
        curves[selected_idx].setPen(base_pen)
        curves[selected_idx].setZValue(0)

    # select new
    selected_idx = i
    curves[i].setPen(sel_pen)
    curves[i].setZValue(10)

    ch = int(chan_idx[i])
    selected_label.setText(f"<b>Selected channel: {ch}</b>")
# --------------------------------------------------------------------


win.addItem(label, row=1, col=0)


start_s = 0.0

def clamp_start(x):
    max_start = max(0.0, duration_s - window_s)
    return float(np.clip(x, 0.0, max_start))

def jump_dialog():
    global start_s
    # default text shows current start time
    text, ok = QtWidgets.QInputDialog.getText(
        win,
        "Jump to time",
        "Enter time in seconds:",
        text=f"{start_s:.3f}"
    )
    if not ok:
        return
    try:
        t = float(text)
    except ValueError:
        return
    start_s = clamp_start(t)
    update_plot()

def update_plot():
    global start_s
    t, X = buf.get(start_s, window_s)

    # Update only curve data (fast). X is (samples, channels_in_chan_idx)
    for i, c in enumerate(curves):
        if c is None:
            continue
        y = X[:, i] + offsets[i]
        c.setData(t, y)

    label.setText(f"start = {start_s:.3f} s    window = {window_s:.3f} s    decim={decim}")

def step(delta_s):
    global start_s
    start_s = clamp_start(start_s + delta_s)
    update_plot()

def keyPressEvent(ev):
    global window_s, start_s, spacing_factor, spacing, offsets, home_yrange, home_xrange
    k = ev.key()
    if k in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
        step(+step_s)
    elif k in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
        step(-step_s)
    elif k in (QtCore.Qt.Key_Home,):
        start_s = 0.0
        update_plot()
    elif k in (QtCore.Qt.Key_End,):
        start_s = clamp_start(duration_s - window_s)
        update_plot()
    elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
        window_s = max(0.25, window_s / 1.5)
        plot.setXRange(0.0, float(window_s), padding=0)
        vb.setLimits(xMin=0.0, xMax=float(window_s))  # if you're using limits
        start_s = clamp_start(start_s)
        update_home_ranges()
        update_plot()
    elif k in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
        window_s = min(60.0, window_s * 1.5)
        plot.setXRange(0.0, float(window_s), padding=0)
        vb.setLimits(xMin=0.0, xMax=float(window_s))
        start_s = clamp_start(start_s)
        update_home_ranges()
        update_plot()
    elif k == QtCore.Qt.Key_X:
        spacing_factor *= 1.3
        spacing = float(spacing_factor * typical)
        offsets = (np.arange(len(chan_idx))[::-1] * spacing).astype(np.float64)
        plot.setYRange(float(offsets[-1] - spacing), float(offsets[0] + spacing), padding=0)
        update_home_ranges()
        update_y_axis_labels(step=2, start=0)
        update_plot()
    elif k == QtCore.Qt.Key_Z:
        spacing_factor /= 1.3
        spacing = float(spacing_factor * typical)
        offsets = (np.arange(len(chan_idx))[::-1] * spacing).astype(np.float64)
        plot.setYRange(float(offsets[-1] - spacing), float(offsets[0] + spacing), padding=0)
        update_home_ranges()
        update_y_axis_labels(step=2, start=0)
        update_plot()
    elif k == QtCore.Qt.Key_J:
        jump_dialog()
    elif k == QtCore.Qt.Key_V:          # toggle box zoom mode
        if zoom_mode:
            exit_zoom_mode()
        else:
            enter_zoom_mode()
    elif k == QtCore.Qt.Key_Escape:     # reset view + exit zoom
        reset_view_and_exit_zoom()
    else:
        ev.ignore()
        return
    ev.accept()

# Install key handler on the window
win.keyPressEvent = keyPressEvent

# Mouse wheel scroll
def wheelEvent(ev):
    delta = ev.angleDelta().y()
    if delta > 0:
        step(-step_s)
    elif delta < 0:
        step(+step_s)
    ev.accept()

win.wheelEvent = wheelEvent

update_plot()
QtWidgets.QApplication.instance().exec_()
