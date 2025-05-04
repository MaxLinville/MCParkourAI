import ctypes
from ctypes import wintypes
import sys

user32 = ctypes.WinDLL('user32', use_last_error=True)

# Types & constants
EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HWND,
    wintypes.LPARAM
)
GetWindowTextW = user32.GetWindowTextW
GetWindowTextLengthW = user32.GetWindowTextLengthW
IsWindowVisible = user32.IsWindowVisible
SystemParametersInfo = user32.SystemParametersInfoW
SPI_GETWORKAREA = 0x0030
SetWindowPos = user32.SetWindowPos
SWP_NOZORDER = 0x0004

# Storage for handles
hwnds = []

def foreach_window(hwnd, lParam):
    if not IsWindowVisible(hwnd):
        return True
    length = GetWindowTextLengthW(hwnd)
    if not length:
        return True
    buf = ctypes.create_unicode_buffer(length + 1)
    GetWindowTextW(hwnd, buf, length + 1)
    title = buf.value
    if 'Minecraft' in title:
        hwnds.append(hwnd)
    return True

# 1) Enumerate all Minecraft windows
EnumWindows(EnumWindowsProc(foreach_window), 0)
N = len(hwnds)
if N == 0:
    print("No Minecraft windows found.")
    sys.exit(1)

# 2) Get the work area (excludes taskbar)
# RECT: left, top, right, bottom
class RECT(ctypes.Structure):
    _fields_ = [('left', wintypes.LONG),
                ('top', wintypes.LONG),
                ('right', wintypes.LONG),
                ('bottom', wintypes.LONG)]
work_area = RECT()
if not SystemParametersInfo(SPI_GETWORKAREA, 0, ctypes.byref(work_area), 0):
    raise ctypes.WinError(ctypes.get_last_error())
screen_width  = work_area.right  - work_area.left
screen_height = work_area.bottom - work_area.top

# 3) Compute grid metrics (3 rows × 8 cols)
ROWS, COLS = 4, 4
cell_w = screen_width  // COLS
cell_h = screen_height // ROWS

# 4) Position each window
for idx, hwnd in enumerate(hwnds):
    r = idx // COLS
    c = idx %  COLS
    x = work_area.left + c * cell_w
    y = work_area.top  + r * cell_h
    # SetWindowPos(hwnd, hWndInsertAfter, X, Y, Width, Height, Flags)
    SetWindowPos(hwnd, None, x, y, cell_w, cell_h, SWP_NOZORDER)

print(f"Tiled {N} Minecraft windows into {ROWS}×{COLS} grid.")
