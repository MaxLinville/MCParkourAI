#!/usr/bin/env python3
# filepath: /home/wslmax/classwork/ECE4524/MCParkourAI/minecraft_escape.py

import subprocess
import time
import json

def get_minecraft_windows():
    """Get a list of all Minecraft windows with their handles and titles"""
    powershell_command = '''
    powershell.exe -Command "
    Add-Type @'
    using System;
    using System.Runtime.InteropServices;
    using System.Collections.Generic;
    using System.Text;
    
    public class WindowFinder {
        [DllImport(\\"user32.dll\\")]
        public static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);
        
        [DllImport(\\"user32.dll\\")]
        public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
        
        [DllImport(\\"user32.dll\\")]
        public static extern bool IsWindowVisible(IntPtr hWnd);
        
        public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
        
        public static List<dynamic> FindMinecraftWindows() {
            List<dynamic> results = new List<dynamic>();
            EnumWindows(new EnumWindowsProc((hWnd, lParam) => {
                StringBuilder sb = new StringBuilder(256);
                if (IsWindowVisible(hWnd) && GetWindowText(hWnd, sb, sb.Capacity) > 0) {
                    string title = sb.ToString();
                    if (title.Contains(\\"Minecraft\\")) {
                        results.Add(new { Handle = hWnd.ToInt64(), Title = title });
                    }
                }
                return true;
            }), IntPtr.Zero);
            return results;
        }
    }
'@
    
    [WindowFinder]::FindMinecraftWindows() | ConvertTo-Json
    "
    '''
    
    try:
        result = subprocess.run(powershell_command, capture_output=True, text=True, shell=True)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
        return []
    except Exception as e:
        print(f"Error finding Minecraft windows: {e}")
        return []

def focus_window_and_press_escape(window_handle):
    """Focus a specific window and press the Escape key"""
    powershell_command = f'''
    powershell.exe -Command "
    Add-Type @'
    using System;
    using System.Runtime.InteropServices;
    
    public class WindowController {{
        [DllImport(\\"user32.dll\\")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetForegroundWindow(IntPtr hWnd);
        
        [DllImport(\\"user32.dll\\")]
        public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, int dwExtraInfo);
        
        public const byte VK_ESCAPE = 0x1B;
        public const uint KEYEVENTF_KEYDOWN = 0x0000;
        public const uint KEYEVENTF_KEYUP = 0x0002;
        
        public static void PressEscape(IntPtr hWnd) {{
            SetForegroundWindow(hWnd);
            System.Threading.Thread.Sleep(200);  // Wait for window to come to foreground
            keybd_event(VK_ESCAPE, 0, KEYEVENTF_KEYDOWN, 0);
            System.Threading.Thread.Sleep(50);
            keybd_event(VK_ESCAPE, 0, KEYEVENTF_KEYUP, 0);
        }}
    }}
'@
    
    [WindowController]::PressEscape([IntPtr]::new({window_handle}))
    "
    '''
    
    try:
        subprocess.run(powershell_command, shell=True, check=True)
        return True
    except Exception as e:
        print(f"Error interacting with window: {e}")
        return False

def autofocus_minecraft():
    print("Finding Minecraft windows...")
    minecraft_windows = get_minecraft_windows()
    
    if not minecraft_windows:
        print("No Minecraft windows found.")
        return
    
    print(f"Found {len(minecraft_windows)} Minecraft window(s).")
    
    for i, window in enumerate(minecraft_windows, 1):
        title = window.get('Title', 'Unknown')
        handle = window.get('Handle')
        
        print(f"Processing window {i}/{len(minecraft_windows)}: {title}")
        
        if handle:
            success = focus_window_and_press_escape(handle)
            if success:
                print(f"Successfully pressed Escape on window: {title}")
            else:
                print(f"Failed to interact with window: {title}")
        
        # Wait a moment before moving to the next window
        time.sleep(0.1)
    
    print("All Minecraft windows processed.")

if __name__ == "__main__":
    autofocus_minecraft()