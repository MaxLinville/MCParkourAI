<#
.SYNOPSIS
  Closes all running instances of Minecraft (Launcher & in-game).
#>

# 1) Stop the official Minecraft Launcher (if it’s running)
Stop-Process -Name MinecraftLauncher -ErrorAction SilentlyContinue -Force

# 2) Stop any javaw.exe processes whose window title contains “Minecraft”
Get-Process javaw -ErrorAction SilentlyContinue |
  Where-Object { $_.MainWindowTitle -like '*Minecraft*' } |
  ForEach-Object {
    Write-Host "Stopping Minecraft session (PID $($_.Id))..."
    Stop-Process -Id $_.Id -Force
  }

Write-Host "All Minecraft processes have been terminated."
