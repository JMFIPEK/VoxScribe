# ============================================================
#  Creates "VoxScribe" shortcuts on the Desktop and in the Start menu
# ============================================================
#  Run once: right-click this file > "Run with PowerShell"
#  (or:  powershell -ExecutionPolicy Bypass -File Create-Shortcuts.ps1)
#
#  For a taskbar icon, right-click VoxScribe in the Start menu > "Pin to
#  taskbar" - Windows doesn't let scripts pin anything themselves.
#
#  Both shortcuts start VoxScribe.bat in this repo folder - re-run this
#  script if you move the folder. Safe to run again anytime; it overwrites
#  the existing shortcuts.
# ============================================================

$ErrorActionPreference = 'Stop'

# Must match APP_ID in gui_qt.py: Windows groups the running window under a
# pinned shortcut only if both carry the same AppUserModelID.
$appId = 'voxscribe.app'

$repo   = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $repo 'VoxScribe.bat'
$icon   = Join-Path $repo 'Logo.ico'

if (-not (Test-Path $target)) {
    throw "VoxScribe.bat not found next to this script ($target)"
}

# WScript.Shell can't set the AppUserModelID, so write it through the
# shortcut's property store (PKEY_AppUserModel_ID).
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class ShortcutAppId
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct PropertyKey { public Guid FormatId; public int PropertyId; }

    [StructLayout(LayoutKind.Sequential)]
    struct PropVariant { public ushort vt; ushort r1, r2, r3; public IntPtr value; IntPtr pad; }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore
    {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHGetPropertyStoreFromParsingName(
        string path, IntPtr bindCtx, int flags, ref Guid iid,
        [MarshalAs(UnmanagedType.Interface)] out IPropertyStore store);

    public static void Set(string lnkPath, string appId)
    {
        Guid iid = typeof(IPropertyStore).GUID;
        IPropertyStore store;
        SHGetPropertyStoreFromParsingName(lnkPath, IntPtr.Zero, 2 /* GPS_READWRITE */,
                                          ref iid, out store);
        var key = new PropertyKey {
            FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), PropertyId = 5 };
        var value = new PropVariant { vt = 31 /* VT_LPWSTR */,
                                      value = Marshal.StringToCoTaskMemUni(appId) };
        try {
            store.SetValue(ref key, ref value);
            store.Commit();
        } finally {
            Marshal.FreeCoTaskMem(value.value);
            Marshal.ReleaseComObject(store);
        }
    }
}
'@

$folders = @(
    [Environment]::GetFolderPath('Desktop'),
    [Environment]::GetFolderPath('Programs')   # Start menu
)

$shell = New-Object -ComObject WScript.Shell
foreach ($folder in $folders) {
    $lnkPath = Join-Path $folder 'VoxScribe.lnk'
    $sc = $shell.CreateShortcut($lnkPath)
    # Through cmd.exe, since Windows won't pin a shortcut to a .bat directly
    $sc.TargetPath       = Join-Path $env:SystemRoot 'System32\cmd.exe'
    $sc.Arguments        = "/c `"$target`""
    $sc.WorkingDirectory = $repo
    $sc.Description      = 'VoxScribe - local audio recording and transcription'
    $sc.WindowStyle      = 7   # the launcher window starts minimized
    if (Test-Path $icon) { $sc.IconLocation = $icon }
    $sc.Save()
    [ShortcutAppId]::Set($lnkPath, $appId)
    Write-Host "Created $lnkPath" -ForegroundColor Green
}
Write-Host "To get a taskbar icon: Start menu > right-click VoxScribe > Pin to taskbar."
