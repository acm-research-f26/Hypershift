$ErrorActionPreference = 'Stop'
$taskWorkspace = 'C:\Users\rsand\OneDrive\Documents\Hypershift'
$fontDirectory = Join-Path $taskWorkspace 'presentations\build\canva_dataset\poppins\fonts'
$sourcePath = Join-Path $taskWorkspace 'docs\largecap_retraining_20261010\Hypershift_Large_Company_Retraining_Results.docx'
$pdfPath = Join-Path $taskWorkspace 'docs\largecap_retraining_20261010\Hypershift_Large_Company_Retraining_Results.pdf'
if (-not (Test-Path -LiteralPath $sourcePath)) { throw 'The completed report document is not present' }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class LargecapReportFonts {
 [DllImport("gdi32.dll",CharSet=CharSet.Unicode)] public static extern int AddFontResourceEx(string path,uint flags,IntPtr reserved);
 [DllImport("gdi32.dll",CharSet=CharSet.Unicode)] public static extern bool RemoveFontResourceEx(string path,uint flags,IntPtr reserved);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern IntPtr SendMessageTimeout(IntPtr window,uint message,IntPtr wparam,IntPtr lparam,uint flags,uint timeout,out IntPtr result);
}
'@
$loadedFonts = @()
$wordApp = $null
$sourceDocument = $null
try {
 foreach ($name in @('Poppins-Regular.ttf','Poppins-Bold.ttf','Poppins-Black.ttf')) {
  $fontPath = Join-Path $fontDirectory $name
  if ([LargecapReportFonts]::AddFontResourceEx($fontPath,0,[IntPtr]::Zero) -eq 0) { throw "Could not temporarily load $name" }
  $loadedFonts += $fontPath
 }
 $fontMessageResult = [IntPtr]::Zero
 [LargecapReportFonts]::SendMessageTimeout([IntPtr]0xffff,0x001D,[IntPtr]::Zero,[IntPtr]::Zero,2,1000,[ref]$fontMessageResult) | Out-Null
 $wordApp = New-Object -ComObject Word.Application
 $wordApp.Visible = $false
 $wordApp.DisplayAlerts = 0
 $sourceDocument = $wordApp.Documents.Open($sourcePath,$false,$true,$false)
 $sourceDocument.Repaginate()
 $pageCount = $sourceDocument.ComputeStatistics(2)
 $sourceDocument.ExportAsFixedFormat($pdfPath,17)
 Write-Output "Rendered completed report in Word: $pageCount pages"
} finally {
 if ($null -ne $sourceDocument) { $sourceDocument.Close(0); [Runtime.InteropServices.Marshal]::ReleaseComObject($sourceDocument) | Out-Null }
 if ($null -ne $wordApp) { if ($wordApp.Documents.Count -eq 0) { $wordApp.Quit() }; [Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null }
 foreach ($fontPath in $loadedFonts) { [LargecapReportFonts]::RemoveFontResourceEx($fontPath,0,[IntPtr]::Zero) | Out-Null }
 $fontMessageResult = [IntPtr]::Zero
 [LargecapReportFonts]::SendMessageTimeout([IntPtr]0xffff,0x001D,[IntPtr]::Zero,[IntPtr]::Zero,2,1000,[ref]$fontMessageResult) | Out-Null
}
