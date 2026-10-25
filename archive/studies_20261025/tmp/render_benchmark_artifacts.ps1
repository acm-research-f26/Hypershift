$ErrorActionPreference = 'Stop'
$taskWorkspace = 'C:\Users\rsand\OneDrive\Documents\Hypershift'
$taskWordApp = $null
$taskDocument = $null
try {
    $taskWordApp = New-Object -ComObject Word.Application
    $taskWordApp.Visible = $false
    $taskWordApp.DisplayAlerts = 0
    $taskDocument = $taskWordApp.Documents.Open((Join-Path $taskWorkspace 'docs\largecap_retraining_20261010\Hypershift_Large_Company_Retraining_Results_Benchmark_Corrected.docx'), $false, $true, $false)
    $taskDocument.Repaginate()
    $taskDocument.ExportAsFixedFormat((Join-Path $taskWorkspace 'tmp\benchmark_doc_corrected.pdf'), 17)
    Write-Output "Corrected document pages: $($taskDocument.ComputeStatistics(2))"
} finally {
    if ($null -ne $taskDocument) { $taskDocument.Close(0) }
    if ($null -ne $taskWordApp -and $taskWordApp.Documents.Count -eq 0) { $taskWordApp.Quit() }
}
