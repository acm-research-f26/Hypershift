$ErrorActionPreference = 'Stop'

$PaperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$MainFile = Join-Path $PaperDirectory 'main.tex'

Push-Location -LiteralPath $PaperDirectory
try {
    & nvim-qt -- $MainFile
} finally {
    Pop-Location
}
