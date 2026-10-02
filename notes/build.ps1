$ErrorActionPreference = 'Stop'

$PaperDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    throw 'pdflatex was not found on PATH. Use your installed MiKTeX or TeX Live distribution.'
}

Push-Location -LiteralPath $PaperDirectory
try {
    foreach ($Pass in 1..3) {
        & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX compilation failed on pass $Pass (exit code $LASTEXITCODE). See main.log."
        }
    }

    Write-Host "Built $(Join-Path $PaperDirectory 'main.pdf')"
} finally {
    Pop-Location
}
