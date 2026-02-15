$ErrorActionPreference = 'Stop'
cd $PSScriptRoot

# Identify notebooks that look like placeholders (very small)
$small = Get-ChildItem -Recurse -Filter *.ipynb | Where-Object { $_.Length -lt 2000 } | ForEach-Object { $_.FullName.Substring((Get-Location).Path.Length+1).Replace('\\','/') }

Write-Host "Small/placeholder notebooks: $($small.Count)" -ForegroundColor Cyan

foreach ($p in $small) {
  Write-Host "Restoring $p from 063d281..." -ForegroundColor Yellow
  git checkout 063d281 -- $p
}

Write-Host "Done." -ForegroundColor Green
